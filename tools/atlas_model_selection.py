#!/usr/bin/env python3
"""Choose verified Atlas programs by native model cycles, breaking ties by candidate order."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


# Compiler policy: candidate pools and search work limits.
RANDOM_SEEDS = tuple(range(8))
MREG_SEEDS = tuple(range(8))
MREG_RANK_CHOICES = tuple(range(32))
MREG_VECTOR_LENGTH = 64
NEIGHBOR_SHUFFLE_SEED = 0
GROUP_MAX_NODES = 4
BATCH_WIDTH = 12
API_INITIAL_ATTEMPTS = 32
API_GROUP_ATTEMPTS = 0
FULL_INITIAL_ATTEMPTS = 18
FULL_GROUP_ATTEMPTS = 134
INITIAL_WAIT_PLACEMENT = "source"
CURVE_GROUP_WAIT_PLACEMENTS = ("default", "source")


@dataclass(frozen=True)
class Candidate:
    random_seed: int | None = None
    wait_placement: str = "default"
    virtual_orders: tuple[tuple[int, ...], ...] = ()
    mreg_ranks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.random_seed is not None and (self.random_seed < 0):
            raise ValueError("random-seed must be nonnegative")
        if self.virtual_orders and self.random_seed is not None:
            raise ValueError("explicit virtual orders exclude random-seed")
        if any(rank < 0 or rank > 0xFFFFFFFF for rank in self.mreg_ranks):
            raise ValueError("MREG ranks must be unsigned 32-bit integers")
        if self.wait_placement not in ("default", "source"):
            raise ValueError("wait-placement must be default or source")


def _option_path(path: Path) -> str:
    return '"' + str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def evidence_option(model_facts: Path | None = None) -> list[str]:
    if model_facts is None:
        return []  # A library caller may supply IR with selected evidence.
    directory = Path(model_facts).resolve()
    if not directory.is_dir():
        raise ValueError("--model-facts must name a bundle directory")
    for name in ("facts.lock.json", "model-timing.json"):
        if not (directory / name).is_file():
            raise ValueError(f"model facts bundle requires {name}")
    return ["--select-atlas-model-evidence=facts=" + _option_path(directory)]


def candidate_options(candidate: Candidate, provider: str, spill: bool, *, graph_output: bool = False) -> list[str]:
    options = [f"provider={provider}", f"spill={str(spill).lower()}", "allocation-report=allocation.json",
               "pack-relayout-rows=2", "wait-reused-dma-slots=true"]
    for enabled, option in (
        (candidate.virtual_orders, "order-file=order.json"),
        (graph_output, "graph-output=graph.json"),
        (candidate.random_seed is not None, f"random-seed={candidate.random_seed}"),
        (candidate.mreg_ranks, "mreg-ranks='" + ",".join(map(str, candidate.mreg_ranks)) + "'"),
        (candidate.wait_placement != "default", "wait-placement=" + candidate.wait_placement),
    ):
        if enabled:
            options.append(option)
    return ["--atlas-compile-candidate=" + " ".join(options)]


def group_order_neighbors(graphs: list[dict], orders: tuple[tuple[int, ...], ...],
                          max_nodes: int = GROUP_MAX_NODES):
    """Move a node together with the dependencies crossed by its insertion."""
    moves = []
    seen = set()
    parent_edges = [set(zip((-1,) + order, order + (-2,))) for order in orders]
    for block, (graph, order) in enumerate(zip(graphs, orders)):
        positions = {node: index for index, node in enumerate(order)}
        successors = [[] for _ in order]
        for node, preds in enumerate(graph["preds"]):
            for pred in preds:
                successors[pred].append(node)
        for origin, node in enumerate(order):
            for target in range(len(order)):
                if origin == target:
                    continue
                backward = target < origin
                edges = graph["preds"] if backward else successors
                members, pending = {node}, [node]
                while pending and len(members) <= max_nodes:
                    for related in edges[pending.pop()]:
                        crosses = positions[related] >= target if backward else positions[related] <= target
                        if crosses and related not in members:
                            members.add(related)
                            pending.append(related)
                if not 2 <= len(members) <= max_nodes:
                    continue
                remaining = [other for other in order if other not in members]
                boundary = target if backward else target + 1
                insertion = sum(positions[other] < boundary for other in remaining)
                group = [other for other in order if other in members]
                moved = tuple(remaining[:insertion] + group + remaining[insertion:])
                rank = {other: index for index, other in enumerate(moved)}
                if any(rank[pred] >= rank[other] for other in moved for pred in graph["preds"][other]):
                    continue
                # Added adjacencies identify permutations without retaining every full order.
                signature = (block, tuple(edge for edge in zip((-1,) + moved, moved + (-2,))
                                          if edge not in parent_edges[block]))
                if moved != order and signature not in seen:
                    seen.add(signature)
                    moves.append((block, tuple(group), insertion))
    random.Random(NEIGHBOR_SHUFFLE_SEED).shuffle(moves)
    for block, group, insertion in moves:
        members = set(group)
        remaining = [node for node in orders[block] if node not in members]
        moved = tuple(remaining[:insertion]) + group + tuple(remaining[insertion:])
        yield orders[:block] + (moved,) + orders[block + 1:]


@dataclass(frozen=True)
class SelectionConfig:
    provider: str
    profile: str | None
    model: Path | None
    model_regions: tuple[Path, ...]
    spill: bool
    spill_mxu: bool
    boundary: str
    guarded_pack: bool
    model_facts: Path | None
    max_candidates: int
    random_seeds: tuple[int, ...]
    workers: int
    mreg_candidates: tuple[tuple[int, ...], ...] | None
    search_group_evaluations: int

    def resolved(self) -> SelectionConfig:
        profile = self.profile
        if profile is None:
            profile = "curve" if self.provider == "model" else "fifo43"
        if self.provider not in ("builtin", "model") or profile not in ("fifo43", "curve"):
            raise ValueError("unsupported provider or memory profile")
        if self.max_candidates < 1 or self.workers < 1 or self.boundary not in ("boundary", "none"):
            raise ValueError("candidate and worker counts must be positive and boundary must be boundary or none")
        if self.search_group_evaluations < 0:
            raise ValueError("search-group-evaluations must be nonnegative")
        mreg_candidates = self.mreg_candidates
        if mreg_candidates is None:
            mreg_candidates = tuple(tuple(random.Random(seed).choices(
                MREG_RANK_CHOICES, k=MREG_VECTOR_LENGTH))
                for seed in MREG_SEEDS)
        return replace(self, profile=profile, mreg_candidates=mreg_candidates)

    def policy(self) -> dict:
        return {"effective": dict(
            provider=self.provider, profile=self.profile, spill=self.spill, spill_mxu=self.spill_mxu,
            boundary=self.boundary, guarded_pack=self.guarded_pack, max_candidates=self.max_candidates,
            random_seeds=list(self.random_seeds), workers=self.workers,
            mreg_candidates=[list(ranks) for ranks in self.mreg_candidates],
            search_group_evaluations=self.search_group_evaluations,
        )}


class Selection:
    def __init__(self, source: Path, output_dir: Path, atlas_bin_dir: Path,
                 config: SelectionConfig, scorer: Callable | None, effective_policy: dict):
        self.config = config
        self.scorer = scorer
        self.source = Path(source).resolve(strict=True)
        self.output = Path(output_dir).resolve()
        self.binaries = Path(atlas_bin_dir).resolve()
        if self.output.exists() and (not self.output.is_dir() or any(self.output.iterdir())):
            raise ValueError("output directory must be empty to prevent stale artifacts")
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.commands, self.logs = [], []
        self.log_lock, self.score_lock = threading.Lock(), threading.Lock()
        self.receipt = {
            "schema": "atlas.model_selection.v1", "provider": config.provider, "profile": config.profile,
            "guarded_pack": config.guarded_pack,
            "max_candidates": config.max_candidates, "workers": config.workers,
            "search_group_evaluations": config.search_group_evaluations,
            "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "search_policy": effective_policy,
            "candidates": []}
        self.standard_scorer = scorer is None
        self.prepared = self.output / "1-virtual.mlir"
        self.seen: set[Candidate] = set()
        # Profiles must not share scores; model and regions are fixed for this selection.
        self.score_cache: dict[tuple[str, tuple[int, ...]], int | Future] = {}
        self.score_executor: ProcessPoolExecutor | None = None
        self.best: dict | None = None

    def initialize_scorer(self) -> None:
        if self.standard_scorer:
            if self.config.model is None:
                raise ValueError("model scoring requires --npu-model")
            from atlas_model_scorer import ModelScorer
            self.scorer = ModelScorer(self.config.model, self.config.model_regions)

    def prepare(self) -> None:
        model_facts = self.config.model_facts
        if model_facts is None and self.config.model is not None:
            model_facts = self.output / "model-facts"
            self.command([sys.executable, "-B", str(Path(__file__).with_name("atlas_model_facts.py")),
                          "--npu-model", str(self.config.model), "--out", str(model_facts)], self.output)
        options = evidence_option(model_facts)
        options.append("--atlas-compile-prepare=" +
                       f"spill-mxu-state={str(self.config.spill_mxu).lower()} boundary={self.config.boundary}")
        if self.config.guarded_pack:
            options[-1] += " guarded-pack=true"
        self.command([str(self.binaries / "atlas-opt"), str(self.source), *options,
                      "--mlir-print-op-on-diagnostic=false", "-o", str(self.prepared)], self.output)

    def command(self, argv: list[str], directory: Path, stdout: Path | None = None) -> str:
        rendered = f"cd {shlex.quote(str(directory))} && " + shlex.join(argv)
        if stdout:
            rendered += f" > {shlex.quote(str(stdout))}"
        with self.log_lock:
            self.commands.append(rendered)
            (self.output / "commands.txt").write_text("\n".join(self.commands) + "\n")
        try:
            result = subprocess.run(argv, cwd=directory, capture_output=True, text=True,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        except OSError as error:
            result = subprocess.CompletedProcess(argv, 1, "", str(error))
        with self.log_lock:
            self.logs.append(f"$ {rendered}\n{result.stderr}exit {result.returncode}\n")
            (self.output / "log.txt").write_text("\n".join(self.logs))
        if result.returncode:
            raise ValueError(result.stderr.strip() or f"{argv[0]} exited {result.returncode}")
        if stdout:
            stdout.write_text(result.stdout)
        return result.stdout

    def cached_score(self, program: dict, scoring_profile: str) -> tuple[int | Future, bool]:
        key = scoring_profile, tuple(instruction["word_u32"] for instruction in program["instructions"])
        with self.score_lock:
            if self.standard_scorer:
                profiles = self.receipt.setdefault("scoring_configurations", {})
                if scoring_profile not in profiles:
                    profiles[scoring_profile] = self.score_executor.submit(
                        self.scorer.profile_receipt, program, scoring_profile).result()
            hit = key in self.score_cache
            if not hit:
                self.score_cache[key] = (self.score_executor.submit(self.scorer, program, scoring_profile)
                                         if self.score_executor else self.scorer(program, profile=scoring_profile))
            return self.score_cache[key], hit

    def compile_candidate(self, candidate: Candidate, row: dict) -> None:
        directory = self.output / row["directory"]
        before = time.monotonic()
        stage = "compile"
        try:
            if candidate.virtual_orders:
                (directory / "order.json").write_text(json.dumps(candidate.virtual_orders) + "\n")
            self.command([str(self.binaries / "atlas-opt"), str(self.prepared),
                          *candidate_options(candidate, self.config.provider, self.config.spill,
                                             graph_output=bool(self.config.search_group_evaluations)),
                          "--mlir-print-op-on-diagnostic=false", "-o", "final.mlir"], directory)
            stage = "emit"
            program = json.loads(self.command([str(self.binaries / "atlas-emit"), "final.mlir", "--program-json"],
                                              directory, directory / "program.json"))
            words = tuple(instruction["word_u32"] for instruction in program["instructions"])
            (directory / "program.hex").write_text("".join(f"{word:08x}\n" for word in words))
            if self.scorer is not None:
                stage = "score"
                cached, row["score_cache_hit"] = self.cached_score(program, self.config.profile)
                row["cycles"] = cached.result() if isinstance(cached, Future) else cached
            row.update(status="PASS", words=len(words), program_sha256=hashlib.sha256(
                b"".join(word.to_bytes(4, "little") for word in words)).hexdigest())
        except (OSError, ValueError, RuntimeError) as error:
            row.update(status="FAIL", error=str(error), error_stage=stage)
        finally:
            row["elapsed_seconds"] = time.monotonic() - before

    def fail(self, message: str) -> None:
        (self.output / "selection.json").write_text(json.dumps(self.receipt, indent=2) + "\n")
        raise ValueError(message)

    def evaluate_batch(self, specifications: list[tuple[Candidate, str]], limit: int | None = None) -> None:
        if limit is None:
            limit = self.config.max_candidates
        pending, before = [], len(self.receipt["candidates"])
        for candidate, category in specifications:
            if candidate in self.seen or len(self.receipt["candidates"]) >= limit:
                continue
            index = len(self.receipt["candidates"])
            directory = self.output / "candidates" / f"{index:03d}-{category}"
            directory.mkdir(parents=True)
            row = {"index": index, "class": category, "options": asdict(candidate),
                   "directory": str(directory.relative_to(self.output))}
            pending.append((candidate, row))
            self.seen.add(candidate)
            self.receipt["candidates"].append(row)
        with ThreadPoolExecutor(max_workers=min(self.config.workers, max(1, len(pending)))) as executor:
            futures = [executor.submit(self.compile_candidate, candidate, row) for candidate, row in pending]
            for future in futures:
                future.result()
        for row in self.receipt["candidates"][before:]:
            # Native faults stop the search instead of hiding an invalid program.
            if row["status"] == "FAIL" and row["error_stage"] == "score":
                self.fail(f"candidate {row['index']} failed model scoring: {row['error']}")
        passing = [row for row in self.receipt["candidates"] if row["status"] == "PASS"]
        # Assign canonical indices before parallel work to preserve deterministic ties.
        self.best = min(passing, key=lambda row: (row.get("cycles", 0), row["index"]), default=None)

    def winner(self) -> Candidate:
        return Candidate(**self.best["options"])  # asdict preserves tuple fields.

    def search_initial(self) -> None:
        self.evaluate_batch([(Candidate(), "default")])
        initial = [(Candidate(random_seed=seed), "random-seed") for seed in self.config.random_seeds]
        self.evaluate_batch(initial, limit=self.config.max_candidates)
        if self.best is None:
            self.fail("no legal candidate: every initial candidate was refused")

        self.evaluate_batch([(replace(self.winner(), wait_placement=INITIAL_WAIT_PLACEMENT),
                              "wait-placement")], limit=self.config.max_candidates)
        register_base = self.winner()
        self.receipt["mreg_pool_size"] = len(self.config.mreg_candidates)
        self.evaluate_batch([(replace(register_base, mreg_ranks=tuple(ranks)), "mreg-ranks")
                             for ranks in self.config.mreg_candidates])

    @staticmethod
    def group_candidates(base: Candidate, orders, profile: str):
        placements = (CURVE_GROUP_WAIT_PLACEMENTS
                      if profile == "curve" else (base.wait_placement,))
        for order in orders:
            for placement in placements:
                yield replace(base, virtual_orders=order, wait_placement=placement)

    def search_groups(self) -> None:
        search_started = time.monotonic()
        initial_count = len(self.receipt["candidates"])
        limit = initial_count + self.config.search_group_evaluations
        phase = self.receipt["group_search"] = {
            "start_index": self.best["index"], "accepted_indices": []}
        while len(self.receipt["candidates"]) < limit:
            parent = self.best
            graphs = json.loads((self.output / parent["directory"] / "graph.json").read_text())
            orders = tuple(tuple(graph["order"]) for graph in graphs)
            base = replace(self.winner(), random_seed=None, virtual_orders=orders)
            self.seen.add(base)
            proposed = group_order_neighbors(graphs, orders)
            pending = self.group_candidates(base, proposed, self.config.profile)
            while len(self.receipt["candidates"]) < limit:
                batch = []
                width = min(BATCH_WIDTH, limit - len(self.receipt["candidates"]))
                while len(batch) < width:
                    candidate = next(pending, None)
                    if candidate is None:
                        break
                    if candidate not in self.seen and candidate not in batch:
                        batch.append(candidate)
                if not batch:
                    break
                before = len(self.receipt["candidates"])
                self.evaluate_batch([(candidate, "group-order") for candidate in batch], limit)
                for row in self.receipt["candidates"][before:]:
                    row["parent_index"] = parent["index"]
                if self.best["index"] != parent["index"]:
                    phase["accepted_indices"].append(self.best["index"])
                    break
            if self.best["index"] == parent["index"]:
                break
        phase.update(evaluations=len(self.receipt["candidates"]) - initial_count,
                     elapsed_seconds=time.monotonic() - search_started)

    def publish(self) -> dict:
        selected = self.output / self.best["directory"]
        for name in ("final.mlir", "allocation.json", "program.json", "program.hex"):
            shutil.copyfile(selected / name, self.output / name)
        self.receipt.update(selected_index=self.best["index"], selected_class=self.best["class"],
                            default_cycles=self.receipt["candidates"][0].get("cycles"),
                            selected_cycles=self.best.get("cycles"),
                            score_cache_entries=sum(key[0] == self.config.profile for key in self.score_cache),
                            score_cache_hits=sum(row.get("score_cache_hit", False)
                                                 for row in self.receipt["candidates"]),
                            elapsed_seconds=time.monotonic() - self.started)
        (self.output / "selection.json").write_text(json.dumps(self.receipt, indent=2) + "\n")
        return self.receipt

    def run(self) -> dict:
        self.initialize_scorer()
        self.prepare()
        # Keep model execution isolated from compiler threads.
        with ExitStack() as stack:
            if self.standard_scorer:
                self.score_executor = stack.enter_context(ProcessPoolExecutor(
                    max_workers=self.config.workers,
                    mp_context=multiprocessing.get_context("spawn")))
            self.search_initial()
            if self.config.search_group_evaluations:
                self.search_groups()
        return self.publish()


def select_program(source: Path, output_dir: Path, atlas_bin_dir: Path, *,
                   provider: str = "model",
                   profile: str | None = None, model: Path | None = None, model_regions: tuple[Path, ...] = (),
                   spill: bool = True, spill_mxu: bool = True,
                   boundary: str = "boundary",
                   guarded_pack: bool = False,
                   model_facts: Path | None = None,
                   max_candidates: int = API_INITIAL_ATTEMPTS,
                   random_seeds: tuple[int, ...] = RANDOM_SEEDS, workers: int = 12,
                   mreg_candidates: tuple[tuple[int, ...], ...] | None = (),
                   search_group_evaluations: int = API_GROUP_ATTEMPTS,
                   scorer: Callable | None = None) -> dict:
    """Verify candidates and publish the lowest-cycle program in canonical order."""
    config = SelectionConfig(
        provider=provider, profile=profile, model=model, model_regions=model_regions,
        spill=spill, spill_mxu=spill_mxu, boundary=boundary,
        guarded_pack=guarded_pack, model_facts=model_facts,
        max_candidates=max_candidates, random_seeds=random_seeds, workers=workers,
        mreg_candidates=mreg_candidates, search_group_evaluations=search_group_evaluations).resolved()
    effective_policy = config.policy()
    return Selection(source, output_dir, atlas_bin_dir, config, scorer, effective_policy).run()


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("builtin", "model"), default="builtin")
    parser.add_argument("--profile", choices=("fifo43", "curve"),
                        help="memory scoring profile (model default: curve)")
    parser.add_argument("--npu-model", type=Path, help="npu_model source tree used for native scoring")
    parser.add_argument("--model-region", type=Path, action="append", default=[],
                        help="DRAM region JSON with address, initial_hex and optional expected_hex")
    parser.add_argument("--model-facts", type=Path,
                        help="model facts directory (default: regenerate from --npu-model)")
    parser.add_argument("--search-budget", choices=("full", "single"), default="full",
                        help="full search (18 initial and 134 group attempts) or one default candidate")
    parser.add_argument("--max-candidates", type=int, default=FULL_INITIAL_ATTEMPTS,
                        help="initial attempt limit in full mode (default: 18)")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--bank-candidates", action=argparse.BooleanOptionalAction, default=True,
                        help="score eight legal MREG rank vectors in full mode")
    parser.add_argument("--search-group-evaluations", type=int, default=FULL_GROUP_ATTEMPTS,
                        help="group attempt limit in full mode (default: 134)")


def model_arguments(args: argparse.Namespace) -> dict:
    single = args.search_budget == "single"
    profile = args.profile or ("curve" if args.provider == "model" else "fifo43")
    return {
        "provider": args.provider, "profile": profile,
        "model": args.npu_model, "model_regions": tuple(args.model_region),
        "model_facts": args.model_facts, "workers": args.workers,
        "mreg_candidates": () if single or not args.bank_candidates else None,
        "max_candidates": 1 if single else args.max_candidates,
        "search_group_evaluations": 0 if single else args.search_group_evaluations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atlas-bin-dir", type=Path,
                        default=Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin")))
    parser.add_argument("--boundary", choices=("boundary", "none"), default="boundary")
    parser.add_argument("--no-spill", action="store_true", default=False)
    parser.add_argument("--guarded-pack", action="store_true", default=False)
    add_model_arguments(parser)
    parser.set_defaults(provider="model")
    args = parser.parse_args()
    try:
        receipt = select_program(args.source, args.output_dir, args.atlas_bin_dir,
                                 boundary=args.boundary, spill=not args.no_spill,
                                 guarded_pack=args.guarded_pack, **model_arguments(args))
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps({key: receipt[key] for key in ("selected_index", "selected_class", "default_cycles",
                                                  "selected_cycles", "elapsed_seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
