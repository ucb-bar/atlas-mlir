#!/usr/bin/env python3
"""Choose verified Atlas programs by native model cycles.

Candidate order is deterministic; ties use the earliest index regardless of workers.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from itertools import islice
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
    order_file: tuple[tuple[int, ...], ...] = ()
    mreg_ranks: tuple[int, ...] = ()


def _option_path(path: Path) -> str:
    return '"' + str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def evidence_option(model_facts: Path | None = None) -> list[str]:
    if model_facts is None:
        return []  # A library caller may supply IR with selected evidence.
    return ["--select-atlas-model-evidence=facts=" + _option_path(Path(model_facts))]


def candidate_options(candidate: Candidate, provider: str, spill: bool, *, graph_output: bool = False) -> list[str]:
    options = [f"provider={provider}", f"spill={str(spill).lower()}", "allocation-report=allocation.json",
               "pack-relayout-rows=2", "wait-reused-dma-slots=true"]
    for field in fields(candidate):
        value = getattr(candidate, field.name)
        if value == field.default:
            continue
        if field.name == "order_file":
            value = "order.json"
        elif isinstance(value, tuple):
            value = "'" + ",".join(map(str, value)) + "'"
        options.append(field.name.replace("_", "-") + "=" + str(value))
    if graph_output:
        options.append("graph-output=graph.json")
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
    provider: str = "model"
    profile: str | None = None
    model: Path | None = None
    model_regions: tuple[Path, ...] = ()
    spill: bool = True
    guarded_pack: bool = False
    model_facts: Path | None = None
    max_candidates: int = API_INITIAL_ATTEMPTS
    random_seeds: tuple[int, ...] = RANDOM_SEEDS
    workers: int = 12
    mreg_candidates: tuple[tuple[int, ...], ...] | None = ()
    search_group_evaluations: int = API_GROUP_ATTEMPTS

    def resolved(self) -> SelectionConfig:
        profile = self.profile
        if profile is None:
            profile = "curve" if self.provider == "model" else "fifo43"
        if self.max_candidates < 1:
            raise ValueError("max-candidates must be positive")
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if self.search_group_evaluations < 0:
            raise ValueError("search-group-evaluations must be nonnegative")
        mreg_candidates = self.mreg_candidates
        if mreg_candidates is None:
            mreg_candidates = tuple(tuple(random.Random(seed).choices(
                MREG_RANK_CHOICES, k=MREG_VECTOR_LENGTH))
                for seed in MREG_SEEDS)
        return replace(self, profile=profile, mreg_candidates=mreg_candidates)


class Selection:
    def __init__(self, source: Path, output_dir: Path, atlas_bin_dir: Path,
                 config: SelectionConfig, scorer: Callable | None):
        self.config = config
        self.scorer = scorer
        self.source = Path(source).resolve(strict=True)
        self.output = Path(output_dir).resolve()
        self.binaries = Path(atlas_bin_dir).resolve()
        if self.output.exists() and (not self.output.is_dir() or any(self.output.iterdir())):
            raise ValueError("output directory must be empty to prevent stale artifacts")
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.log_lock, self.score_lock = threading.Lock(), threading.Lock()
        self.result = {"profile": config.profile, "guarded_pack": config.guarded_pack, "candidates": []}
        self.standard_scorer = scorer is None
        self.prepared = self.output / "1-virtual.mlir"
        self.seen: set[Candidate] = set()
        # Model, profile, and regions are fixed for this selection.
        self.score_cache: dict[tuple[int, ...], int | Future] = {}
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
        options.append("--atlas-compile-prepare" + ("=guarded-pack=true" if self.config.guarded_pack else ""))
        self.command([str(self.binaries / "atlas-opt"), str(self.source), *options,
                      "--mlir-print-op-on-diagnostic=false", "-o", str(self.prepared)], self.output)

    def command(self, argv: list[str], directory: Path, stdout: Path | None = None) -> str:
        rendered = f"cd {shlex.quote(str(directory))} && " + shlex.join(argv)
        if stdout:
            rendered += f" > {shlex.quote(str(stdout))}"
        with self.log_lock, (self.output / "commands.txt").open("a") as stream:
            stream.write(rendered + "\n")
        try:
            result = subprocess.run(argv, cwd=directory, capture_output=True, text=True,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        except OSError as error:
            result = subprocess.CompletedProcess(argv, 1, "", str(error))
        with self.log_lock, (self.output / "log.txt").open("a") as stream:
            stream.write(f"$ {rendered}\n{result.stderr}exit {result.returncode}\n\n")
        if result.returncode:
            raise ValueError(result.stderr.strip() or f"{argv[0]} exited {result.returncode}")
        if stdout:
            stdout.write_text(result.stdout)
        return result.stdout

    def cached_score(self, program: dict) -> int | Future:
        key = tuple(instruction["word_u32"] for instruction in program["instructions"])
        with self.score_lock:
            if key not in self.score_cache:
                self.score_cache[key] = (self.score_executor.submit(self.scorer, program, self.config.profile)
                                         if self.score_executor else self.scorer(program, profile=self.config.profile))
            return self.score_cache[key]

    def compile_candidate(self, candidate: Candidate, row: dict) -> None:
        directory = self.output / row["directory"]
        stage = "compile"
        try:
            if candidate.order_file:
                (directory / "order.json").write_text(json.dumps(candidate.order_file) + "\n")
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
                cached = self.cached_score(program)
                row["cycles"] = cached.result() if isinstance(cached, Future) else cached
            row.update(status="PASS", program_sha256=hashlib.sha256(
                b"".join(word.to_bytes(4, "little") for word in words)).hexdigest())
        except (OSError, ValueError, RuntimeError) as error:
            row.update(status="FAIL", error=str(error), error_stage=stage)

    def fail(self, message: str) -> None:
        (self.output / "selection.json").write_text(json.dumps(self.result, indent=2) + "\n")
        raise ValueError(message)

    def evaluate_batch(self, specifications: list[tuple[Candidate, str]], limit: int | None = None) -> None:
        if limit is None:
            limit = self.config.max_candidates
        pending, before = [], len(self.result["candidates"])
        for candidate, category in specifications:
            if candidate in self.seen or len(self.result["candidates"]) >= limit:
                continue
            index = len(self.result["candidates"])
            directory = self.output / "candidates" / f"{index:03d}-{category}"
            directory.mkdir(parents=True)
            row = {"index": index, "class": category, "options": asdict(candidate),
                   "directory": str(directory.relative_to(self.output))}
            pending.append((candidate, row))
            self.seen.add(candidate)
            self.result["candidates"].append(row)
        with ThreadPoolExecutor(max_workers=min(self.config.workers, max(1, len(pending)))) as executor:
            futures = [executor.submit(self.compile_candidate, candidate, row) for candidate, row in pending]
            for future in futures:
                future.result()
        for row in self.result["candidates"][before:]:
            # Native faults stop the search instead of hiding an invalid program.
            if row["status"] == "FAIL" and row["error_stage"] == "score":
                self.fail(f"candidate {row['index']} failed model scoring: {row['error']}")
        passing = [row for row in self.result["candidates"] if row["status"] == "PASS"]
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
        self.evaluate_batch([(replace(register_base, mreg_ranks=tuple(ranks)), "mreg-ranks")
                             for ranks in self.config.mreg_candidates])

    def search_groups(self) -> None:
        initial_count = len(self.result["candidates"])
        limit = initial_count + self.config.search_group_evaluations
        phase = self.result["group_search"] = {
            "start_index": self.best["index"], "accepted_indices": []}
        while len(self.result["candidates"]) < limit:
            parent = self.best
            graphs = json.loads((self.output / parent["directory"] / "graph.json").read_text())
            orders = tuple(tuple(graph["order"]) for graph in graphs)
            base = replace(self.winner(), random_seed=None, order_file=orders)
            self.seen.add(base)
            placements = CURVE_GROUP_WAIT_PLACEMENTS if self.config.profile == "curve" else (base.wait_placement,)
            pending = (replace(base, order_file=order, wait_placement=placement)
                       for order in group_order_neighbors(graphs, orders) for placement in placements)
            while len(self.result["candidates"]) < limit:
                width = min(BATCH_WIDTH, limit - len(self.result["candidates"]))
                # Each neighbor and wait-placement sibling is unique in this stream.
                batch = list(islice((candidate for candidate in pending if candidate not in self.seen), width))
                if not batch:
                    break
                before = len(self.result["candidates"])
                self.evaluate_batch([(candidate, "group-order") for candidate in batch], limit)
                for row in self.result["candidates"][before:]:
                    row["parent_index"] = parent["index"]
                if self.best["index"] != parent["index"]:
                    phase["accepted_indices"].append(self.best["index"])
                    break
            if self.best["index"] == parent["index"]:
                break
        phase["evaluations"] = len(self.result["candidates"]) - initial_count

    def publish(self) -> dict:
        selected = self.output / self.best["directory"]
        for name in ("final.mlir", "allocation.json", "program.json", "program.hex"):
            shutil.copyfile(selected / name, self.output / name)
        self.result.update(selected_index=self.best["index"], selected_class=self.best["class"],
                            default_cycles=self.result["candidates"][0].get("cycles"),
                            selected_cycles=self.best.get("cycles"),
                            elapsed_seconds=time.monotonic() - self.started)
        (self.output / "selection.json").write_text(json.dumps(self.result, indent=2) + "\n")
        return self.result

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


def select_program(source: Path, output_dir: Path, atlas_bin_dir: Path,
                   config: SelectionConfig = SelectionConfig(), *, scorer: Callable | None = None) -> dict:
    """Verify candidates and publish the lowest-cycle program in canonical order."""
    return Selection(source, output_dir, atlas_bin_dir, config.resolved(), scorer).run()


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
                        help="full search (18 initial and 134 group attempts) or one default candidate; "
                             "deterministic candidate order, with ties going to the earliest index")
    parser.add_argument("--max-candidates", type=int, default=FULL_INITIAL_ATTEMPTS,
                        help="initial attempt limit in full mode (default: 18)")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--bank-candidates", action=argparse.BooleanOptionalAction, default=True,
                        help="score eight legal MREG rank vectors in full mode")
    parser.add_argument("--search-group-evaluations", type=int, default=FULL_GROUP_ATTEMPTS,
                        help="group attempt limit in full mode (default: 134)")


def model_arguments(args: argparse.Namespace) -> SelectionConfig:
    single = args.search_budget == "single"
    return SelectionConfig(
        provider=args.provider, profile=args.profile, model=args.npu_model,
        model_regions=tuple(args.model_region), model_facts=args.model_facts, workers=args.workers,
        guarded_pack=getattr(args, "guarded_pack", False),
        mreg_candidates=() if single or not args.bank_candidates else None,
        max_candidates=1 if single else args.max_candidates,
        search_group_evaluations=0 if single else args.search_group_evaluations)
