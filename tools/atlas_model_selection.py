#!/usr/bin/env python3
"""Choose a fully verified Atlas program using cold cycles from the selected npu_model.

The default is retained as candidate zero, exact curve ties use FIFO43 cycles
then candidate order, and builtin compiles once without importing the model. The bounded
search exposes existing lowering choices and optional legal order mutations.
"""

from __future__ import annotations

if __package__:
    from .atlas_model_diagnostics import ModelTimingEvidenceError
else:
    from atlas_model_diagnostics import ModelTimingEvidenceError

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import hashlib
import heapq
import json
from itertools import chain
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
SEARCH_BUDGETS = {
    "selection": dict(max_candidates=24, search_evaluations=0, search_group_evaluations=0,
                      search_patience=0, bank_candidates=False,
                      pack_reload_candidates=False, pack_preload_candidates=False),
    "full": dict(max_candidates=32, search_evaluations=96, search_group_evaluations=204,
                 search_patience=96, bank_candidates=True,
                 pack_reload_candidates=True, pack_preload_candidates=True),
}


@dataclass(frozen=True)
class Candidate:
    lookahead: int = 0
    random_seed: int | None = None
    store_windows: tuple[int, ...] = ()
    wait_placement: str = "default"
    virtual_orders: tuple[tuple[int, ...], ...] = ()
    mreg_ranks: tuple[int, ...] = ()
    pack_reload_after_weight: bool = False
    pack_weight_before_pack: bool = False

    def __post_init__(self) -> None:
        if self.lookahead not in (0, 1, 2, 4, 8):
            raise ValueError("lookahead must be 0, 1, 2, 4, or 8")
        if self.random_seed is not None and (self.random_seed < 0 or self.lookahead):
            raise ValueError("random-seed must be nonnegative and excludes lookahead")
        if any(rank < 0 or rank > 0xFFFFFFFF for rank in self.store_windows):
            raise ValueError("store-window ranks must be unsigned 32-bit integers")
        if self.virtual_orders and (self.lookahead or self.random_seed is not None):
            raise ValueError("explicit virtual orders exclude lookahead and random-seed")
        if any(rank < 0 or rank > 0xFFFFFFFF for rank in self.mreg_ranks):
            raise ValueError("MREG ranks must be unsigned 32-bit integers")
        if self.wait_placement not in ("default", "source", "preloaded"):
            raise ValueError("wait-placement must be default, source or preloaded")
        if self.pack_reload_after_weight and self.pack_weight_before_pack:
            raise ValueError("PACK weight placement options are mutually exclusive")
        if self.wait_placement == "preloaded" and not self.pack_weight_before_pack:
            raise ValueError("preloaded wait placement requires PACK weight preloading")


def _option_path(path: Path) -> str:
    return '"' + str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def evidence_option(model_facts: Path | None = None) -> list[str]:
    if model_facts is None:
        return []  # A library caller may supply IR with selected evidence.
    def bundle_path(value: Path) -> Path:
        path = Path(value)
        if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
            raise ValueError("prohibited path")
        if ".." in path.parts:
            raise ValueError("parent traversal is unsupported")
        path = Path(os.path.abspath(path))
        if any(parent.is_symlink() for parent in reversed((path, *path.parents))):
            raise ValueError("model facts paths must not contain symlinks")
        return path
    directory = bundle_path(model_facts)
    if not directory.is_dir():
        raise ValueError("--model-facts must name a qualified bundle directory")
    for name in ("facts.lock.json", "model-timing.json", "model-overlap.json"):
        if not bundle_path(directory / name).is_file():
            raise ValueError(f"model facts bundle requires {name}")
    return ["--select-atlas-model-evidence=facts=" + _option_path(directory)]


def candidate_options(candidate: Candidate, provider: str, spill: bool, *, graph_output: bool = False) -> list[str]:
    options = [f"provider={provider}", f"spill={str(spill).lower()}", "allocation-report=allocation.json"]
    if candidate.virtual_orders:
        options.append("order-file=order.json")
    if graph_output:
        options.append("graph-output=graph.json")
    if candidate.lookahead:
        options.append(f"lookahead={candidate.lookahead}")
    if candidate.random_seed is not None:
        options.append(f"random-seed={candidate.random_seed}")
    if candidate.store_windows:
        options.append("store-windows='" + ",".join(map(str, candidate.store_windows)) + "'")
    if candidate.mreg_ranks:
        options.append("mreg-ranks='" + ",".join(map(str, candidate.mreg_ranks)) + "'")
    if candidate.pack_reload_after_weight:
        options.append("pack-reload-after-weight=true")
    if candidate.pack_weight_before_pack:
        options.append("pack-weight-before-pack=true")
    if candidate.wait_placement != "default":
        options.append("wait-placement=" + candidate.wait_placement)
    return ["--atlas-compile-candidate=" + " ".join(options)]


def order_neighbors(graphs: list[dict], orders: tuple[tuple[int, ...], ...]):
    """Enumerate single insertions within each node's dependency window."""
    moves = []
    for block, (graph, order) in enumerate(zip(graphs, orders)):
        positions = {node: index for index, node in enumerate(order)}
        successors = [[] for _ in order]
        for node, preds in enumerate(graph["preds"]):
            for pred in preds:
                successors[pred].append(node)
        for origin, node in enumerate(order):
            first = max((positions[pred] + 1 for pred in graph["preds"][node]), default=0)
            last = min((positions[succ] - 1 for succ in successors[node]), default=len(order) - 1)
            for target in range(first, last + 1):
                if origin == target:
                    continue
                moves.append((block, origin, target))
    random.Random(0).shuffle(moves)
    for block, origin, target in moves:
        moved = list(orders[block])
        moved.insert(target, moved.pop(origin))
        yield orders[:block] + (tuple(moved),) + orders[block + 1:]


def group_order_neighbors(graphs: list[dict], orders: tuple[tuple[int, ...], ...], max_nodes: int = 4):
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
                # Added adjacency edges identify the resulting permutation exactly,
                # without retaining a complete order for every legal group move.
                signature = (block, tuple(edge for edge in zip((-1,) + moved, moved + (-2,))
                                          if edge not in parent_edges[block]))
                if moved != order and signature not in seen:
                    seen.add(signature)
                    moves.append((block, tuple(group), insertion))
    random.Random(0).shuffle(moves)
    for block, group, insertion in moves:
        members = set(group)
        remaining = [node for node in orders[block] if node not in members]
        moved = tuple(remaining[:insertion]) + group + tuple(remaining[insertion:])
        yield orders[:block] + (moved,) + orders[block + 1:]


def proposal_estimate(graph: dict, order: tuple[int, ...]) -> tuple[int, int]:
    """Replay the native virtual Machine; this priority never establishes completion."""
    proposal = graph["proposal"]
    costs = proposal["costs"]
    free = {}
    finished = [0] * len(costs)
    transfers = []
    transfer_end = 0
    frontend = makespan = waiting = 0
    for node in order:
        cost = costs[node]
        unit = cost["unit"]
        cycle = frontend if cost["queued_dma"] else max(frontend, free.get(unit, 0))
        if cost["drains"]:
            cycle = max(cycle, transfer_end)
        for pred in graph["preds"][node]:
            cycle = max(cycle, finished[pred])
        if cost["queued_dma"]:
            # Frontend cycles never decrease. Retire completed transfers once;
            # the active completion heap stays bounded by DMA capacity.
            while transfers and transfers[0] <= cycle:
                heapq.heappop(transfers)
            capacity = proposal["dma_capacity"]
            while len(transfers) >= capacity:
                cycle = heapq.heappop(transfers)
        work = max(cycle + cost["frontend"] - 1, free.get(unit, 0)) if cost["queued_dma"] else cycle
        finished[node] = work + cost["latency"]
        free[unit] = work + cost["busy"]
        if cost["queued_dma"]:
            end = work + cost["busy"]
            heapq.heappush(transfers, end)
            transfer_end = max(transfer_end, end)
        waiting += cycle - frontend
        frontend = cycle + cost["frontend"]
        makespan = max(makespan, finished[node])
    return makespan, waiting


def validate_proposal_graphs(graphs: list[dict], program: dict) -> None:
    """Fail closed before ranking if metadata differs from the scored parent's facts."""
    def reject(why):
        raise ModelTimingEvidenceError("curve proposal metadata: " + why)

    evidence = program.get("model_evidence") if isinstance(program, dict) else None
    identity = evidence.get("op_timing_sha256") if isinstance(evidence, dict) else None
    if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        reject("parent program requires its selected op_timing_sha256")
    if not isinstance(graphs, list):
        reject("graphs must be an array")
    for graph in graphs:
        if not isinstance(graph, dict) or not isinstance(graph.get("preds"), list):
            reject("missing graph predecessors")
        count = len(graph["preds"])
        order = graph.get("order")
        if not isinstance(order, list) or any(type(node) is not int for node in order) or sorted(order) != list(range(count)):
            reject("parent order is not a complete permutation")
        positions = {node: index for index, node in enumerate(order)}
        for node, preds in enumerate(graph["preds"]):
            if not isinstance(preds, list) or any(type(pred) is not int or not 0 <= pred < count for pred in preds):
                reject("invalid graph predecessor")
            if len(set(preds)) != len(preds) or any(positions[pred] >= positions[node] for pred in preds):
                reject("parent order violates its graph")
        proposal = graph.get("proposal")
        if not isinstance(proposal, dict) or set(proposal) != {"schema", "facts_sha256", "units", "dma_capacity", "costs", "estimate"}:
            reject("missing or unsupported proposal schema")
        if proposal["schema"] != "atlas.virtual_proposal.v1" or proposal["facts_sha256"] != identity:
            reject("proposal identity differs from parent model evidence")
        for key in ("units", "dma_capacity"):
            if type(proposal[key]) is not int or not 0 < proposal[key] <= 2**31 - 1:
                reject("invalid " + key)
        costs = proposal["costs"]
        if not isinstance(costs, list) or len(costs) != count:
            reject("cost coverage differs from graph")
        cycle_bound = 0
        for cost in costs:
            if not isinstance(cost, dict) or set(cost) != {"unit", "busy", "latency", "frontend", "queued_dma", "drains"}:
                reject("missing or unsupported cost fields")
            if type(cost["unit"]) is not int or not 0 <= cost["unit"] < proposal["units"]:
                reject("invalid functional unit")
            for key in ("busy", "latency", "frontend"):
                if type(cost[key]) is not int or not 0 <= cost[key] <= 2**31 - 1:
                    reject("cost outside nonnegative native cycle domain")
                cycle_bound += cost[key]
            if type(cost["queued_dma"]) is not bool or type(cost["drains"]) is not bool:
                reject("invalid DMA flags")
            if cost["queued_dma"] and cost["frontend"] < 1:
                reject("queued DMA must issue before its transfer starts")
        if cycle_bound > 2**31 - 1:
            reject("block exceeds native cycle domain")
        estimate = proposal["estimate"]
        if not isinstance(estimate, dict) or set(estimate) != {"makespan", "exposed_waiting"} or any(
                type(value) is not int or value < 0 for value in estimate.values()):
            reject("invalid native replay estimate")
        if proposal_estimate(graph, tuple(order)) != (estimate["makespan"], estimate["exposed_waiting"]):
            reject("native replay parity failure")


def rank_order_neighbors(graphs: list[dict], neighbors, max_orders: int | None = None,
                         eligible: Callable | None = None):
    """Stream estimates and retain only the stable prefix the attempt budget can use.

    Every unseen legal order must still be estimated to preserve the exhaustive
    ranking's prefix. Storage for full orders and sorting scale with max_orders,
    not neighborhood size. Compact adjacency signatures deduplicate before replay.
    """
    if max_orders is not None and max_orders < 0:
        raise ValueError("ranked order limit must be nonnegative")
    if max_orders == 0:
        return iter(())
    baseline = [tuple(graph["order"]) for graph in graphs]
    parent_edges = [set(zip((-1,) + order, order + (-2,))) for order in baseline]
    estimates_at_parent = [proposal_estimate(graph, order) for graph, order in zip(graphs, baseline)]

    def unique_neighbors():
        seen = set()
        for orders in neighbors:
            if eligible is not None and not eligible(orders):
                continue
            signature = tuple((block, tuple(edge for edge in zip((-1,) + order, order + (-2,))
                                             if edge not in parent_edges[block]))
                              for block, order in enumerate(orders) if order != baseline[block])
            if signature not in seen:
                seen.add(signature)
                yield orders

    def key(orders):
        estimates = [estimates_at_parent[block] if order == baseline[block] else proposal_estimate(graph, order)
                     for block, (graph, order) in enumerate(zip(graphs, orders))]
        return tuple(sum(estimate[index] for estimate in estimates) for index in (0, 1))
    if max_orders is None:
        return iter(sorted(unique_neighbors(), key=key))
    return iter(heapq.nsmallest(max_orders, unique_neighbors(), key=key))


def order_candidates(base: Candidate, orders, curve_candidates: bool):
    """Each wait sibling is a separate ordinary attempt under existing search limits."""
    for order in orders:
        if curve_candidates:
            # Local search precedes PACK-placement trials. Reset preloaded state
            # defensively so inherited options cannot suppress either wait sibling.
            for placement in ("default", "source"):
                yield replace(base, virtual_orders=order, wait_placement=placement,
                              pack_weight_before_pack=False)
        else:
            yield replace(base, virtual_orders=order)


def select_program(source: Path, output_dir: Path, atlas_bin_dir: Path, *, provider: str = "model",
                   profile: str | None = None, model: Path | None = None, model_regions: tuple[Path, ...] = (),
                   spill: bool = True, spill_mxu: bool = True, boundary: str = "boundary",
                   guarded_pack: bool = False,
                   model_facts: Path | None = None,
                   max_candidates: int = 32, random_seeds: tuple[int, ...] = tuple(range(8)), workers: int = 12,
                   store_window_candidates: tuple[tuple[int, ...], ...] = (),
                   mreg_candidates: tuple[tuple[int, ...], ...] | None = (),
                   search_evaluations: int = 0, search_group_evaluations: int = 0,
                   search_patience: int = 0,
                   pack_reload_candidates: bool = False, pack_preload_candidates: bool = False,
                   curve_candidates: bool | None = None, scorer: Callable | None = None) -> dict:
    """Return a receipt; selected artifacts are also copied to output_dir.

    All candidates run allocation, timing and generated-schedule verification
    before scoring. Compiler refusals remain in the receipt and cannot win. A
    failed builtin default, a canonical scoring failure (including a mismatched
    model identity) or no passing initial candidate fails the compilation.
    max_candidates bounds initial attempts, including compiler rejections. Search
    budgets count every attempted candidate, including compiler refusals.
    """
    if profile is None:
        profile = "curve" if provider == "model" else "fifo43"
    if provider not in ("builtin", "model") or profile not in ("fifo43", "curve"):
        raise ValueError("unsupported provider or memory profile")
    if curve_candidates is None:
        curve_candidates = provider == "model" and profile == "curve"
    if type(curve_candidates) is not bool or (curve_candidates and (provider != "model" or profile != "curve")):
        raise ValueError("curve-candidates requires the model curve profile")
    if max_candidates < 1 or workers < 1 or boundary not in ("boundary", "none"):
        raise ValueError("candidate and worker counts must be positive and boundary must be boundary or none")
    if provider != "model" and mreg_candidates != ():
        raise ValueError("MREG candidates require the model provider")
    if search_evaluations < 0 or (search_evaluations and provider != "model"):
        raise ValueError("search-evaluations must be nonnegative and requires the model provider")
    if search_group_evaluations < 0 or (search_group_evaluations and provider != "model"):
        raise ValueError("search-group-evaluations must be nonnegative and requires the model provider")
    if search_patience < 0 or (search_patience and provider != "model"):
        raise ValueError("search-patience must be nonnegative and requires the model provider")
    if pack_reload_candidates and provider != "model":
        raise ValueError("pack-reload-candidates requires the model provider")
    if pack_preload_candidates and provider != "model":
        raise ValueError("pack-preload-candidates requires the model provider")
    source, output, binaries = Path(source).resolve(strict=True), Path(output_dir).resolve(), Path(atlas_bin_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be empty to prevent stale artifacts")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    commands, logs = [], []
    log_lock, score_lock = threading.Lock(), threading.Lock()
    receipt = {"schema": "atlas.model_selection.v1", "provider": provider, "profile": profile,
               "guarded_pack": guarded_pack, "curve_candidates": curve_candidates,
               "max_candidates": max_candidates, "workers": workers, "search_evaluations": search_evaluations,
               "search_group_evaluations": search_group_evaluations,
               "search_patience": search_patience,
               "pack_reload_candidates": pack_reload_candidates,
               "pack_preload_candidates": pack_preload_candidates,
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "candidates": []}

    def command(argv: list[str], directory: Path, stdout: Path | None = None) -> str:
        rendered = f"cd {shlex.quote(str(directory))} && " + shlex.join(argv)
        if stdout:
            rendered += f" > {shlex.quote(str(stdout))}"
        with log_lock:
            commands.append(rendered)
            (output / "commands.txt").write_text("\n".join(commands) + "\n")
        try:
            result = subprocess.run(argv, cwd=directory, capture_output=True, text=True,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        except OSError as error:
            result = subprocess.CompletedProcess(argv, 1, "", str(error))
        with log_lock:
            logs.append(f"$ {rendered}\n{result.stderr}exit {result.returncode}\n")
            (output / "log.txt").write_text("\n".join(logs))
        if result.returncode:
            raise ValueError(result.stderr.strip() or f"{argv[0]} exited {result.returncode}")
        if stdout:
            stdout.write_text(result.stdout)
        return result.stdout

    if provider == "model" and model_facts is None and model is not None:
        model_facts = output / "model-facts"
        command([sys.executable, "-B", str(Path(__file__).with_name("atlas_model_facts.py")),
                 "--npu-model", str(model), "--out", str(model_facts)], output)
    options = evidence_option(model_facts) if provider == "model" else []
    options.append("--atlas-compile-prepare=" +
                   f"spill-mxu-state={str(spill_mxu).lower()} boundary={boundary}")
    if guarded_pack:
        options[-1] += " guarded-pack=true"
    prepared = output / "1-virtual.mlir"
    command([str(binaries / "atlas-opt"), str(source), *options,
             "--mlir-print-op-on-diagnostic=false", "-o", str(prepared)], output)
    standard_scorer = scorer is None
    if standard_scorer and provider == "model":
        if __package__:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
        from atlas_model_scorer import ModelScorer
        if model is None:
            raise ValueError("model scoring requires --npu-model")
        scorer = ModelScorer(model, model_regions)
    seen: set[Candidate] = set()
    # The scorer/model/regions are fixed for this invocation; profiles must not share scores.
    score_cache: dict[tuple[str, tuple[int, ...]], int | Future] = {}
    score_executor: ProcessPoolExecutor | None = None
    best: dict | None = None

    def cached_score(program: dict, scoring_profile: str) -> tuple[int | Future, bool]:
        key = scoring_profile, tuple(instruction["word_u32"] for instruction in program["instructions"])
        with score_lock:
            hit = key in score_cache
            if not hit:
                score_cache[key] = (score_executor.submit(scorer, program, scoring_profile)
                                    if score_executor else scorer(program, profile=scoring_profile))
            return score_cache[key], hit

    def compile_candidate(candidate: Candidate, row: dict) -> None:
        directory = output / row["directory"]
        before = time.monotonic()
        stage = "compile"
        try:
            if candidate.virtual_orders:
                (directory / "order.json").write_text(json.dumps(candidate.virtual_orders) + "\n")
            command([str(binaries / "atlas-opt"), str(prepared),
                     *candidate_options(candidate, provider, spill,
                                        graph_output=bool(search_evaluations or search_group_evaluations)),
                     "--mlir-print-op-on-diagnostic=false", "-o", "final.mlir"], directory)
            stage = "emit"
            program = json.loads(command([str(binaries / "atlas-emit"), "final.mlir", "--program-json"],
                                         directory, directory / "program.json"))
            words = tuple(instruction["word_u32"] for instruction in program["instructions"])
            (directory / "program.hex").write_text("".join(f"{word:08x}\n" for word in words))
            if provider == "model":
                stage = "score"
                cached, row["score_cache_hit"] = cached_score(program, profile)
                row["cycles"] = cached.result() if isinstance(cached, Future) else cached
            row.update(status="PASS", words=len(words), program_sha256=hashlib.sha256(
                b"".join(word.to_bytes(4, "little") for word in words)).hexdigest())
        except (OSError, ValueError, RuntimeError) as error:
            row.update(status="FAIL", error=str(error), error_stage=stage)
        finally:
            row["elapsed_seconds"] = time.monotonic() - before

    def fail(message: str) -> None:
        (output / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
        raise ValueError(message)

    def evaluate_batch(specifications: list[tuple[Candidate, str]], limit: int = max_candidates) -> None:
        nonlocal best
        pending, before = [], len(receipt["candidates"])
        for candidate, category in specifications:
            if candidate in seen or len(receipt["candidates"]) >= (limit if provider == "model" else 1):
                continue
            index = len(receipt["candidates"])
            directory = output / "candidates" / f"{index:03d}-{category}"
            directory.mkdir(parents=True)
            row = {"index": index, "class": category, "options": asdict(candidate),
                   "directory": str(directory.relative_to(output))}
            pending.append((candidate, row))
            seen.add(candidate)
            receipt["candidates"].append(row)
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as executor:
            futures = [executor.submit(compile_candidate, candidate, row) for candidate, row in pending]
            for future in futures:
                future.result()
        for row in receipt["candidates"][before:]:
            if row["status"] == "FAIL" and provider != "model":
                fail("default candidate failed: " + row["error"])
            # A verified program the model cannot score is an identity or soundness alarm.
            if row["status"] == "FAIL" and row["error_stage"] == "score":
                fail(f"candidate {row['index']} failed model scoring: {row['error']}")
        passing = [row for row in receipt["candidates"] if row["status"] == "PASS"]
        best = min(passing, key=lambda row: (row.get("cycles", 0), row["index"]), default=None)
        if provider == "model" and profile == "curve" and best is not None:
            tied = [row for row in passing if row["cycles"] == best["cycles"]]
            if len(tied) > 1:
                pending_scores = []
                for row in tied:
                    if "fifo43_cycles" not in row:
                        program = json.loads((output / row["directory"] / "program.json").read_text())
                        try:
                            cached, row["fifo43_score_cache_hit"] = cached_score(program, "fifo43")
                            pending_scores.append((row, cached))
                        except (OSError, ValueError, RuntimeError) as error:
                            row.update(status="FAIL", error=str(error), error_stage="score")
                            fail(f"candidate {row['index']} failed FIFO43 tie scoring: {error}")
                for row, cached in pending_scores:
                    try:
                        row["fifo43_cycles"] = cached.result() if isinstance(cached, Future) else cached
                    except (OSError, ValueError, RuntimeError) as error:
                        row.update(status="FAIL", error=str(error), error_stage="score")
                        fail(f"candidate {row['index']} failed FIFO43 tie scoring: {error}")
                best = min(tied, key=lambda row: (row["fifo43_cycles"], row["index"]))

    # Model imports, configuration and execution are isolated from compiler threads.
    with ExitStack() as stack:
        if standard_scorer and provider == "model":
            score_executor = stack.enter_context(ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")))
        evaluate_batch([(Candidate(), "default")])
        if provider == "model":
            evaluate_batch([(Candidate(lookahead=lookahead), "lookahead") for lookahead in (1, 2, 4)] +
                           [(Candidate(random_seed=seed), "random-seed") for seed in random_seeds], limit=min(24, max_candidates))
            if best is None:
                fail("no legal candidate: every initial candidate was refused")
            def winner() -> Candidate:
                options = dict(best["options"])
                options["store_windows"] = tuple(options["store_windows"])
                options["virtual_orders"] = tuple(tuple(order) for order in options["virtual_orders"])
                options["mreg_ranks"] = tuple(options["mreg_ranks"])
                return Candidate(**options)

            program = json.loads((output / best["directory"] / "program.json").read_text())
            store_count = sum(instruction.get("operation") == "atlas.dma" and
                              instruction.get("fields", {}).get("direction") == "store"
                              for instruction in program["instructions"])
            # All legal per-store free ranks are expressible; allocation maps ranks
            # modulo that store's actual free set and never claims a held window.
            window_base = winner()
            placements = [(replace(window_base, wait_placement="source"), "wait-placement")]
            placements += [(replace(window_base, store_windows=tuple(ranks)), "explicit-store-windows")
                           for ranks in store_window_candidates]
            placements += [(replace(window_base, store_windows=(rank,) * store_count), "uniform-store-windows")
                           for rank in range(1, 8) if store_count]
            evaluate_batch(placements, limit=min(24, max_candidates))
            register_base = winner()
            if mreg_candidates is None:
                mreg_candidates = tuple(tuple(random.Random(seed).choices(range(32), k=64)) for seed in range(8))
            receipt["mreg_pool_size"] = len(mreg_candidates)
            evaluate_batch([(replace(register_base, mreg_ranks=tuple(ranks)), "mreg-ranks") for ranks in mreg_candidates])
            for field, budget, neighbors, category in (
                    ("search", search_evaluations, order_neighbors, "local-order"),
                    ("group_search", search_group_evaluations, group_order_neighbors, "group-order")):
                if not budget:
                    continue
                search_started = time.monotonic()
                initial_count = len(receipt["candidates"])
                limit = initial_count + budget
                receipt[field] = {"start_index": best["index"], "accepted_indices": [], "stopped_for_patience": False}
                while len(receipt["candidates"]) < limit:
                    parent = best
                    nonimproving = 0
                    graphs = json.loads((output / parent["directory"] / "graph.json").read_text())
                    orders = tuple(tuple(graph["order"]) for graph in graphs)
                    base = replace(winner(), lookahead=0, random_seed=None, virtual_orders=orders)
                    seen.add(base)
                    proposed = neighbors(graphs, orders)
                    if curve_candidates:
                        parent_program = json.loads((output / parent["directory"] / "program.json").read_text())
                        try:
                            validate_proposal_graphs(graphs, parent_program)
                        except ValueError as error:
                            fail(str(error))
                        # Bank choices or a new order can change the best wait placement.
                        # Include the incumbent order; the seen set drops its unchanged sibling.
                        # Each retained order has at least one unseen wait sibling.
                        # Thus remaining attempts also bound the orders needed for
                        # every canonical batch within the attempt budget.
                        remaining = limit - len(receipt["candidates"])
                        proposed = rank_order_neighbors(
                            graphs, chain((orders,), proposed), remaining,
                            eligible=lambda order: any(candidate not in seen for candidate in
                                                       order_candidates(base, (order,), True)))
                        receipt[field]["proposal_policy"] = "native-cost-rank-with-wait-siblings-v1"
                        receipt[field]["proposal_facts_sha256"] = parent_program["model_evidence"]["op_timing_sha256"]
                    pending = order_candidates(base, proposed, curve_candidates)
                    while len(receipt["candidates"]) < limit:
                        batch = []
                        width = min(12, limit - len(receipt["candidates"]))
                        while len(batch) < width:
                            candidate = next(pending, None)
                            if candidate is None:
                                break
                            if candidate not in seen and candidate not in batch:
                                batch.append(candidate)
                        if not batch:
                            break
                        before = len(receipt["candidates"])
                        evaluate_batch([(candidate, category) for candidate in batch], limit)
                        for row in receipt["candidates"][before:]:
                            row["parent_index"] = parent["index"]
                        if best["index"] != parent["index"]:
                            receipt[field]["accepted_indices"].append(best["index"])
                            break
                        nonimproving += len(receipt["candidates"]) - before
                        if search_patience and nonimproving >= search_patience:
                            receipt[field]["stopped_for_patience"] = True
                            break
                    if best["index"] == parent["index"]:
                        break
                receipt[field].update(evaluations=len(receipt["candidates"]) - initial_count,
                                      elapsed_seconds=time.monotonic() - search_started)
            if pack_reload_candidates:
                evaluate_batch([(replace(winner(), pack_reload_after_weight=True), "pack-reload-order")],
                               len(receipt["candidates"]) + 1)
            if pack_preload_candidates:
                preloaded = replace(winner(), pack_reload_after_weight=False, pack_weight_before_pack=True)
                evaluate_batch([(replace(preloaded, wait_placement="preloaded"), "pack-preload-waits")],
                               len(receipt["candidates"]) + 1)
    selected = output / best["directory"]
    for name in ("final.mlir", "allocation.json", "program.json", "program.hex"):
        shutil.copyfile(selected / name, output / name)
    receipt.update(selected_index=best["index"], selected_class=best["class"],
                   default_cycles=receipt["candidates"][0].get("cycles"),
                   selected_cycles=best.get("cycles"),
                   score_cache_entries=sum(key[0] == profile for key in score_cache),
                   fifo43_tiebreak_cache_entries=sum(key[0] == "fifo43" for key in score_cache) if profile == "curve" else 0,
                   score_cache_hits=sum(row.get("score_cache_hit", False) for row in receipt["candidates"]),
                   elapsed_seconds=time.monotonic() - started)
    (output / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("builtin", "model"), default="builtin")
    parser.add_argument("--profile", choices=("fifo43", "curve"),
                        help="memory scoring profile (model default: curve)")
    parser.add_argument("--npu-model", type=Path, help="selected npu_model source tree; must match provider evidence")
    parser.add_argument("--model-region", type=Path, action="append", default=[],
                        help="DRAM region JSON with address, initial_hex and optional expected_hex")
    parser.add_argument("--model-facts", type=Path,
                        help="qualified model facts bundle (default: regenerate from --npu-model)")
    parser.add_argument("--search-budget", choices=tuple(SEARCH_BUDGETS), default="full",
                        help="selection knobs only or bounded full search (default)")
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--bank-candidates", action=argparse.BooleanOptionalAction, default=None,
                        help="score eight legal MREG rank vectors (model only)")
    parser.add_argument("--search-evaluations", type=int,
                        help="additional verified local order attempts after knob selection (model only)")
    parser.add_argument("--search-group-evaluations", type=int,
                        help="additional verified dependency-group attempts after local search (model only)")
    parser.add_argument("--search-patience", type=int,
                        help="stop a phase after this many nonimproving attempts, checked after each batch; zero disables")
    parser.add_argument("--curve-candidates", action=argparse.BooleanOptionalAction, default=None,
                        help="rank legal order proposals by derived costs and score default/source wait siblings (curve default)")
    parser.add_argument("--pack-reload-candidates", action=argparse.BooleanOptionalAction, default=None,
                        help="score loading a pending weight before an adjacent PACK reload (model only)")
    parser.add_argument("--pack-preload-candidates", action=argparse.BooleanOptionalAction, default=None,
                        help="score preloading disjoint weights before PACK with selective source waits (model only)")
    parser.add_argument("--store-windows", action="append", default=[], metavar="RANKS",
                        help="additional candidate's comma-separated free ranks, one per DMA store")


def model_arguments(args: argparse.Namespace) -> dict:
    preset = SEARCH_BUDGETS[args.search_budget if args.provider == "model" else "selection"]
    budget = {name: getattr(args, name) if getattr(args, name) is not None else value for name, value in preset.items()}
    banks = budget.pop("bank_candidates")
    profile = args.profile or ("curve" if args.provider == "model" else "fifo43")
    return {"provider": args.provider, "profile": profile, "model": args.npu_model,
            "model_regions": tuple(args.model_region), "model_facts": args.model_facts,
            "curve_candidates": args.curve_candidates,
            "workers": args.workers,
            "mreg_candidates": None if banks else (), **budget,
            "store_window_candidates": tuple(tuple(map(int, value.split(','))) for value in args.store_windows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atlas-bin-dir", type=Path,
                        default=Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin")))
    parser.add_argument("--boundary", choices=("boundary", "none"), default="boundary")
    parser.add_argument("--no-spill", action="store_true")
    parser.add_argument("--guarded-pack", action="store_true")
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
