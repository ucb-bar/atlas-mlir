#!/usr/bin/env python3
"""Choose verified Atlas programs by native model cycles, breaking ties by candidate order."""

from __future__ import annotations

if __package__:
    from .atlas_model_diagnostics import ModelTimingEvidenceError
    from .atlas_search_policy import POLICY, COMPILER_POLICY_SHA256, policy_receipt
else:
    from atlas_model_diagnostics import ModelTimingEvidenceError
    from atlas_search_policy import POLICY, COMPILER_POLICY_SHA256, policy_receipt

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


def capture_selection_inputs(source, output, binaries, model, regions, search_policy):
    """Capture production inputs without importing the model into compiler threads."""
    from atlas_run_provenance import capture_run_provenance, safe_path

    model = safe_path(model)
    revisions = {}
    for role, root in (("compiler", ROOT), ("model", model)):
        revisions[role] = None
        # A source-only model copy must not inherit an enclosing repository's HEAD.
        if safe_path(root / ".git").exists():
            result = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
                                    capture_output=True, text=True)
            revisions[role] = result.stdout.strip() if result.returncode == 0 else None
    return capture_run_provenance(
        source, regions=regions, archive_dir=output / "provenance", revisions=revisions,
        search_policy=search_policy,
        files={"compiler_build_definition": ROOT / "CMakeLists.txt",
               "instruction_inventory": ROOT / "docs/selected-variant-inventory.json",
               **{name: binaries / name for name in ("atlas-opt", "atlas-emit")}},
        trees={"model_source": (model / "npu_model", (".py", ".json", ".yaml")),
               "numerical_data": model / "npu_model/configs/data",
               "compiler_lib": ROOT / "lib", "compiler_include": ROOT / "include",
               "tools": (ROOT / "tools", (".py", ".json", ".yaml"))})


def bind_selection_facts(capture, model_facts):
    from atlas_run_provenance import safe_path, selection_facts_identity

    directory = safe_path(model_facts)
    capture.bind(evidence_files={"facts/" + name: directory / name
                                 for name in ("facts.lock.json", "model-timing.json", "model-overlap.json")})
    facts = json.loads((directory / "model-timing.json").read_text())
    semantic_digest, provider_digest = selection_facts_identity(facts)
    capture.bind(semantic={"derived_facts_sha256": semantic_digest})
    return provider_digest


SEARCH_BUDGETS = POLICY["presets"]
POOLS = POLICY["pools"]
CANDIDATES = POLICY["candidates"]
DEFAULTS = POLICY["defaults"]


def objective_receipt(provider: str, profile: str, standard_scorer: bool,
                      configuration_sha256: str | None = None) -> dict:
    """Identify selection policy separately from model-derived timing facts."""
    model = provider == "model"
    payload = {
        "schema": "atlas.model_selection_objective.v1",
        "policy": POLICY["algorithms"]["selection_objective"] if model else "builtin-default-v1",
        "scoring": ("native-model" if standard_scorer else "injected-scorer") if model else "none",
        "profile": profile if model else None,
        "minimize": list(POLICY["objectives"][profile]) if model else ["index"],
        "tie_rule": POLICY["algorithms"]["selection_tie"],
        "scoring_configuration_sha256": configuration_sha256,
    }
    return dict(payload, sha256=hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest())


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
        if self.lookahead not in POOLS["supported_lookaheads"]:
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
    for enabled, option in (
        (candidate.virtual_orders, "order-file=order.json"),
        (graph_output, "graph-output=graph.json"),
        (candidate.lookahead, f"lookahead={candidate.lookahead}"),
        (candidate.random_seed is not None, f"random-seed={candidate.random_seed}"),
        (candidate.store_windows, "store-windows='" + ",".join(map(str, candidate.store_windows)) + "'"),
        (candidate.mreg_ranks, "mreg-ranks='" + ",".join(map(str, candidate.mreg_ranks)) + "'"),
        (candidate.pack_reload_after_weight, "pack-reload-after-weight=true"),
        (candidate.pack_weight_before_pack, "pack-weight-before-pack=true"),
        (candidate.wait_placement != "default", "wait-placement=" + candidate.wait_placement),
    ):
        if enabled:
            options.append(option)
    return ["--atlas-compile-candidate=" + " ".join(options)]


def group_order_neighbors(graphs: list[dict], orders: tuple[tuple[int, ...], ...], max_nodes: int = POOLS["group_max_nodes"]):
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
    random.Random(POOLS["neighbor_shuffle_seed"]).shuffle(moves)
    for block, group, insertion in moves:
        members = set(group)
        remaining = [node for node in orders[block] if node not in members]
        moved = tuple(remaining[:insertion]) + group + tuple(remaining[insertion:])
        yield orders[:block] + (moved,) + orders[block + 1:]


def proposal_estimate(graph: dict, order: tuple[int, ...]) -> tuple[int, int]:
    """Interpret the selected common projected-Core program; proposal policy only."""
    from atlas_model_replay import replay
    return replay(graph, order)


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
        if not isinstance(proposal, dict) or set(proposal) != {"schema", "facts_sha256", "units", "dma_capacity", "costs", "estimate", "replay"}:
            reject("missing or unsupported proposal schema")
        if proposal["schema"] != "atlas.virtual_proposal.v2" or proposal["facts_sha256"] != identity:
            reject("proposal identity differs from parent model evidence")
        from atlas_model_replay import validate_program, semantic_digest
        try:
            validate_program(proposal["replay"])
            if semantic_digest(proposal["replay"]) != evidence.get("replay_timing_sha256"):
                reject("replay program identity differs from parent model evidence")
        except (ValueError, TypeError) as error:
            reject(str(error))
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
    """Rank every unseen order, retaining only the stable prefix within the attempt budget."""
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
        return tuple(sum(estimate[index] for estimate in estimates) for index in [
            ("makespan", "exposed_waiting").index(field) for field in CANDIDATES["curve_ranking_fields"]])
    if max_orders is None:
        return iter(sorted(unique_neighbors(), key=key))
    return iter(heapq.nsmallest(max_orders, unique_neighbors(), key=key))


def order_candidates(base: Candidate, orders, curve_candidates: bool):
    """Each wait sibling is a separate ordinary attempt under existing search limits."""
    for order in orders:
        if curve_candidates:
            # Clear preloading so neither wait sibling is suppressed by inherited options.
            for placement in CANDIDATES["curve_wait_placements"]:
                yield replace(base, virtual_orders=order, wait_placement=placement,
                              pack_weight_before_pack=False)
        else:
            yield replace(base, virtual_orders=order)


def select_program(source: Path, output_dir: Path, atlas_bin_dir: Path, *, provider: str = DEFAULTS["selection_provider"],
                   profile: str | None = None, model: Path | None = None, model_regions: tuple[Path, ...] = (),
                   spill: bool = DEFAULTS["spill"], spill_mxu: bool = DEFAULTS["spill_mxu"],
                   boundary: str = DEFAULTS["boundary"],
                   guarded_pack: bool = DEFAULTS["guarded_pack"],
                   model_facts: Path | None = None,
                   max_candidates: int = DEFAULTS["api_max_candidates"],
                   random_seeds: tuple[int, ...] = tuple(POOLS["random_seeds"]), workers: int = DEFAULTS["workers"],
                   store_window_candidates: tuple[tuple[int, ...], ...] = (),
                   mreg_candidates: tuple[tuple[int, ...], ...] | None = (),
                   search_group_evaluations: int = DEFAULTS["api_search_group_evaluations"],
                   search_patience: int = DEFAULTS["api_search_patience"],
                   pack_reload_candidates: bool = DEFAULTS["api_pack_reload_candidates"],
                   pack_preload_candidates: bool = DEFAULTS["api_pack_preload_candidates"],
                   curve_candidates: bool | None = None, scorer: Callable | None = None,
                   search_policy=None) -> dict:
    """Verify and score candidates; count refusals toward budgets and publish the winner with its receipt."""
    if profile is None:
        profile = DEFAULTS["model_profile"] if provider == "model" else DEFAULTS["builtin_profile"]
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
    for name, value in (("search-group-evaluations", search_group_evaluations), ("search-patience", search_patience)):
        if value < 0 or (value and provider != "model"):
            raise ValueError(name + " must be nonnegative and requires the model provider")
    for name, value in (("pack-reload-candidates", pack_reload_candidates), ("pack-preload-candidates", pack_preload_candidates)):
        if value and provider != "model":
            raise ValueError(name + " requires the model provider")
    if mreg_candidates is None:
        mreg_candidates = tuple(tuple(random.Random(seed).choices(POOLS["mreg_rank_choices"],
                                                                k=POOLS["mreg_vector_length"]))
                                for seed in POOLS["mreg_seeds"])
    effective_policy = policy_receipt(dict(
        provider=provider, profile=profile, spill=spill, spill_mxu=spill_mxu,
        boundary=boundary, guarded_pack=guarded_pack, max_candidates=max_candidates,
        random_seeds=list(random_seeds), workers=workers,
        store_window_candidates=[list(ranks) for ranks in store_window_candidates],
        mreg_candidates=[list(ranks) for ranks in mreg_candidates],
        search_group_evaluations=search_group_evaluations,
        search_patience=search_patience, pack_reload_candidates=pack_reload_candidates,
        pack_preload_candidates=pack_preload_candidates, curve_candidates=curve_candidates,
        objective=POLICY["objectives"][profile],
        scoring_implementation=("none" if provider == "builtin" else
                                "native_model" if scorer is None else "caller_callback")))
    if search_policy is not None and search_policy != effective_policy:
        raise ValueError("search policy override does not match the versioned effective policy")
    if provider == "model":
        if __package__:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
        from atlas_run_provenance import safe_path
        source, output, binaries = map(safe_path, (source, output_dir, atlas_bin_dir))
    else:
        source, output, binaries = Path(source).resolve(strict=True), Path(output_dir).resolve(), Path(atlas_bin_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be empty to prevent stale artifacts")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    commands, logs = [], []
    log_lock, score_lock = threading.Lock(), threading.Lock()
    receipt = {"schema": "atlas.model_selection.v1", "provider": provider, "profile": profile,
               "objective": objective_receipt(provider, profile, scorer is None),
               "guarded_pack": guarded_pack, "curve_candidates": curve_candidates,
               "max_candidates": max_candidates, "workers": workers,
               "search_group_evaluations": search_group_evaluations,
               "search_patience": search_patience,
               "pack_reload_candidates": pack_reload_candidates,
               "pack_preload_candidates": pack_preload_candidates,
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "search_policy": effective_policy,
               "candidates": []}
    standard_scorer = scorer is None
    capture = None
    if standard_scorer and provider == "model":
        if model is None:
            raise ValueError("model scoring requires --npu-model")
        capture = capture_selection_inputs(source, output, binaries, model, model_regions, effective_policy)
        capture.bind(semantic={"invocation": {
            key: value for key, value in receipt.items() if key not in ("candidates", "source_sha256", "workers", "search_policy")
        } | dict(boundary=boundary, spill=spill, spill_mxu=spill_mxu,
                 random_seeds=random_seeds, store_window_candidates=store_window_candidates,
                 mreg_candidates=mreg_candidates)})
        from atlas_model_scorer import ModelScorer
        scorer = ModelScorer(model, model_regions)
        # Capture configuration before preparation, keeping model imports out of compiler threads.
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
            runtime = executor.submit(scorer.runtime_receipt).result()
        capture.bind(files={"selected_curve": runtime.pop("curve_path")},
                     semantic={"base_configuration": runtime.pop("configuration")})
        receipt["runtime_versions"] = runtime

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
    if capture is not None:
        receipt["provider_facts_sha256"] = bind_selection_facts(capture, model_facts)
    options = evidence_option(model_facts) if provider == "model" else []
    options.append("--atlas-compile-prepare=" +
                   f"spill-mxu-state={str(spill_mxu).lower()} boundary={boundary}")
    if guarded_pack:
        options[-1] += " guarded-pack=true"
    prepared = output / "1-virtual.mlir"
    command([str(binaries / "atlas-opt"), str(source), *options,
             "--mlir-print-op-on-diagnostic=false", "-o", str(prepared)], output)
    seen: set[Candidate] = set()
    # The scorer/model/regions are fixed for this invocation; profiles must not share scores.
    score_cache: dict[tuple[str, tuple[int, ...]], int | Future] = {}
    score_executor: ProcessPoolExecutor | None = None
    best: dict | None = None

    def cached_score(program: dict, scoring_profile: str) -> tuple[int | Future, bool]:
        key = scoring_profile, tuple(instruction["word_u32"] for instruction in program["instructions"])
        with score_lock:
            if standard_scorer:
                profiles = receipt.setdefault("scoring_configurations", {})
                if scoring_profile not in profiles:
                    profiles[scoring_profile] = score_executor.submit(
                        scorer.profile_receipt, program, scoring_profile).result()
                    receipt["objective"] = objective_receipt(
                        provider, profile, standard_scorer, profiles[scoring_profile]["sha256"])
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
                                        graph_output=bool(search_group_evaluations)),
                     "--mlir-print-op-on-diagnostic=false", "-o", "final.mlir"], directory)
            stage = "emit"
            program = json.loads(command([str(binaries / "atlas-emit"), "final.mlir", "--program-json"],
                                         directory, directory / "program.json"))
            if provider == "model" and (standard_scorer or "compiler_search_policy_sha256" in program):
                stage = "policy"
                if program.get("compiler_search_policy_sha256") != COMPILER_POLICY_SHA256:
                    raise ValueError("compiler search policy differs from driver; rebuild the selected compiler")
            words = tuple(instruction["word_u32"] for instruction in program["instructions"])
            (directory / "program.hex").write_text("".join(f"{word:08x}\n" for word in words))
            if provider == "model":
                stage = "score"
                cached, row["score_cache_hit"] = cached_score(program, profile)
                row["cycles"] = cached.result() if isinstance(cached, Future) else cached
                if (directory / "graph.json").is_file():
                    from atlas_model_replay import prediction_error
                    graphs = json.loads((directory / "graph.json").read_text())
                    row["proposal_prediction_error"] = prediction_error(graphs, row["cycles"], profile)
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
            if row["status"] == "FAIL" and row["error_stage"] in ("score", "policy"):
                fail(f"candidate {row['index']} failed model scoring: {row['error']}")
        passing = [row for row in receipt["candidates"] if row["status"] == "PASS"]
        # Assign canonical indices before parallel work to preserve deterministic ties.
        best = min(passing, key=lambda row: (row.get("cycles", 0), row["index"]), default=None)

    # Model imports, configuration and execution are isolated from compiler threads.
    with ExitStack() as stack:
        if standard_scorer and provider == "model":
            score_executor = stack.enter_context(ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")))
        evaluate_batch([(Candidate(), "default")])
        if provider == "model":
            evaluate_batch([(Candidate(lookahead=lookahead), "lookahead") for lookahead in POOLS["initial_lookaheads"]] +
                           [(Candidate(random_seed=seed), "random-seed") for seed in random_seeds], limit=min(CANDIDATES["initial_knob_cap"], max_candidates))
            if best is None:
                fail("no legal candidate: every initial candidate was refused")
            def winner() -> Candidate:
                return Candidate(**best["options"])  # asdict preserves tuple fields.

            program = json.loads((output / best["directory"] / "program.json").read_text())
            store_count = sum(instruction.get("operation") == "atlas.dma" and
                              instruction.get("fields", {}).get("direction") == "store"
                              for instruction in program["instructions"])
            # Allocation maps each free rank modulo the store's actual free set.
            window_base = winner()
            placements = [(replace(window_base, wait_placement=CANDIDATES["initial_wait_placement"]), "wait-placement")]
            placements += [(replace(window_base, store_windows=tuple(ranks)), "explicit-store-windows")
                           for ranks in store_window_candidates]
            placements += [(replace(window_base, store_windows=(rank,) * store_count), "uniform-store-windows")
                           for rank in POOLS["uniform_store_ranks"] if store_count]
            evaluate_batch(placements, limit=min(CANDIDATES["initial_knob_cap"], max_candidates))
            register_base = winner()
            receipt["mreg_pool_size"] = len(mreg_candidates)
            evaluate_batch([(replace(register_base, mreg_ranks=tuple(ranks)), "mreg-ranks") for ranks in mreg_candidates])
            if search_group_evaluations:
                search_started = time.monotonic()
                initial_count = len(receipt["candidates"])
                limit = initial_count + search_group_evaluations
                phase = receipt["group_search"] = {
                    "start_index": best["index"], "accepted_indices": [], "stopped_for_patience": False}
                while len(receipt["candidates"]) < limit:
                    parent = best
                    nonimproving = 0
                    graphs = json.loads((output / parent["directory"] / "graph.json").read_text())
                    orders = tuple(tuple(graph["order"]) for graph in graphs)
                    base = replace(winner(), lookahead=0, random_seed=None, virtual_orders=orders)
                    seen.add(base)
                    proposed = group_order_neighbors(graphs, orders)
                    if curve_candidates:
                        parent_program = json.loads((output / parent["directory"] / "program.json").read_text())
                        try:
                            validate_proposal_graphs(graphs, parent_program)
                        except ValueError as error:
                            fail(str(error))
                        # Revisit incumbent wait placement after bank/order changes. Each retained
                        # order has an unseen sibling, so remaining attempts bound the required prefix.
                        remaining = limit - len(receipt["candidates"])
                        proposed = rank_order_neighbors(
                            graphs, chain((orders,), proposed), remaining,
                            eligible=lambda order: any(candidate not in seen for candidate in
                                                       order_candidates(base, (order,), True)))
                        phase["proposal_policy"] = "native-cost-rank-with-wait-siblings-v1"
                        phase["proposal_facts_sha256"] = parent_program["model_evidence"]["op_timing_sha256"]
                    pending = order_candidates(base, proposed, curve_candidates)
                    while len(receipt["candidates"]) < limit:
                        batch = []
                        width = min(CANDIDATES["batch_width"], limit - len(receipt["candidates"]))
                        while len(batch) < width:
                            candidate = next(pending, None)
                            if candidate is None:
                                break
                            if candidate not in seen and candidate not in batch:
                                batch.append(candidate)
                        if not batch:
                            break
                        before = len(receipt["candidates"])
                        evaluate_batch([(candidate, "group-order") for candidate in batch], limit)
                        for row in receipt["candidates"][before:]:
                            row["parent_index"] = parent["index"]
                        if best["index"] != parent["index"]:
                            phase["accepted_indices"].append(best["index"])
                            break
                        nonimproving += len(receipt["candidates"]) - before
                        if search_patience and nonimproving >= search_patience:
                            phase["stopped_for_patience"] = True
                            break
                    if best["index"] == parent["index"]:
                        break
                phase.update(evaluations=len(receipt["candidates"]) - initial_count,
                             elapsed_seconds=time.monotonic() - search_started)
            if pack_reload_candidates:
                evaluate_batch([(replace(winner(), pack_reload_after_weight=True), "pack-reload-order")],
                               len(receipt["candidates"]) + 1)
            if pack_preload_candidates:
                preloaded = replace(winner(), pack_reload_after_weight=False, pack_weight_before_pack=True)
                evaluate_batch([(replace(preloaded, wait_placement=CANDIDATES["pack_preload_wait_placement"]), "pack-preload-waits")],
                               len(receipt["candidates"]) + 1)
    if capture is not None:
        try:
            receipt["provenance"] = capture.complete(
                scoring_configurations=receipt.get("scoring_configurations", {}),
                runtime_versions=receipt["runtime_versions"])
        except (OSError, ValueError) as error:
            receipt["publication_status"] = "REFUSED"
            fail(str(error))
    selected = output / best["directory"]
    for name in ("final.mlir", "allocation.json", "program.json", "program.hex"):
        shutil.copyfile(selected / name, output / name)
    receipt.update(selected_index=best["index"], selected_class=best["class"],
                   default_cycles=receipt["candidates"][0].get("cycles"),
                   selected_cycles=best.get("cycles"),
                   score_cache_entries=sum(key[0] == profile for key in score_cache),
                   score_cache_hits=sum(row.get("score_cache_hit", False) for row in receipt["candidates"]),
                   elapsed_seconds=time.monotonic() - started)
    (output / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("builtin", "model"), default=DEFAULTS["lower_provider"])
    parser.add_argument("--profile", choices=("fifo43", "curve"),
                        help="memory scoring profile (model default: curve)")
    parser.add_argument("--npu-model", type=Path, help="selected npu_model source tree; must match provider evidence")
    parser.add_argument("--model-region", type=Path, action="append", default=[],
                        help="DRAM region JSON with address, initial_hex and optional expected_hex")
    parser.add_argument("--model-facts", type=Path,
                        help="qualified model facts bundle (default: regenerate from --npu-model)")
    parser.add_argument("--search-budget", choices=tuple(SEARCH_BUDGETS), default=DEFAULTS["search_budget"],
                        help="selection knobs only or bounded full search (default)")
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--workers", type=int, default=min(DEFAULTS["workers"], os.cpu_count() or 1))
    parser.add_argument("--bank-candidates", action=argparse.BooleanOptionalAction, default=None,
                        help="score eight legal MREG rank vectors (model only)")
    parser.add_argument("--search-group-evaluations", type=int,
                        help="additional verified dependency-group attempts after knob selection (model only)")
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
    profile = args.profile or (DEFAULTS["model_profile"] if args.provider == "model" else DEFAULTS["builtin_profile"])
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
    parser.add_argument("--boundary", choices=("boundary", "none"), default=DEFAULTS["boundary"])
    parser.add_argument("--no-spill", action="store_true", default=not DEFAULTS["spill"])
    parser.add_argument("--guarded-pack", action="store_true", default=DEFAULTS["guarded_pack"])
    add_model_arguments(parser)
    parser.set_defaults(provider=DEFAULTS["selection_provider"])
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
