"""Source qualification for asynchronous DMA issue; never a completion estimate.

This deliberately preserves the compiler's matching-wait and alias policies.
Full reviewed ASTs establish the ordering/grant predicates; numeric pipeline
and configuration values are extracted without executing the model.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

from atlas_derive_curve import _engine_values
from atlas_derive_overlap import safe_path
from atlas_model_diagnostics import UnsupportedModelTiming
from atlas_model_schema import SOURCE_SHAPES, ast_digest, sha256


CONTRACT = "source-asynchronous-dma-admission-v1"
CORE_SHAPE = "faaabbe3276439f901d643b8c32e1746d6c2642fd18630db1283095fce1201d5"
SOURCE_NAMES = ("core.py", "dma.py", "lsu.py", "bank_conflict.py", "memory_backend.py")
SEMANTICS = dict(
    operands="captured_at_issue",
    launch="enqueue_after_registered_progress",
    requests="oldest_undispatched_slot",
    vmem_grants="lsu_priority_then_dma_load_write_then_dma_store_read",
    completion="all_beats_dispatched_and_acknowledged",
    wait="busy_flag_checked_before_engine_progress",
    channel_reuse="matching_wait_required",
    aliases="existing_completion_conflict_policy",
    curve_history="persists_across_wait_and_engine_drain",
    fixed_dma_windows="none",
    command_ring="separate_unchanged_rtl_contract")


class UnsupportedDMAAdmission(UnsupportedModelTiming):
    """DMA control flow/configuration has no reviewed admission interpretation."""


def review_sources(sources):
    """Pure AST contract, including a mutation-test interface."""
    if set(sources) != set(SOURCE_NAMES):
        raise UnsupportedDMAAdmission("missing DMA admission source")
    trees, shapes = {}, {}
    for name in SOURCE_NAMES:
        tree = ast.parse(sources[name])
        digest = ast_digest(tree)
        expected = CORE_SHAPE if name == "core.py" else SOURCE_SHAPES[name]
        if digest != expected:
            raise UnsupportedDMAAdmission("unsupported DMA admission control-flow semantics: " + name)
        trees[name], shapes[name] = tree, digest
    startup, interval = _engine_values(trees["dma.py"])
    # The reviewed Core has a single S1 launch per clock edge. Extract its
    # increment instead of inferring issue width from the DMA beat stream.
    cls = next(n for n in trees["core.py"].body if isinstance(n, ast.ClassDef) and n.name == "Core")
    tick = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "tick")
    increments = [n.value.value for n in ast.walk(tick)
                  if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Attribute)
                  and n.target.attr == "cycle_count" and isinstance(n.op, ast.Add)
                  and isinstance(n.value, ast.Constant) and type(n.value.value) is int]
    if increments != [1] or interval != 1:
        raise UnsupportedDMAAdmission("unsupported DMA admission clock/issue progression")
    return shapes, dict(ordinary_gap=increments[0], first_engine_age=startup["load"],
                        store_read_to_request_cycles=startup["store"] - startup["load"])


def derive_dma_admission(model, per_op):
    """Return frozen, unvalidated admission facts; no Core allocation or ticks."""
    model = safe_path(model)
    paths = {name: safe_path(model / "npu_model" / "hardware" / name) for name in SOURCE_NAMES}
    sources = {name: path.read_text() for name, path in paths.items()}
    shapes, issue = review_sources(sources)
    config = per_op["model"]["configuration"]
    configuration = {name: config[key] for name, key in (
        ("num_channels", "dma_num_channels"), ("beat_bytes", "dma_beat_bytes"),
        ("max_in_flight", "dma_max_in_flight"), ("max_transfer_bytes", "dma_max_transfer_bytes"),
        ("vmem_bank_bytes", "vmem_bank_bytes"))}
    if any(type(value) is not int or value < 1 for value in configuration.values()):
        raise UnsupportedDMAAdmission("invalid DMA admission geometry")
    if (configuration["num_channels"] != 8 or configuration["beat_bytes"] != 32
            or configuration["max_transfer_bytes"] != 4096 or configuration["vmem_bank_bytes"] != 262144
            or config["fetch_width"] != 1):
        raise UnsupportedDMAAdmission("unsupported DMA admission geometry")
    frontend = per_op.get("frontend_timing", per_op.get("frontend", {}))
    if frontend and frontend.get("ordinary_issue_gap", frontend.get("ordinary_gap", 1)) != issue["ordinary_gap"]:
        raise UnsupportedDMAAdmission("DMA admission differs from frontend issue gap")
    return dict(schema="atlas.dma_admission.v1", age_convention="issue_edge_zero", contract=CONTRACT,
                source_sha256={name: sha256(sources[name].encode()) for name in SOURCE_NAMES},
                source_shapes=shapes, configuration=configuration, issue=issue,
                semantics=deepcopy(SEMANTICS),
                producer_sources={name: sha256(safe_path(Path(__file__).with_name(name)).read_bytes())
                                  for name in ("atlas_derive_dma_admission.py", "atlas_validate_dma_admission.py")},
                validation=dict(passed=False, report_sha256="0" * 64, positive_runs=0,
                                negative_runs=0, numerical_checks=0))
