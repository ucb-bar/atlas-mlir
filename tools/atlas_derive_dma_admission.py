"""Source qualification for asynchronous DMA issue; never a completion estimate."""
from __future__ import annotations

from copy import deepcopy

from atlas_derive_curve import review_dma_sources, UnsupportedCurve
from pathlib import Path
from atlas_model_diagnostics import UnsupportedModelTiming
from atlas_model_schema import DMA_SOURCE_SHAPES, canonical, sha256


CONTRACT = "source-asynchronous-dma-admission-v1"
SOURCE_NAMES = tuple(DMA_SOURCE_SHAPES)
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
    try:
        return review_dma_sources(sources)
    except UnsupportedCurve as error:
        raise UnsupportedDMAAdmission(str(error)) from error


def derive_dma_admission(model, per_op):
    """Return frozen, unvalidated admission facts; no Core allocation or ticks."""
    model = Path(model)
    sources = {name: (model / "npu_model" / "hardware" / name).read_text() for name in SOURCE_NAMES}
    shared = review_sources(sources)
    timing = per_op.get("dma_timing", {})
    if timing.get("source_contract") != shared:
        raise UnsupportedDMAAdmission("DMA admission differs from authoritative DMA source/phase contract")
    shapes, issue = shared["source_shapes"], shared["issue"]
    if (timing.get("startup_cycles") != dict(load=issue["first_engine_age"],
            store=issue["first_engine_age"] + issue["store_read_to_request_cycles"])
            or timing.get("issue_interval_cycles") != issue["ordinary_gap"]
            or timing.get("proposal_contract", {}).get("source_evidence", {}).get("edge_order") != shared["edge_order"]):
        raise UnsupportedDMAAdmission("DMA admission differs from authoritative launch/staging/phase fields")
    config = per_op["model"]["configuration"]
    configuration = {name: config[key] for name, key in (
        ("num_channels", "dma_num_channels"), ("beat_bytes", "dma_beat_bytes"),
        ("max_in_flight", "dma_max_in_flight"), ("max_transfer_bytes", "dma_max_transfer_bytes"),
        ("vmem_bank_bytes", "vmem_bank_bytes"))}
    for key in ("num_channels", "beat_bytes", "max_in_flight", "max_transfer_bytes"):
        if timing.get(key) != configuration[key]:
            raise UnsupportedDMAAdmission("DMA admission differs from authoritative geometry: " + key)
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
                source_contract_sha256=sha256(canonical(shared)),
                semantics=deepcopy(SEMANTICS))
