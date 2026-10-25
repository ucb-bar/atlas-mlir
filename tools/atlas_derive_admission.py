"""Extract engine initiation predicates, independently of overlap pair coverage."""
from __future__ import annotations

import ast
from copy import deepcopy

from atlas_derive_overlap import extract_contract, safe_path, _source, _named, _method
from atlas_derive_memory import _source as memory_source
from atlas_derive_frontend import derive_frontend
from atlas_model_diagnostics import UnsupportedModelTiming
from atlas_model_schema import sha256
from atlas_model_timing import configuration, source_digest

CORE_INIT_SHAPE = "5abd67591485003d3f345e7b4f15564b3b3d8a4ff8449ed3efd5a392cf03adf9"
CONTRACT = "source-engine-initiation-v1"


class UnsupportedAdmission(UnsupportedModelTiming):
    """The selected source has no reviewed engine admission interpretation."""


def semantic_admission(admission):
    return {key: deepcopy(value) for key, value in admission.items()
            if key not in {"validation", "producer_sources"}}


def _vpu_live_ages(cfg, blocks):
    """Read native write_last, which controls issue liveness and retirement."""
    from atlas_derive_vpu import _plans
    final = {name: operation.write_last for name, _, _, operation in _plans(cfg)}
    if set(final) != {name for name, block in blocks.items() if block["engine"] == "vpu"}:
        raise UnsupportedAdmission("VPU source live endpoint coverage differs from selected operations")
    for name, endpoint in final.items():
        block = blocks[name]
        writes = [value["last_age"] for key, value in block["events"].items()
                  if key.startswith("write")]
        if not writes or max(writes) > endpoint or block["first_free_age"] != endpoint + 1:
            raise UnsupportedAdmission("VPU source live endpoint differs from lifetime or precedes row writes")
    return final


def derive_admission(model, per_op):
    """Pure source derivation; native validation never supplies a predicate."""
    model = safe_path(model)
    try:
        structural = extract_contract(model, per_op)
    except ValueError as error:
        raise UnsupportedAdmission(str(error)) from error
    cfg, config = configuration()
    from npu_model.hardware import lsu, xlu
    # Memory extractor guards full control flow, including pre-progress issue.
    memory_source(lsu)
    memory_source(xlu)
    derive_frontend(cfg)
    _, core, _ = _source(model, "core.py")
    init = _method(_named(core, "Core"), "__init__")
    if sha256(ast.dump(init, include_attributes=False).encode()) != CORE_INIT_SHAPE:
        raise UnsupportedAdmission("unsupported engine construction/routing semantics")
    units = cfg.execution_units
    required = {"VectorExecutionUnit", "CrossLaneExecutionUnit", "LoadStoreUnit"}
    if any(list(units.values()).count(name) != 1 for name in required):
        raise UnsupportedAdmission("engine admission requires unique independent native units")
    blocks = {block["name"]: block for block in per_op["op_timing"]}
    final = _vpu_live_ages(cfg, blocks)
    if not final:
        raise UnsupportedAdmission("missing VPU admission coverage")
    source = dict(structural["source_sha256"])
    for name in ("core.py", "xlu.py"):
        source[name] = _source(model, name)[2]
    result = dict(schema="atlas.engine_admission.v1", age_convention="issue_edge_zero",
        contract=CONTRACT,
        model={key: per_op["model"][key] for key in
               ("source_sha256", "configuration_sha256", "numerical_data_sha256")},
        source_contract=dict(source_sha256=source, core_init_shape=CORE_INIT_SHAPE,
                             execution_units=deepcopy(config["execution_units"])),
        vpu=dict(capacity=structural["capacity"]["vpu_slots"],
                 live="age_lt_final_write", final_write_age=final,
                 logic_groups=[["vpu." + name for name in group]
                               for group in structural["vpu"]["logic_groups"]],
                 double=["vpu." + name for name in structural["vpu"]["double"]]),
        xlu=dict(independent_from_vpu=True, initiation="prior_first_free",
                 initiation_age=blocks["xlu.transpose"]["first_free_age"]),
        vlsu=dict(independent_load_store_paths=True, initiation="same_path_prior_first_free",
                  initiation_ages={name: blocks[name]["first_free_age"]
                                   for name in ("vlsu.vload", "vlsu.vstore")}),
        scalar_lsu=dict(load_response_gap=structural["scalar_lsu"]["load_response_gap"],
                        store_busy="none", independent_vector_paths=True),
        policy=dict(mreg_ports="vpu_same_logical_register_row_read_coalescing", unknown_vmem="aliases_all_banks",
                    scalar_lsu="existing_alias_dependency_and_writeback_checks", mxu="existing_source_overlap_rules",
                    waits="invalidate_live_admission"),
        validation=dict(passed=False))
    if source_digest(model)[0] != per_op["model"]["source_sha256"]:
        raise UnsupportedAdmission("model changed during engine admission derivation")
    return result


def vpu_blocked(admission, issued, name, cycle):
    """Reference structural predicate; excludes operand/resource obligations."""
    vpu = admission["vpu"]
    live = [old for old, issue in issued
            if cycle - issue < vpu["final_write_age"][old]]
    return bool(live and (len(live) >= vpu["capacity"] or name in vpu["double"] or
                any(old in vpu["double"] or old == name or
                    any(old in group and name in group for group in vpu["logic_groups"])
                    for old in live)))
