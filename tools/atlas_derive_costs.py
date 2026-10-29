"""Source-derived operands for representative compiler proposal costs."""
from __future__ import annotations

from atlas_model_diagnostics import UnsupportedModelTiming
from atlas_search_policy import POLICY


class UnsupportedCosts(UnsupportedModelTiming):
    """The selected model has no reviewed operand-independent timing class."""


def derive_costs(cfg, architecture, isa_metadata):
    from atlas_derive_vocabulary import _parameters
    from npu_model import isa_types
    from npu_model.isa import EXU

    # The extraction entry point already guards the complete planners.
    timing_class = dict(kind="guarded_operand_independent_isolated_ages",
                        vpu="mnemonic_latency_and_register_independent_row_predicates",
                        mxu="guarded_age_geometry_expressions",
                        memory="guarded_age_geometry_predicates",
                        interval_scope="selected_representative_pair_only")
    stride = architecture["bf16_registers_per_tile"]
    if stride <= 0:
        raise UnsupportedCosts("invalid source-derived register span")
    # Fixed field order keeps the compiler's preexisting proposal trajectory.
    field_index = dict(rd=0, rs1=1, rs2=2)
    limits = dict(MReg=architecture["mreg_count"], XReg=architecture["xreg_count"],
                  EReg=architecture["ereg_count"], Weight=architecture["weight_buffer_count"])
    probes = {}
    for name, cls in sorted(cfg.isa.operations.items()):
        if name not in isa_metadata and cls.exu != EXU.SCALAR:
            continue
        metadata = isa_metadata.get(name)
        operands = metadata["operands"] if metadata else _parameters(cls)
        mxu = metadata is not None and metadata["engine"] in {"mxu0", "mxu1"}
        pair = [dict(rd=0, rs1=0, rs2=0, imm=0) for _ in range(2)]
        # Unused encoded scalar destinations retain the compiler encoding
        # adapter's field position, while constructors omit nonexistent fields.
        if cls.exu == EXU.SCALAR:
            pair[1]["rd"] = stride * len(field_index)
        for operand in operands:
            resource, field = operand["resource"], operand["compiler_operand"]
            if resource is None and field == "imm":
                continue
            if field not in field_index:
                raise UnsupportedCosts("unsupported compiler cost operand " + field)
            kind = getattr(isa_types, operand["type"])
            lower, upper = kind.lower_bound, kind.upper_bound
            capacity = min(upper, limits.get(resource, upper))
            if lower != 0 or capacity <= 0:
                raise UnsupportedCosts("unsupported operand capacity " + name + "." + field)
            if mxu:
                first, second = 0, stride if resource == "MReg" else 0 if resource == "EReg" else 1
            else:
                first = stride * field_index[field]
                second = stride * (field_index[field] + len(field_index))
            accesses = operand.get("accesses", [])
            extent = max((row["offset"] + row["registers"] for row in accesses), default=1)
            if first + extent > capacity or second + extent > capacity:
                raise UnsupportedCosts("canonical disjoint cost probes exceed " + name + "." + field)
            pair[0][field], pair[1][field] = first, second
        # Constructors enforce the selected ISA's encoded operand domains.
        for probe in pair:
            cls(**{row["name"]: probe[row["compiler_operand"]] for row in operands})
        probes[name] = pair
    return dict(schema="atlas.proposal_costs.v1", timing_class=timing_class, probes=probes,
                placement_policy=POLICY["algorithms"]["proposal_operand_placement"])
