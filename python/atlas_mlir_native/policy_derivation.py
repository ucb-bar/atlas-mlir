"""Export-contract checks shared by the generated export policy.

PolicyDerivationError names the export field that cannot support the selected
Atlas implementation. The MXU checks hold the reviewed mapping from exporter
arithmetic to the one Atlas MXU family; _capture_bindings and _in_capture name
the export sites a capture executes. No model names or capture schedules occur here.
"""
from __future__ import annotations

import json


class PolicyDerivationError(ValueError):
    """An export field cannot support the selected Atlas implementation."""

    def __init__(self, field, reason):
        self.field = field
        super().__init__(f"{field}: {reason}")


def _fail(field, reason):
    raise PolicyDerivationError(field, reason)


# Reviewed mapping from the exporter's MXU arithmetic to the one Atlas MXU family.
# "ipt" is the exporter's inner-product tree model of MXU1. MXU1 model47 matches it
# bit for bit on every admitted single product and on sampled long K chains: the
# same anchor, truncating alignment, 32-bit wrap, and one RNE BF16 conversion per
# instruction. The K order is the one the already-quantized family executes:
# ascending K in the stored operand layout, 32-wide instructions each seeded by the
# previous BF16 partial, and a short final instruction padded with positive zero.
# MXU0's systolic array ("sa") rounds after every MAC and has no reviewed family.
# The opt-in keys (allowed_mxu, allowed_mxu_semantics, k_order, units, and a
# site's allowed_mxu) follow an unmerged exporter proposal; absent keys mean mxu1.
MXU_UNITS = ("mxu0", "mxu1")
MXU1_ARITHMETIC = "ipt"
K_ORDER = {"order": "ascending", "tile": 32}


def _check_allowed_mxu(field, allowed, within=MXU_UNITS):
    """An allowed-unit list is a preference order; Atlas runs it on MXU1 when listed."""
    if not isinstance(allowed, list) or not allowed or not all(isinstance(unit, str) for unit in allowed):
        _fail(field, "requires a nonempty list of MXU unit names")
    if len(set(allowed)) != len(allowed) or set(allowed) - set(within):
        _fail(field, f"units {allowed} must be distinct and within {list(within)}")
    if "mxu1" not in allowed:
        _fail(field, "export excludes mxu1; Atlas implements this accumulation only on MXU1 model47 "
              "and has no reviewed mxu0 family")


def _check_mxu(accumulation, family):
    """Admit the export's MXU arithmetic only through the reviewed MXU1 mapping."""
    mxu = accumulation.get("mxu")
    if mxu != family and not (accumulation.get("exported_mode_accumulation") == "framework"
                              and mxu == MXU1_ARITHMETIC):
        _fail("accumulate.mxu", f"no reviewed Atlas MXU family for {mxu!r}: admits ipt (MXU1 model47) "
              "from framework exports, or the exact declared family")
    allowed = accumulation.get("allowed_mxu")
    if allowed is not None:
        _check_allowed_mxu("accumulate.allowed_mxu", allowed)
    semantics = accumulation.get("allowed_mxu_semantics")
    if semantics is not None and not isinstance(semantics, str):
        _fail("accumulate.allowed_mxu_semantics", "requires the exporter's descriptive sentence")
    units = accumulation.get("units")
    if units is not None:
        mxu1 = units.get("mxu1") if isinstance(units, dict) else None
        model = mxu1.get("mxu_model") if isinstance(mxu1, dict) else None
        if not isinstance(units, dict) or set(units) - set(MXU_UNITS) or model != MXU1_ARITHMETIC:
            _fail("accumulate.units", f"requires mxu1 with mxu_model {MXU1_ARITHMETIC!r}, "
                  f"the only arithmetic MXU1 model47 matches, and no unit outside {list(MXU_UNITS)}")
    k_order = accumulation.get("k_order")
    if k_order is not None and (not isinstance(k_order, dict)
                                or {key: k_order.get(key) for key in K_ORDER} != K_ORDER):
        _fail("accumulate.k_order", f"MXU1 model47 executes only {K_ORDER}: 32-wide instructions each "
              "continuing from the previous BF16 partial, a short final one zero-padded")
    return allowed or MXU_UNITS


def _capture_bindings(capture, receipt, quantization):
    """Name the export sites this capture executes, from the adapter's records.

    A site is in the capture when the adapter recorded a native call for it, or
    when the capture binds a tensor the site declares. Sites outside a boundary
    capture contribute no arithmetic and are not checked against target families.
    """
    from merlin.compiler.frontend.inputs import _checked_member

    atlas = quantization.get("atlas")
    buffers = atlas.get("buffers") if isinstance(atlas, dict) else None
    constants = atlas.get("graph_constants", {}) if isinstance(atlas, dict) else None
    records = quantization.get("sites")
    if not isinstance(buffers, dict) or not isinstance(constants, dict) or not isinstance(records, list):
        _fail("capture.provenance", "requires the adapter's site, buffer, and graph-constant records")
    rows = json.loads(_checked_member(capture, receipt, "weights.safetensors.manifest.json"))
    if not isinstance(rows, dict):
        _fail("capture.provenance", "requires the captured weight manifest")
    bound = set()
    for row in rows.values():
        if not isinstance(row, dict) or row.get("kind") == "input":
            continue
        # Graph constants are lifted literals, never export state a site declares.
        if row.get("kind") == "constant" and row.get("weight") in constants:
            continue
        record = buffers.get(row.get("weight"))
        if not isinstance(record, dict) or not isinstance(record.get("source_name"), str):
            _fail("capture.provenance", f"bound tensor has no adapter source record: {row.get('weight')}")
        bound.add(record["source_name"])
    native = {row.get("site_id") for row in records
              if isinstance(row, dict) and row.get("execution_route") == "atlas.native_fp8"}
    return native, bound


def _in_capture(sites, captured):
    """Keep the export sites this capture executes.

    A site is in the capture when the adapter recorded a native call for it, or
    when the capture binds a tensor the site declares. Sites outside a boundary
    capture contribute no arithmetic and are not checked against target families.
    """
    native, bound = captured
    if not native <= sites.keys():
        _fail("capture.provenance.sites", "native call names a site absent from the export")

    def declared(site):
        weight = site.get("weight")
        return {weight.get(key) for key in ("tensor", "codes", "scale_exp")} if isinstance(weight, dict) else set()
    return {name: site for name, site in sites.items() if name in native or declared(site) & bound}
