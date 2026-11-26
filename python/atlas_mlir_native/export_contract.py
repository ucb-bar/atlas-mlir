"""Decode the exporter contract without enabling the legacy application menu.

Inputs are producer policy records; the caller must authenticate their bytes and
capture bindings separately. Reviewed MXU selection is conditional on the exact
exporter schema and numeric mapping, not graph names or a model-quality claim.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class CheckedExportContract:
    """Immutable capture-scoped producer fields and recorded schema defaults."""

    target_identity: str
    canonical_export_sha256: str
    export_json: str
    allowed_mxu: tuple[str, ...]
    defaults_json: str

    @property
    def defaulted_fields(self):
        return tuple(json.loads(self.defaults_json))

    @property
    def sites(self):
        """Return a fresh copy; mutation cannot change the checked contract."""
        return json.loads(self.export_json)["sites"]

    def record(self):
        return {
            "schema": "atlas.checked-export-contract.v1",
            "target_identity": self.target_identity,
            "canonical_export_sha256": self.canonical_export_sha256,
            "export": json.loads(self.export_json),
            "allowed_mxu": list(self.allowed_mxu),
            "schema_defaults": json.loads(self.defaults_json),
            "status": "exporter_contract_checked; source_and_runtime_admission_required",
        }


def _validate_export_fields(export, *, captured=None):
    """Keep the producer-specific legacy field checks and diagnostics in one place."""
    from .binding import MXU1_CAP448_POLICY
    from .numerical_contracts import describe_numerical_families
    from .policy_derivation import _check_allowed_mxu, _check_mxu, _fail, _in_capture

    family = next(
        row
        for row in describe_numerical_families()
        if row.family == "atlas.ordered-quantized-contraction"
    )
    encoding = json.loads(dict(family.numerical_contract)["encoding"])
    expected = {
        "dtype": "f8E4M3",
        "exponent_bits": encoding["exponent_bits"],
        "mantissa_bits": encoding["mantissa_bits"],
        "bias": encoding["bias"],
        "exponent_range": encoding["normal_exponent_range"],
        "exponent_15_finite": encoding["exponent_15_finite"],
        "finite_max": float(encoding["finite_max"]),
        "min_normal": encoding["min_normal"],
        "reserved_codes": encoding["reserved_codes"],
        "subnormal_flush": True,
        "signed_zero": False,
        "rounding": "rne",
        "saturate": True,
    }
    operand = export.get("operand", {})
    for key, value in expected.items():
        if operand.get(key) != value:
            _fail(f"operand.{key}", f"Atlas cap448 family requires {value!r}")
    if export.get("accumulate", {}).get("readout") != "bf16":
        _fail("accumulate.readout", "Atlas family requires BF16 readout")
    if export.get("scale", {}).get("encoding") != "e8m0_exponent_int32":
        _fail("scale.encoding", "requires signed integer power-of-two exponents")
    if export.get("accumulate", {}).get("unit") != "MXU":
        _fail("accumulate.unit", "requires declared MXU contraction")
    # Framework exports declare an on-chip assumption; hardware exports need the exact family.
    allowed_mxu = _check_mxu(export["accumulate"], MXU1_CAP448_POLICY)
    chip = export.get("runtime", {}).get("chip_options", {})
    if chip.get("chip_vadd", "rn") != "rn":
        _fail("runtime.chip_options.chip_vadd", "declared BF16 family requires RNE")
    sites = export.get("sites")
    if not isinstance(sites, dict) or not sites:
        _fail("sites", "requires a nonempty exported site inventory")
    if captured is not None:
        sites = _in_capture(sites, captured)
    for name, site in sites.items():
        if site.get("allowed_mxu") is not None:
            _check_allowed_mxu(
                f"sites.{name}.allowed_mxu", site["allowed_mxu"], allowed_mxu
            )
        if site.get("residual_passes") is not None:
            _fail(f"sites.{name}.residual_passes", "no declared residual-pass family")
        # An unquantized BF16 linear or Conv2d site stays on the host. Its captured
        # operators are admitted or refused one by one by the host families (an
        # unfolded Conv2d by the host BF16 contraction); it adds no MXU work.
        # Conv2d: intake only. No front-door kit holds a host BF16 Conv2d yet, and
        # the independent reference has no aten convolution handler, so no such
        # kit can pass its exact check until that handler exists.
        if site.get("status") in {"skipped", "host"} and site.get("kind", "").lower() in {"linear", "conv2d"}:
            if site.get("weight", {}).get("dtype") != "bfloat16":
                _fail(f"sites.{name}.weight.dtype", "host linear and conv2d families require BF16")
            continue
        if site.get("kind") == "conv2d":
            # qconv2d consumes already unfolded codes and normalizes to the
            # same qlinear family. Im2col/cast stay in the captured prefix;
            # normalization still checks ABI, shapes, and operand roles.
            # Temporary prose match until the exporter declares structured
            # fields for the unfolding and input-cast prefix.
            conv = site.get("conv")
            lowering = conv.get("lowering") if isinstance(conv, dict) else None
            if (
                not isinstance(lowering, str)
                or re.fullmatch(
                    r"im2col; per-tensor cast of the unfolded input(?: \([^()\n]*\))?",
                    lowering,
                )
                is None
            ):
                _fail(
                    f"sites.{name}.conv.lowering",
                    "requires declared im2col and per-tensor unfolded-input cast for qlinear",
                )
        # A quantized embedding is storage only: the captured graph gathers code
        # and exponent rows and decodes them with ordinary operators, which the
        # existing host families admit or refuse per operator. No device family.
        if site.get("kind") not in {"linear", "attention", "conv2d", "embedding"}:
            _fail(
                f"sites.{name}.kind",
                "no declared implementation for this exported site kind",
            )
        if site.get("status") != "quantized":
            _fail(f"sites.{name}.status", "requires explicit quantized site contract")
    return sites, tuple(allowed_mxu)


def decode_export_contract(export, *, captured=None):
    """Return checked scoped fields and separately selected reviewed precision IDs.

    ``captured`` is the existing ``_capture_bindings`` native-site/bound-tensor
    pair. ``None`` validates every site. Optional unit/K-order fields retain the
    execution-policy-v1 defaults checked by ``_check_mxu`` and recorded below.
    No device route, broad host policy or precision change outside the reviewed
    MXU1 exporter mapping is enabled by this decoder.
    """
    from .binding import MXU1_CAP448_POLICY, TARGET_IDENTITY
    from .numerical_contracts import MXU1_PRECISION_AUTHORIZATION
    from .policy_derivation import K_ORDER, MXU1_ARITHMETIC, _fail

    if (
        not isinstance(export, dict)
        or export.get("schema") != "atlas_torchao.execution_policy.v1"
    ):
        _fail("policy.schema", "requires atlas_torchao.execution_policy.v1")
    if export.get("format_version") != 1:
        _fail("policy.format_version", "requires version 1")
    sites, allowed = _validate_export_fields(export, captured=captured)
    accumulation = export["accumulate"]
    defaults = {
        "accumulate." + key: value
        for key, value in {
            "allowed_mxu": list(allowed),
            "allowed_mxu_semantics": None,
            "units": {"mxu1": {"mxu_model": MXU1_ARITHMETIC}},
            "k_order": K_ORDER,
        }.items()
        if accumulation.get(key) is None
    }
    if export.get("runtime", {}).get("chip_options", {}).get("chip_vadd") is None:
        defaults["runtime.chip_options.chip_vadd"] = "rn"
    scoped = {**export, "sites": sites}
    checked = CheckedExportContract(
        TARGET_IDENTITY,
        sha256(_json(export).encode()).hexdigest(),
        _json(scoped),
        allowed,
        _json(defaults),
    )
    # Embedding storage and skipped host BF16 linears contribute no MXU work.
    requires_mxu = any(
        site["kind"] in {"linear", "attention", "conv2d"}
        and site.get("status") == "quantized"
        for site in sites.values()
    )
    if MXU1_PRECISION_AUTHORIZATION != MXU1_CAP448_POLICY:
        _fail(
            "accumulate.mxu",
            "reviewed numerical declaration differs from the selected MXU1 family",
        )
    authorizations = (MXU1_PRECISION_AUTHORIZATION,) if requires_mxu else ()
    return checked, authorizations
