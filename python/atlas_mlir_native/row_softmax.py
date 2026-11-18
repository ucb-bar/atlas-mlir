"""Selected Atlas primitives and input qualification for whole-row softmax."""

from __future__ import annotations

import struct
from fractions import Fraction

from merlin.semantic_compiler.model import TensorType
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.semantic_compiler.tiled_softmax import (
    decompose_row_softmax,
    materialize_row_softmax_inputs,
)

from .layout import pack_bf16_pair_halves

SOFTMAX_POLICY = "bf16-tiled-row-softmax-ltr-fp32-rne-chop-rcp-lut-v1"
MASKED_SOFTMAX_POLICY = "bf16-tiled-masked-row-softmax-zero-neginf-v1"
GUARDED_SOFTMAX_POLICY = "bf16-tiled-masked-row-softmax-empty-zero-v1"
SOFTMAX_DOMAIN = "bf16-signed-zero-or-normal-exp8-to132-abs-at-most32-v1"
ROWSPAN80_DOMAIN = "bf16-row-relative-span80-pad32-v1"
ROWSPAN80_MAX_COLUMNS = 1024


def score_domain(request):
    attrs = dict(request.nodes[0].attrs)
    domain = attrs.get("input_domain")
    if domain not in {SOFTMAX_DOMAIN, ROWSPAN80_DOMAIN}:
        raise ValueError("unsupported row-softmax score domain")
    return domain


def prepare_row_softmax(request):
    from .binding import TARGET_IDENTITY, AtlasNativeBinding

    masked = request.nodes[-1].op == "masked_softmax"
    empty_rows_zero = masked and request.nodes[-1].type.numerical_policy == GUARDED_SOFTMAX_POLICY
    if (
        request.target_identity != TARGET_IDENTITY
        or request.lowering_policy != "diagnostic"
        or len(request.nodes) != (3 if masked else 2)
        or len(request.nodes[0].type.shape) != 2
        or request.nodes[0].attrs not in {(("input_domain", SOFTMAX_DOMAIN),), (("input_domain", ROWSPAN80_DOMAIN),)}
        or request.nodes[0].type.dtype != "bf16"
        or request.nodes[0].type.shape[1] > (
            ROWSPAN80_MAX_COLUMNS
            if request.nodes[0].attrs == (("input_domain", ROWSPAN80_DOMAIN),)
            else 65536
        )
        or request.boundary_layouts
        or request.output_layouts
    ):
        raise NativeCompilationError(
            "unsupported_semantics",
            "row softmax requires its explicit bounded BF16 policy",
        )
    catalog = {descriptor.name: descriptor for descriptor in AtlasNativeBinding.profile().descriptors}
    names = {
        "row_max": "atlas.vpu_row_max_bf16_tile",
        "combine_max": "atlas.vpu_max_bf16_tile",
        "subtract": "atlas.vpu_sub_bf16_rne_tile",
        "exp": "atlas.vpu_exp_bf16_tile",
        "row_sum": "atlas.vpu_row_sum_bf16_tile",
        "combine_sum": "atlas.vpu_add_bf16_rne_tile",
        "reciprocal": "atlas.vpu_recip_bf16_wide_tile",
        "multiply": "atlas.vpu_mul_bf16_zero_tile",
    }
    shape = tuple(axis.minimum for axis in catalog[names["row_max"]].output_axis_bounds)
    return decompose_row_softmax(
        request,
        tile_type=TensorType(shape, "bf16", "raw-bit-pattern"),
        operations={role: catalog[name] for role, name in names.items()},
        policy=GUARDED_SOFTMAX_POLICY if empty_rows_zero else MASKED_SOFTMAX_POLICY if masked else SOFTMAX_POLICY,
        pad_element=struct.pack("<H", 0xC200),
        empty_rows_zero=empty_rows_zero,
    )


def prepare_inputs(plan, source: bytes, keep: bytes | None = None):
    """Check score bounds and pack explicit padding into BF16 pair halves.

    Nonzero scores have magnitude at least 2^-119, so a nonzero difference
    of admitted BF16 scores is normal. The original absolute domain keeps
    differences in [-64,0]. The separately selected row-relative domain
    proves kept maxima in [-32,48] and exact spans at most 80, keeping both
    valid and padding differences in [-80,0]. Exponentials remain normal.
    At most 1024 columns in the row-relative domain keep the minimum final
    probability normal under the selected reciprocal and product primitives.
    Padding is masked after exponentiation and cannot affect normalization.
    """
    if plan != prepare_row_softmax(plan.source):
        raise ValueError("softmax tiling differs from the selected source and primitive catalog")
    if type(source) is not bytes or len(source) != plan.shape[0] * plan.shape[1] * 2:
        raise ValueError("softmax source bytes differ from its complete shape")
    relative = score_domain(plan.source) == ROWSPAN80_DOMAIN
    for (code,) in struct.iter_unpack("<H", source):
        magnitude = code & 0x7FFF
        limit, exponent_limit = (0x4300, 134) if relative else (0x4200, 132)
        if magnitude > limit or (magnitude and not 8 <= code >> 7 & 255 <= exponent_limit):
            raise ValueError("softmax score is outside the selected bounded normal/signed-zero domain")
    if plan.masked and keep is not None:
        for at in range(0, len(source), 2):
            if keep[at : at + 2] == b"\0\0" and source[at : at + 2] != plan.pad_element:
                raise ValueError("masked scores must use the selected finite padding value")
    if relative:
        validate_row_span80(source, plan.shape, keep if plan.masked else None, empty_rows_zero=plan.empty_rows_zero,
                            masked=plan.masked)
    return tuple(
        {name: pack_bf16_pair_halves(payload) for name, payload in group.items()}
        for group in materialize_row_softmax_inputs(plan, source, keep)
    )


def validate_row_span80(source, shape, keep=None, *, empty_rows_zero=False, masked=False):
    """Prove row-relative bounds exactly on already-encoded BF16 values.

    Fraction prevents a tiny positive maximum and -80 minimum from silently
    passing when their mathematical difference rounds to 80 in host float.
    The caller validates finite code ranges before invoking this check.
    """
    rows, columns = shape
    if columns > ROWSPAN80_MAX_COLUMNS:
        raise ValueError("row-relative span80 domain supports at most 1024 columns")
    if masked and (type(keep) is not bytes or len(keep) != len(source)):
        raise ValueError("row-relative masked softmax needs a complete keep buffer")
    for row in range(rows):
        values = []
        for column in range(columns):
            at = (row*columns+column)*2
            if masked:
                code = int.from_bytes(keep[at:at+2], "little")
                if code not in {0, 0x3f80}:
                    raise ValueError("row-relative softmax keep is not zero or one")
                if not code:
                    continue
            bits = int.from_bytes(source[at:at+2], "little") << 16
            values.append(struct.unpack("<f", struct.pack("<I", bits))[0])
        if not values:
            if empty_rows_zero:
                continue
            raise ValueError("row-relative softmax needs a nonempty row")
        maximum, minimum = max(values), min(values)
        if not -32 <= maximum <= 48 or Fraction(maximum)-Fraction(minimum) > 80:
            raise ValueError("softmax row is outside the selected row-relative span80 domain")
