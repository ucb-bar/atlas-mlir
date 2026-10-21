"""Selected Atlas primitives and input qualification for whole-row softmax."""

from __future__ import annotations

import struct

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


def prepare_row_softmax(request):
    from .binding import TARGET_IDENTITY, AtlasNativeBinding

    masked = request.nodes[-1].op == "masked_softmax"
    empty_rows_zero = masked and request.nodes[-1].type.numerical_policy == GUARDED_SOFTMAX_POLICY
    if (
        request.target_identity != TARGET_IDENTITY
        or request.lowering_policy != "diagnostic"
        or len(request.nodes) != (3 if masked else 2)
        or len(request.nodes[0].type.shape) != 2
        or request.nodes[0].attrs != (("input_domain", SOFTMAX_DOMAIN),)
        or request.nodes[0].type.dtype != "bf16"
        or request.nodes[0].type.shape[1] > 65536
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
    of two admitted BF16 scores is normal. The global maximum keeps subtracted
    scores in [-64, 0]. Exponentials
    remain normal, and the global denominator lies in [1, 65536]. Padded
    columns are masked after exponentiation and cannot affect normalization.
    """
    if plan != prepare_row_softmax(plan.source):
        raise ValueError("softmax tiling differs from the selected source and primitive catalog")
    if type(source) is not bytes or len(source) != plan.shape[0] * plan.shape[1] * 2:
        raise ValueError("softmax source bytes differ from its complete shape")
    for (code,) in struct.iter_unpack("<H", source):
        magnitude = code & 0x7FFF
        if magnitude > 0x4200 or (magnitude and not 8 <= code >> 7 & 255 <= 132):
            raise ValueError("softmax score is outside the selected bounded normal/signed-zero domain")
    if plan.masked and keep is not None:
        for at in range(0, len(source), 2):
            if keep[at : at + 2] == b"\0\0" and source[at : at + 2] != plan.pad_element:
                raise ValueError("masked scores must use the selected finite padding value")
    return tuple(
        {name: pack_bf16_pair_halves(payload) for name, payload in group.items()}
        for group in materialize_row_softmax_inputs(plan, source, keep)
    )
