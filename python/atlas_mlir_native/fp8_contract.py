"""Admission-only Atlas facts for Merlin's generic scaled-FP8 contract.

This helper neither lowers operations nor qualifies device execution. Source
names and the model47 numerical policy stay here, outside the generic compiler.
"""
from dataclasses import dataclass

import numpy as np

from merlin.compiler.quantization.contraction.scaled_fp8 import (
    ContractionNumerics, ScaledFP8Spec,
)


OPS = ("atlas.qlinear", "atlas.qk_matmul", "atlas.pv_matmul")
# atlas-torchao's unpacker decodes exponent field 0 and the codes 0x7f and 0xff as signed zero.
EXPORT_ENCODING = "e4m3fn-zero-flush"


def atlas_numerics(op_name, *, output_dtype="bf16", bias=False):
    """Declare the selected diagnostic MXU1 route and source epilogue order.

    This declaration does not qualify a fused device bias or activation. The
    value admission below remains rank-two and unbiased. An Atlas operation
    name alone does not prove equality to framework accumulation.
    """
    if op_name not in OPS:
        raise ValueError("unsupported Atlas FP8 contraction operation")
    if output_dtype not in {"bf16", "f32"} or type(bias) is not bool:
        raise ValueError("Atlas numerical declaration requires BF16 or f32 output and explicit bias presence")
    stages = ("k32-reset", "ordered-bf16-seeded-continuation",
              "bf16-readout", "restore-signed-exponent-sum",
              "output-bf16" if output_dtype == "bf16" else "exact-widen-f32")
    if bias:
        stages += ("source-bias-after-output-cast",)
    return ContractionNumerics(
        accumulation="mxu1-model47-anchor-bf16-ordered-k32",
        readout="bf16-normal-exponent-restore",
        stages=stages,
        bias_rounding=("bf16-rne" if output_dtype == "bf16" else "f32-rne") if bias else "none",
    )


@dataclass(frozen=True)
class AdmittedNTContraction:
    """Logical decoded operands and policy; no device arithmetic prediction."""

    lhs: np.ndarray
    rhs: np.ndarray
    numerics: ContractionNumerics


def _operand(codes, exponents):
    if (not isinstance(codes, np.ndarray) or codes.dtype != np.int8
            or codes.ndim != 2 or not all(codes.shape)):
        raise ValueError("Atlas operand requires nonempty rank-two i8 storage")
    raw = codes.view(np.uint8)
    if np.any(~((raw == 0) | ((raw >= 8) & (raw <= 126))
                | ((raw >= 136) & (raw <= 254)))):
        raise ValueError("Atlas operand requires finite normal cap448 codes or positive zero")
    if not isinstance(exponents, np.ndarray) or exponents.dtype != np.int32:
        raise ValueError("Atlas exponents require signed i32 storage")
    if exponents.shape == ():
        axes, compact = (), exponents
        rows = np.broadcast_to(exponents, (codes.shape[0],))
    elif exponents.shape == (codes.shape[0], 1):
        axes, compact = (0,), exponents[:, 0]
        rows = compact
    elif exponents.shape == (1, 1):
        axes, compact = (), exponents.reshape(())
        rows = np.broadcast_to(compact, (codes.shape[0],))
    else:
        raise ValueError("Atlas exponents must be scalar or per-row and cannot vary over K")
    if np.any((exponents < -127) | (exponents > 127)):
        raise ValueError("Atlas operand exponent leaves [-127,127]")
    return ScaledFP8Spec(exponent_axes=axes, encoding=EXPORT_ENCODING).decode(codes, compact), rows


def admit_rank2_nt(op_name, lhs_codes, lhs_exponents, rhs_codes, rhs_exponents):
    """Admit unbiased NT operands, preserving codes and signed exponent data.

    Inputs use Atlas source shapes [M,K], [N,K], with scalar, [1,1],
    or [rows,1] exponent tensors. No per-K scales, bias, activation, folded
    pop shift, or predicted MXU result is represented. Admitting exponent
    sums does not prove that an actual restored BF16 result stays normal.
    """
    numerics = atlas_numerics(op_name)
    lhs, lhs_rows = _operand(lhs_codes, lhs_exponents)
    rhs, rhs_rows = _operand(rhs_codes, rhs_exponents)
    if lhs.shape[1] != rhs.shape[1]:
        raise ValueError("Atlas NT operands require matching K")
    sums = lhs_rows.astype(np.int64)[:, None] + rhs_rows.astype(np.int64)[None, :]
    if np.any((sums < -133) | (sums > 127)):
        raise ValueError("Atlas readout exponent sum leaves [-133,127]")
    return AdmittedNTContraction(lhs, rhs, numerics)
