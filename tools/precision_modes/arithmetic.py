"""Offline CPU emulation of the Atlas MXU arithmetic. No compiler, runtime, or device program uses this code.

Operands get a BF16 front cast and Atlas E4M3 packing with a power-of-two scale. Exact products are added in
ascending K with BF16 nearest-even rounding after every MAC. Recipes start a fresh chain for each K chunk and weight
term, add the scaled chains in FP64, and round once, directly, to BF16.
"""

import ctypes
import fnmatch
import functools
import json
import os
from pathlib import Path

import torch


def direct_bf16(x):
    """Binary64 to BF16 nearest-even in one step; a float32 bridge can double round. Zero becomes +0."""
    bits = x.double().contiguous().view(torch.int64)
    bits = (bits + 0xfffffffffff + ((bits >> 45) & 1)) & -(1 << 45)
    return torch.where(x == 0, 0.0, bits.view(torch.float64)).float()


def scale(x, axis="tensor"):
    """Smallest power-of-two exponent keeping the BF16 maximum of each tensor, row, or column within 448."""
    magnitude = x.bfloat16().float().abs()
    maximum = magnitude.max() if axis == "tensor" else magnitude.amax(dim=1 if axis == "row" else 0, keepdim=True)
    fraction, power = torch.frexp(maximum)
    return torch.where(maximum == 0, 0, power - 9 + (fraction > 0.875).int())


def quantize(x, exponent):
    """Decoded Atlas E4M3 of BF16(x) / 2^exponent: nearest even, BF16 subnormals and values below 2^-6 flush to
    +0, and magnitudes saturate at 448."""
    front = x.bfloat16().double()
    fraction, power = torch.frexp(torch.ldexp(front, -torch.as_tensor(exponent)))
    value = torch.ldexp(torch.round(fraction * 16), power - 4).clamp(-448, 448)
    return torch.where((value.abs() < 2**-6) | (front.abs() < 2**-126), 0.0, value)


def pack(x, axis="tensor"):
    exponent = scale(x, axis)
    return quantize(x, exponent), exponent


def two_term(weight, low_axis="tensor"):
    """The default tensor-scaled FP8 weight, plus FP8 of its FP32 residual with a separate scale."""
    high, exponent = pack(weight)
    return (high, exponent), pack((weight.double() - torch.ldexp(high, exponent)).float(), low_axis)


@functools.cache
def _library():
    library = ctypes.CDLL(os.environ["ATLAS_PRECISION_MAC_LIBRARY"])
    library.ordered.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    return library.ordered


def ordered_mm(a, b):
    """Unscaled ordered MAC chain of decoded FP8 operands, through the compiled ordered_mac.cpp."""
    a, b = a.double().contiguous(), b.double().contiguous()
    out = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float64)
    _library()(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.shape[0], a.shape[1], b.shape[1])
    return out


def contraction(a, b, ea, eb, k_chunk=0, low=None):
    """Fresh chains per K chunk, high then low term, scaled and added in ascending order in FP64."""
    total = 0
    chunk = k_chunk or a.shape[1]
    for start in range(0, a.shape[1], chunk):
        for weight, exponent in [(b, eb)] + ([low] if low else []):
            total = total + torch.ldexp(ordered_mm(a[:, start:start + chunk], weight[start:start + chunk]), ea + exponent)
    return direct_bf16(total)


def settings(args, site):
    """Resolve a recipe at one site; unselected sites use the frozen default."""
    mode = args.precision_mode
    selection = getattr(args, "sites", "all")
    if selection.startswith("@"):
        contents = Path(selection[1:]).read_text()
        selection = (json.loads(contents) if contents.lstrip().startswith("[")
                     else contents.split())
    if isinstance(selection, str):
        selection = selection.split(",")
    aliases = {
        "head": site == "lm_head",
        "attention-head": "self_attn." in site or site == "lm_head",
        "nonmlp": "mlp." not in site,
        "mlp": "mlp." in site,
        "down": site.endswith("mlp.down_proj"),
    }
    selected = any(pattern == "all" or aliases.get(pattern, False) or
                   fnmatch.fnmatchcase(site, pattern) for pattern in selection)
    finer = mode == "finer-scales" or getattr(args, "finer_scales", False)
    keep_finer = mode == "k-split" and finer
    if mode == "default" or (not selected and not keep_finer):
        return "tensor", "tensor", 0, False
    a_axis = (getattr(args, "activation_scale", None) or
              ("row" if finer else "tensor"))
    b_axis = (getattr(args, "weight_scale", None) or
              ("column" if finer else "tensor"))
    chunk = args.k_chunk if mode == "k-split" and selected else 0
    return a_axis, b_axis, chunk, mode == "two-term"
