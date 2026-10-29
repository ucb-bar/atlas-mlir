"""CPU emulation only. No compiler, runtime, or device program uses this code.

Operands have a BF16 front cast and normal-only E4M3 packing. Every exact
product is added in ascending K with BF16 RNE after each MAC. Candidate
recipes scale raw partials and merge in FP64, then round directly to BF16.
"""

import ctypes
import fnmatch
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

_mac = None


def configure_library(path):
    """Load the separately compiled, single-threaded ordered-MAC helper."""
    global _mac
    library = ctypes.CDLL(str(Path(path).resolve()))
    _mac = library.ordered
    _mac.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
    _mac.restype = ctypes.c_int


def _finite(x, label):
    if x.device.type != "cpu" or not bool(torch.isfinite(x).all()):
        raise ValueError(f"{label} must contain finite CPU values")


def direct_bf16(x):
    """RNE directly from binary64; a float32 bridge can double round.

    The experiments admit normal finite BF16 outputs and canonical +zero.
    Reject subnormal and overflowing inputs instead of extending that domain.
    """
    x = x.to(torch.float64).contiguous()
    _finite(x, "host merge")
    magnitude = x.abs()
    if bool(((magnitude != 0) & (magnitude < 2.0**-126)).any()):
        raise ValueError("host merge outside normal BF16 domain")
    if bool((magnitude >= 2.0**128 - 2.0**119).any()):
        raise ValueError("host merge overflows BF16")
    bits = x.view(torch.int64)
    sign = bits & (-9223372036854775808)
    mag = bits & 9223372036854775807
    rounded = (mag + 0xfffffffffff + ((mag >> 45) & 1)) & (-35184372088832)
    result = (rounded | sign).view(torch.float64)
    return torch.where(x == 0, 0.0, result).float()


def _exponents(maximum):
    """Smallest power of two keeping each BF16 group maximum <= 448."""
    field, fraction = maximum >> 7, maximum & 127
    top = torch.zeros_like(fraction)
    for bit in range(1, 7):
        top = torch.where(fraction >= (1 << bit), bit, top)
    significand = torch.where(field == 0, fraction << (7 - top),
                              128 + fraction)
    power = torch.where(field == 0, top - 133, field - 127)
    exponent = power - 8 + (significand > 224).int()
    exponent = torch.where(maximum == 0, 0, exponent).to(torch.int32)
    if not bool(((exponent >= -126) & (exponent <= 127)).all()):
        raise ValueError("scale outside admitted exponent range [-126, 127]")
    return exponent


def scale(x, axis="tensor"):
    front = x.to(torch.bfloat16).contiguous()
    _finite(front, "BF16 operand")
    if front.ndim != 2 or not front.numel():
        raise ValueError("operand must be a nonempty matrix")
    words = front.view(torch.int16).int() & 32767
    if axis == "tensor":
        return _exponents(words.max())
    if axis not in ("row", "column", "m32", "n32"):
        raise ValueError(f"unknown scale axis: {axis}")
    rows = axis in ("row", "m32")
    maxima = words.amax(dim=1 if rows else 0)
    if axis in ("m32", "n32"):
        count = maxima.numel()
        maxima = F.pad(maxima, (0, (-count) % 32)).reshape(-1, 32).amax(1)
        maxima = maxima.repeat_interleave(32)[:count]
    exponent = _exponents(maxima)
    return exponent[:, None] if rows else exponent[None, :]


def encode(x, exponent=None):
    """Atlas E4M3 RNE, flush exponent-zero values, saturate at +/-448."""
    front = x.to(torch.bfloat16).contiguous()
    _finite(front, "BF16 operand")
    exponent = scale(front) if exponent is None else torch.as_tensor(exponent)
    if not bool(((exponent >= -126) & (exponent <= 127)).all()):
        raise ValueError("scale outside admitted exponent range [-126, 127]")
    word = front.view(torch.int16).int() & 65535
    field = (word >> 7) & 255
    significand = 128 + (word & 127)
    rounded = (significand + 7 + ((significand >> 4) & 1)) >> 4
    power = field - 127 - exponent + (rounded == 16).int()
    rounded = torch.where(rounded == 16, 8, rounded)
    sign = (word >> 8) & 128
    code = sign | ((torch.clamp(power, min=-6, max=8) + 7) << 3)
    code = code | (rounded - 8)
    code = torch.where((power > 8) | ((power == 8) & (rounded >= 15)),
                       sign | 126, code)
    codes = torch.where((field == 0) | (power < -6), 0, code).to(torch.uint8)
    return codes, exponent


def decode(code):
    words = code.to(torch.int32)
    if bool(((words < 0) | (words > 255) | ((words & 127) == 127)).any()):
        raise ValueError("invalid or reserved FP8 code")
    exponent, fraction = (words >> 3) & 15, words & 7
    values = torch.ldexp((8 + fraction).double(), exponent - 10)
    values = torch.where((words & 128) != 0, -values, values)
    return torch.where(exponent == 0, 0.0, values)


def pack(x, axis="tensor"):
    codes, exponent = encode(x, scale(x, axis))
    return decode(codes), exponent


def two_term(x, low_axis="tensor"):
    """Original tensor-scaled high, plus an encoded FP32 weight residual.

    Subtract from the checkpoint's FP32 weight, before its BF16 front cast,
    matching the explored weight recipe. Only the low scale can be finer.
    """
    high, exponent = pack(x)
    residual = x.double() - torch.ldexp(high, exponent)
    if not torch.equal(residual, residual.float().double()):
        raise ValueError("weight residual is not exactly representable in FP32")
    return (high, exponent), pack(residual.float(), low_axis)


def ordered_mm(a, b, rescale=0):
    """Exact ordered products; the C++ loop is independent of Torch BLAS."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("incompatible contraction matrices")
    if not min(*a.shape, *b.shape) > 0:
        raise ValueError("empty contraction")
    a, b = a.double().contiguous(), b.double().contiguous()
    _finite(a, "decoded activation")
    _finite(b, "decoded weight")
    if float(a.abs().max()) > 448 or float(b.abs().max()) > 448:
        raise ValueError("ordered MAC requires decoded FP8 operands")
    # Every FP8 product lies on the 2^-18 lattice. The conservative bound
    # certifies every ordered prefix, including values not saved by the loop.
    k = a.shape[1]
    bound = math.log2(k) + 2 * math.log2(448) + k * math.log2(1 + 2**-8)
    if bound >= 127:
        raise ValueError("cannot certify normal finite BF16 prefix range")
    if _mac is None:
        path = os.environ.get("ATLAS_PRECISION_MAC_LIBRARY")
        if path is None:
            raise RuntimeError(
                "build ordered_mac.cpp, then pass --ordered-library "
                "or set ATLAS_PRECISION_MAC_LIBRARY")
        configure_library(path)
    out = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32)
    status = _mac(a.data_ptr(), b.data_ptr(), out.data_ptr(),
                  a.shape[0], k, b.shape[1], int(rescale))
    if status:
        raise RuntimeError(f"ordered MAC returned {status}")
    _finite(out, "ordered MAC output")
    return out


def contraction(a, b, ea, eb, k_chunk=0, low=None):
    """Fresh raw BF16 chains; ascending K, high then low, FP64 merge."""
    if k_chunk < 0 or (k_chunk and k_chunk % 32):
        raise ValueError("K chunk must be zero or a positive multiple of 32")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible contraction matrices")
    k = a.shape[1]
    padded = (k + 31) // 32 * 32
    a = F.pad(a, (0, padded - k))
    branches = [(b, eb)] if low is None else [(b, eb), low]
    branches = [(F.pad(weight, (0, 0, 0, padded - k)), ea + exponent)
                for weight, exponent in branches]
    merged = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float64)
    chunk = k_chunk or padded
    for start in range(0, padded, chunk):
        for weight, exponent in branches:
            raw = ordered_mm(a[:, start:start + chunk],
                             weight[start:start + chunk])
            merged = merged + torch.ldexp(raw.double(), exponent)
    return direct_bf16(merged)


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
