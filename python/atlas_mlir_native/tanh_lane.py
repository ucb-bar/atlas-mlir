"""Finite BF16 port of the selected Atlas Tanh/TanhLUT datapath.

Source: atlas-npu 0079c0541111197741a231c002e3843fa6f545b2,
sp26-fp-units gitlink 9a0cc09c41ab918a3548580185f39cca8d559e0d,
TanhBlock.scala, TanhLUT.scala and common.scala, wrapped by TanhRec.scala.
This attributed port preserves Q16 truncation, interpolation, saturation
and BF16 RNE. It is not mathematical tanh rounded once. Nonfinite inputs
remain outside this implementation's admitted domain.
"""

TANH_LUT_Q1_16 = (
    0, 8150, 16051, 23485, 30285, 36346, 41625, 46131,
    49912, 53038, 55593, 57660, 59320, 60643, 61694, 62524,
    63179, 63693, 64096, 64412, 64659, 64852, 65003, 65120,
    65212, 65283, 65339, 65383, 65417, 65443, 65464, 65480,
)


def vtanh_bf16_bits(code: int) -> int:
    """Evaluate one finite raw BF16 lane, preserving both signed zeros."""
    if type(code) is not int or not 0 <= code <= 0xFFFF or (code >> 7) & 255 == 255:
        raise ValueError("VTANH requires a finite BF16 encoding")
    sign = code & 0x8000
    exponent = (code >> 7) & 255
    if exponent >= 129:  # |x| >= 4: the RTL bypasses the LUT with exactly one.
        magnitude = 65536
    else:
        significand = (128 if exponent else 0) | (code & 127)
        shift = (exponent - 127 if exponent else -126) + 9
        fixed = significand << shift if shift >= 0 else significand >> -shift
        address, alpha = divmod(fixed, 8192)
        lower = TANH_LUT_Q1_16[address]
        upper = TANH_LUT_Q1_16[min(address + 1, 31)]
        magnitude = lower + ((upper - lower) * alpha >> 13)
    if not magnitude:
        return sign
    highest = magnitude.bit_length() - 1
    exponent = highest - 16 + 127
    shift = highest - 7
    if shift > 0:
        significand, remainder = divmod(magnitude, 1 << shift)
        halfway = 1 << (shift - 1)
        significand += remainder > halfway or remainder == halfway and bool(significand & 1)
    else:
        significand = magnitude << -shift
    if significand == 256:
        exponent += 1
        significand >>= 1
    return sign | exponent << 7 | (significand & 127)
