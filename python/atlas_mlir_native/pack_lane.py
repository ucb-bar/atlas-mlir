"""BF16 to scaled E4M3 lane conversion of the selected Atlas VPU pack.

This is a verification reference transcribed from
atlas-npu/src/main/scala/atlas/vector/laneBoxes/FP8Pack.scala at RTL revision
0079c0541111197741a231c002e3843fa6f545b2. It does not produce
instructions or replace device execution.
"""

from __future__ import annotations

ROUNDED_PACK_POLICY = "fp8-e4m3-scaled-e8m0-rtl-rne-flush-clamp-122-128-v1"
ROUNDED_PACK_DOMAIN = "bf16-all-bits-rtl-pack-v1"


def pack_bf16_lane(bits: int, scale_code: int) -> int:
    """Return the VPU pack byte for one BF16 word and E8M0 scale code.

    The admitted scale range is the one exposed by the native tile descriptor.
    The RTL flushes BF16 zeros, subnormals and NaNs to positive zero, rounds
    normal significands to nearest even, and clamps overflow to E4M3 0x7e.
    """
    if type(bits) is not int or not 0 <= bits <= 0xffff:
        raise ValueError("pack input must be a BF16 word")
    if type(scale_code) is not int or not 122 <= scale_code <= 128:
        raise ValueError("pack E8M0 scale code is outside the selected range")

    sign = bits >> 15
    exponent = (bits >> 7) & 0xff
    mantissa = bits & 0x7f
    if exponent == 0 or (exponent == 0xff and mantissa != 0):
        return 0
    maximum = (sign << 7) | 0x7e
    if exponent == 0xff:
        return maximum

    # VectorEngine turns the E8M0 byte into a signed exponent shift. Pack
    # subtracts that shift from BF16's unbiased exponent.
    adjusted = (exponent - 127) - (scale_code - 127)
    significand = 0x80 | mantissa
    truncated = significand >> 4
    guard = (significand >> 3) & 1
    sticky = bool(significand & 7)
    rounded = truncated + int(bool(guard and (sticky or (truncated & 1))))
    if rounded == 16:
        adjusted += 1
        fraction = 0
    else:
        fraction = rounded - 8

    if adjusted > 8:
        return maximum
    if adjusted < -6:
        return 0
    if adjusted == 8 and fraction == 7:
        return maximum
    return (sign << 7) | ((adjusted + 7) << 3) | fraction
