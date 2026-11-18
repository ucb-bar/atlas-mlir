"""Finite VCUBE arithmetic from the selected Atlas VectorParam datapath.

Attributed port: atlas-npu 0079c0541111197741a231c002e3843fa6f545b2,
src/main/scala/atlas/vector/laneBoxes/VectorParam.scala::squareCube and
SquareCubeVec.scala. The 24-bit significand product is normalized and
chopped, with input subnormals and result underflow flushed to signed zero.
This is not three rounded BF16 multiplications. Nonfinite input remains
outside the admitted semantics; a finite input may produce infinity.
"""


def vcube_bf16_bits(code: int) -> int:
    """Evaluate one finite raw BF16 lane using integer datapath arithmetic."""
    if type(code) is not int or not 0 <= code <= 0xFFFF or (code >> 7) & 255 == 255:
        raise ValueError("VCUBE requires a finite BF16 encoding")
    sign = code & 0x8000
    exponent, fraction = (code >> 7) & 255, code & 127
    if not exponent:
        return sign
    product = (128 | fraction) ** 3
    highest = product.bit_length() - 1
    adjusted = 3 * (exponent - 127) + highest - 21 + 127
    if adjusted <= 0:
        return sign
    if adjusted >= 255:
        return sign | 0x7F80
    return sign | adjusted << 7 | ((product >> (highest - 7)) & 127)
