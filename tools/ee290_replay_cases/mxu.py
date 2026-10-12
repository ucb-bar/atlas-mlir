"""One FP8 tile through each MXU: vload A and W, push W, overwrite matmul, BF16 pop, store both halves.

Expected results come from an exact fixed-point reference independent of the RTL (the archived
ee290-mxu-reference.h fixture, phase 0): one nonzero weight per output column, so every output is a
single exact product, and any result that would need BF16 rounding is rejected. The accumulate cases
push a BF16 or FP8 accumulator first; the reference adds it exactly, refuses any term set wider than
the inner-product tree's exact anchor window, and the FP8 pop quantizes only exactly representable values.
"""
from fractions import Fraction

from . import END, Case, add, filler, put, vload, vstore
from .vpu_ops import bf16_from_fp8, exact, fp8_from_bf16, lanes, scale_exponent, value


def fp8(integer, scale, negative=False):
    """E4M3 encoding of integer (1..8) * 2^scale, finite normal only."""
    bit = integer.bit_length() - 1
    exponent = bit + scale + 7
    if not 1 <= integer <= 8 or not 1 <= exponent <= 14: raise ValueError('outside the fixture FP8 domain')
    return (0x80 if negative else 0) | exponent << 3 | ((integer << (3 - bit)) - 8)


def decode(bits):
    """E4M3 as (signed numerator, exponent); zero is (0, 0)."""
    exponent, fraction = (bits >> 3) & 15, bits & 7
    if not exponent:
        if fraction: raise ValueError('subnormal outside the fixture reference')
        return 0, 0
    if exponent == 15: raise ValueError('extended exponent outside the fixture reference')
    return (-(8 + fraction) if bits & 0x80 else 8 + fraction), exponent - 10


def exact_bf16(fixed):
    """BF16 of a fixed-point value with 16 fractional bits; rounding is refused."""
    if not fixed: return 0
    magnitude = abs(fixed); bit = magnitude.bit_length() - 1
    exponent = bit - 16 + 127
    if not 1 <= exponent <= 254: raise ValueError('result outside normal BF16')
    if bit > 7:
        if magnitude & ((1 << (bit - 7)) - 1): raise ValueError('result requires BF16 rounding')
        significand = magnitude >> (bit - 7)
    else: significand = magnitude << (7 - bit)
    return (0x8000 if fixed < 0 else 0) | exponent << 7 | (significand - 128)


def fixture(phase=0):
    activation = bytes(fp8(1 + (3 * r + 5 * k + phase % 8) % 8, r // 8 - 1) for r in range(32) for k in range(32))
    weights = bytearray(1024)
    for column in range(32):
        weights[32 * column + (5 * column + phase % 32) % 32] = fp8(1, column // 8 - 2, (column + phase) % 2 != 0)
    return activation, bytes(weights)


def reference(activation, weights, accumulator=None):
    """C[r,j] = acc[r,j] + sum_k A[r,k] W[j,k] as two 32x16 BF16 panels (columns 0..15, then 16..31).

    accumulator[32 r + j] is the BF16 code of the pushed accumulator (None: overwrite). Nonzero terms must
    lie within 2^16 of each other (the anchor tree keeps 17 bits below its largest term) and an accumulated
    total must be nonzero.
    """
    result = bytearray(2048)
    for row in range(32):
        for column in range(32):
            terms = []
            for k in range(32):
                (a, ea), (w, ew) = decode(activation[32 * row + k]), decode(weights[32 * column + k])
                if a and w:
                    if not 0 <= ea + ew + 16 <= 20: raise ValueError('product outside the fixed-point domain')
                    terms.append(a * w << (ea + ew + 16))
            if accumulator is not None:
                fixed = value(accumulator[32 * row + column]) * (1 << 16)
                if fixed.denominator != 1 or abs(fixed) >= 1 << 28: raise ValueError('accumulator outside the fixed-point domain')
                terms.append(int(fixed))
            spans = [abs(t).bit_length() for t in terms if t]
            if spans and max(spans) - min(spans) > 16: raise ValueError('terms exceed the exact anchor window')
            if accumulator is not None and not sum(terms): raise ValueError('signed-zero accumulation outside the reference')
            offset = (column // 16) * 1024 + row * 32 + (column % 16) * 2
            result[offset:offset + 2] = exact_bf16(sum(terms)).to_bytes(2, 'little')
    return bytes(result)


def accumulator_codes(panels):
    """Accumulator rows of a BF16 pair in panel layout (lanes 0..15 from the first register, 16..31 from the second)."""
    return [code for row in range(32) for half in (0, 1) for code in lanes(panels[1024 * half + 32 * row:1024 * half + 32 * row + 32])]


def dequantize(row):
    """Accumulator row loaded by an FP8 acc push: E4M3 lanes to BF16 (e4m3ToBf16)."""
    return b''.join(bf16_from_fp8(b, 0).to_bytes(2, 'little') for b in row)


def quantize(row, scale):
    """FP8 pop of one accumulator row: BF16ScaleToE4M3 multiplies by 2^(scale - 127); rounding, saturation and underflow are refused."""
    return bytes(fp8_from_bf16(c, scale_exponent(scale)) for c in lanes(row))


ACTIVATION, WEIGHTS = fixture()
IMAGE = ACTIVATION + filler(1024, 1024) + WEIGHTS + filler(5120, 3072)


def tile_ops(unit, slot):
    return (add(6, 0), vload(0, 6), add(6, 512), vload(2, 6),
            ('mxu_push', f'kind = "weight_fp8", unit = {unit} : i32, src = 2 : i32, slot = {slot} : i32'),
            ('mxu_matmul', f'unit = {unit} : i32, src = 0 : i32, weight_slot = {slot} : i32, acc_slot = {slot} : i32, accumulate = false'),
            ('mxu_pop', f'format = "bf16", unit = {unit} : i32, dst = 4 : i32, slot = {slot} : i32, scale_reg = 0 : i32'),
            add(8, 1024), vstore(4, 8), add(8, 1280), vstore(5, 8))


def result(vm, dram): put(vm, 128, reference(ACTIVATION, WEIGHTS))


# Accumulate: the pushed accumulator is a multiple of each column's single product, so every total is exact
# (BF16 pop: C0 (1 + f) has at most 8 significant bits; FP8 pop: 1 + f is a power of two and C0 has 4 bits).
OVERWRITE = [int.from_bytes(reference(ACTIVATION, WEIGHTS)[o:o + 2], 'little') for o in range(0, 2048, 2)]
PRODUCTS = accumulator_codes(bytes(b for c in OVERWRITE for b in c.to_bytes(2, 'little')))
BF16_FACTORS = (3, -3, Fraction(1, 2), Fraction(-3, 4), 5, -2, 1, Fraction(7, 4))
FP8_FACTORS = (1, -2, Fraction(-1, 2))
SCALE = 126  # FP8 pop halves


def scaled_bf16(i, factors): return exact(value(PRODUCTS[i]) * factors[(i // 32 + i % 32) % len(factors)])


def panels(codes):  # accumulator rows -> BF16 pair in panel layout
    return b''.join(codes[32 * r + 16 * half + j].to_bytes(2, 'little') for half in (0, 1) for r in range(32) for j in range(16))


ACC_BF16 = [scaled_bf16(i, BF16_FACTORS) for i in range(1024)]
ACC_FP8 = bytes(fp8_from_bf16(scaled_bf16(i, FP8_FACTORS), 0) for i in range(1024))
ACC_IMAGE = ACTIVATION + bytes([SCALE]) + filler(1023, 1025) + WEIGHTS + filler(1024, 3072)


def accumulate_ops(unit, slot, fp8):
    load = ('mxu_push', f'kind = "acc_{"fp8" if fp8 else "bf16"}", unit = {unit} : i32, src = 4 : i32, slot = {slot} : i32')
    pop = ('mxu_pop', f'format = "{"fp8" if fp8 else "bf16"}", unit = {unit} : i32, dst = 6 : i32, slot = {slot} : i32, scale_reg = {3 if fp8 else 0} : i32')
    scale = (('scalar_load', 'kind = "seld", dst = 3 : i32, base = 6 : i32, offset = 1024 : i32'),) if fp8 else ()
    return (add(6, 0), *scale, vload(0, 6), vload(2, 6, 16), vload(4, 6, 32), *(() if fp8 else (vload(5, 6, 40),)),
            ('mxu_push', f'kind = "weight_fp8", unit = {unit} : i32, src = 2 : i32, slot = {slot} : i32'), load,
            ('mxu_matmul', f'unit = {unit} : i32, src = 0 : i32, weight_slot = {slot} : i32, acc_slot = {slot} : i32, accumulate = true'),
            pop, vstore(6, 6, 64), *(() if fp8 else (vstore(7, 6, 72),)))


def accumulated_bf16(vm, dram): put(vm, 256, reference(ACTIVATION, WEIGHTS, ACC_BF16))


def accumulated_fp8(vm, dram):
    total = accumulator_codes(reference(ACTIVATION, WEIGHTS, [int.from_bytes(dequantize(bytes([b])), 'little') for b in ACC_FP8]))
    put(vm, 256, b''.join(quantize(b''.join(c.to_bytes(2, 'little') for c in total[32 * r:32 * r + 32]), SCALE) for r in range(32)))


CASES = {f'mxu{unit}_tile': Case(tile_ops(unit, unit) + END, {0: IMAGE}, result, {'vload': 2, f'mxu{unit}': 3, 'vstore': 2})
         for unit in (0, 1)}
CASES.update({f'mxu{unit}_acc_bf16': Case(accumulate_ops(unit, 1 - unit, False) + END, {0: ACC_IMAGE + panels(ACC_BF16) + filler(4096, 8192)},
                                          accumulated_bf16, {'vload': 4, f'mxu{unit}': 4, 'vstore': 2}) for unit in (0, 1)})
CASES.update({f'mxu{unit}_acc_fp8': Case(accumulate_ops(unit, 1 - unit, True) + END, {0: ACC_IMAGE + ACC_FP8 + filler(5120, 8192)},
                                         accumulated_fp8, {'vload': 3, f'mxu{unit}': 4, 'vstore': 1}) for unit in (0, 1)})
