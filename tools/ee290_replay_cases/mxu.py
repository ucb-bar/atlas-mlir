"""One FP8 tile through each MXU: vload A and W, push W, overwrite matmul, BF16 pop, store both halves.

Expected results come from an exact fixed-point reference independent of the RTL (the archived
ee290-mxu-reference.h fixture, phase 0): one nonzero weight per output column, so every output is a
single exact product, and any result that would need BF16 rounding is rejected.
"""
from . import END, Case, add, filler, put, vload, vstore


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


def reference(activation, weights):
    """C[r,j] = sum_k A[r,k] W[j,k] as two 32x16 BF16 panels (columns 0..15, then 16..31)."""
    result = bytearray(2048)
    for row in range(32):
        for column in range(32):
            total = 0
            for k in range(32):
                (a, ea), (w, ew) = decode(activation[32 * row + k]), decode(weights[32 * column + k])
                if a and w:
                    if not 0 <= ea + ew + 16 <= 20: raise ValueError('product outside the fixed-point domain')
                    total += a * w << (ea + ew + 16)
            offset = (column // 16) * 1024 + row * 32 + (column % 16) * 2
            result[offset:offset + 2] = exact_bf16(total).to_bytes(2, 'little')
    return bytes(result)


ACTIVATION, WEIGHTS = fixture()
IMAGE = ACTIVATION + filler(1024, 1024) + WEIGHTS + filler(5120, 3072)


def tile_ops(unit, slot):
    return (add(6, 0), vload(0, 6), add(6, 512), vload(2, 6),
            ('mxu_push', f'kind = "weight_fp8", unit = {unit} : i32, src = 2 : i32, slot = {slot} : i32'),
            ('mxu_matmul', f'unit = {unit} : i32, src = 0 : i32, weight_slot = {slot} : i32, acc_slot = {slot} : i32, accumulate = false'),
            ('mxu_pop', f'format = "bf16", unit = {unit} : i32, dst = 4 : i32, slot = {slot} : i32, scale_reg = 0 : i32'),
            add(8, 1024), vstore(4, 8), add(8, 1280), vstore(5, 8))


def result(vm, dram): put(vm, 128, reference(ACTIVATION, WEIGHTS))


CASES = {f'mxu{unit}_tile': Case(tile_ops(unit, unit) + END, {0: IMAGE}, result, {'vload': 2, f'mxu{unit}': 3, 'vstore': 2})
         for unit in (0, 1)}
