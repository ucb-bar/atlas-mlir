"""Exact VPU references and cases; rounding, saturation and underflow are refused."""
from dataclasses import replace
from fractions import Fraction

from . import END, Case, add, filler, issue_gaps, put, tile, vload, vstore

def lanes(row): return [int.from_bytes(row[i:i + 2], 'little') for i in range(0, len(row), 2)]
def pack(codes): return b''.join(c.to_bytes(2, 'little') for c in codes)

def value(code):
    """Finite BF16 as a Fraction; zeros are 0; subnormal, infinite and NaN codes are refused."""
    exponent, fraction = (code >> 7) & 255, code & 127
    if not exponent and not fraction: return Fraction(0)
    if not 1 <= exponent <= 254: raise ValueError('BF16 outside the finite-normal domain')
    magnitude = Fraction(128 + fraction) * Fraction(2) ** (exponent - 134)
    return -magnitude if code & 0x8000 else magnitude

def exact(number, bits=8):
    """Normal BF16 code of a nonzero Fraction with at most `bits` significant bits; rounding is refused."""
    if not number: raise ValueError('signed-zero result outside the exact domain')
    magnitude = abs(number); exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < Fraction(2) ** exponent: exponent -= 1
    significand = magnitude / Fraction(2) ** (exponent - bits + 1)
    if significand.denominator != 1: raise ValueError('result requires rounding')
    if not -126 <= exponent <= 127: raise ValueError('result outside normal BF16')
    return (0x8000 if number < 0 else 0) | (exponent + 127) << 7 | ((int(significand) << (8 - bits)) - 128)

def order(code): return (~code & 0xffff) if code & 0x8000 else code ^ 0x8000  # VectorParam.compareReturnMax key
def vmax(a, b): return a if order(a) > order(b) else b
def vmin(a, b): return a if order(a) < order(b) else b
def relu(code): return 0 if code & 0x8000 else code

def exp_special(code):
    """Exp early results: ±0 -> 1, -inf -> 0, +inf or exponent field above maxXExp (0x85) -> +inf."""
    if not code & 0x7fff: return 0x3f80
    if code == 0xff80: return 0
    if code == 0x7f80 or not code & 0x8000 and 0x86 <= (code >> 7) & 255 <= 254: return 0x7f80
    raise ValueError('exp input outside its exact special-value domain')

def exp2_integer(code):
    """ExpLane base-2 path on integers in [-100, 88] (exact Q9.12, ExLUT[0] = 1); larger positive integers overflow."""
    if not code & 0x7fff or code in (0x7f80, 0xff80): return exp_special(code)
    number = value(code)
    if number.denominator != 1 or not -100 <= number <= 255: raise ValueError('exp2 input outside the integer domain')
    return 0x7f80 if number >= 89 else (127 + int(number)) << 7

def total(codes):
    """Exact sum, refused unless every partial sum in any order fits FP32 and the result fits BF16."""
    values = [value(c) for c in codes]
    if not any(values):
        if any(codes): raise ValueError('signed zeros outside the exact sum domain')
        return 0
    unit = min(Fraction(1, v.denominator) for v in values if v) if any(v.denominator > 1 for v in values) else Fraction(1)
    if sum(abs(v) for v in values) >= (1 << 24) * unit: raise ValueError('partial sums may round in FP32')
    return exact(sum(values))

def scale_exponent(scale): return max(-128, min(127, scale - 127))

def fp8_from_bf16(code, shift):
    """BF16 -> E4M3 multiplying by 2^shift (FP8Pack passes -shift); inexact, saturating or underflowing values are refused."""
    if not code & 0x7fff: return 0
    exponent, fraction = (code >> 7) & 255, code & 127
    if not 1 <= exponent <= 254: raise ValueError('FP8 conversion outside the finite-normal domain')
    if fraction & 15: raise ValueError('FP8 conversion requires rounding')
    adjusted = exponent - 127 + shift
    if not -6 <= adjusted <= 8 or adjusted == 8 and fraction >> 4 == 7: raise ValueError('FP8 conversion saturates or underflows')
    return (code >> 8 & 0x80) | (adjusted + 7) << 3 | fraction >> 4

def bf16_from_fp8(byte, shift):
    """E4M3 -> BF16 multiplying by 2^shift (FP8Unpack and MXU dequantization); zero, subnormal and NaN give signed zero."""
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if not exponent or exponent == 15 and mantissa == 7: return (byte & 0x80) << 8
    biased = exponent - 7 + shift + 127
    if not 1 <= biased <= 254: raise ValueError('FP8 unpack saturates or underflows')
    return (byte & 0x80) << 8 | biased << 7 | mantissa << 4

# Row references by mnemonic: elementwise rows map 16 lanes; reductions see both source rows.
ELEMENTWISE = {'vrelu.bf16': relu, 'vmov': lambda c: c, 'vexp.bf16': exp_special, 'vexp2.bf16': exp2_integer,
               'vsquare.bf16': lambda c: exact(value(c) ** 2), 'vcube.bf16': lambda c: exact(value(c) ** 3),
               'vadd.bf16': lambda a, b: exact(value(a) + value(b)), 'vsub.bf16': lambda a, b: exact(value(a) - value(b)),
               'vmaximum.bf16': vmax, 'vminimum.bf16': vmin}

def elementwise(mnemonic, *rows):
    if mnemonic not in ELEMENTWISE: raise ValueError('no exact reference for ' + mnemonic)
    return pack(ELEMENTWISE[mnemonic](*codes) for codes in zip(*map(lanes, rows)))

def row_reduce(mnemonic, low, high):
    codes = lanes(low) + lanes(high)
    result = {'vredsum.row.bf16': total, 'vredmax.row.bf16': lambda c: _fold(vmax, c), 'vredmin.row.bf16': lambda c: _fold(vmin, c)}[mnemonic](codes)
    return pack([result] * 16)

def column_reduce(mnemonic, rows):
    """All 64 physical rows of the source pair reduce lane by lane; every output row is the result."""
    if len(rows) != 64: raise ValueError('column reduction needs the 64 physical source rows')
    columns = list(zip(*map(lanes, rows)))
    fold = {'vredsum.bf16': total, 'vredmax.bf16': lambda c: _fold(vmax, c), 'vredmin.bf16': lambda c: _fold(vmin, c)}[mnemonic]
    return pack(fold(column) for column in columns)

def _fold(function, codes):
    result = codes[0]
    for code in codes[1:]: result = function(result, code)
    return result

def vli(mode, immediate, n):
    """VectorLoadImm row for write n (row index n % 32)."""
    first = n % 32 == 0
    codes = {'all': [immediate] * 16, 'row': [immediate if first else 0] * 16, 'col': [immediate] + [0] * 15,
             'one': [immediate if first else 0] + [0] * 15}[mode]
    return pack(codes)

def fp8_pack(first, second, scale):
    """FP8Pack joins two consecutive BF16 source rows into one 32-byte FP8 row."""
    shift = -scale_exponent(scale)
    return bytes(fp8_from_bf16(c, shift) for c in lanes(first) + lanes(second))

def fp8_unpack(row, half, scale):
    return pack(bf16_from_fp8(b, scale_exponent(scale)) for b in row[16 * half:16 * half + 16])

# Cases. Inputs live at line 0 onward (tile-aligned), results from line 512; nothing else is initialized.

def bf16_tile(code_of): return pack(code_of(i) for i in range(1024))

# BF16 pair whose sign, zero, NaN, infinity and subnormal lanes exercise relu bit for bit.
SPECIAL = (0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc0, 0xffc1, 0x0001, 0x8001, 0x4280, 0xc2b0, 0x0080, 0x8080)
RELU_IN = bf16_tile(lambda i: SPECIAL[(i * 7 + i // 16) % len(SPECIAL)] if i % 5 else (0x3c00 + 37 * i) ^ (0x8000 * (i % 3 == 0)))

def unary(mnemonic, source, destination):
    def expect(vm, dram): put(vm, destination, b''.join(elementwise(mnemonic, vm[source + r]) for r in range(64)))
    return expect

def chain(*expects):
    def expect(vm, dram):
        for function in expects: function(vm, dram)
    return expect

RELU = (add(6, 0), vload(0, 6), vload(1, 6, 8), ('vpu_unary', 'kind = "relu", dst = 2 : i32, src = 0 : i32'),
        vstore(2, 6, 128), vstore(3, 6, 128 + 8))

# Binary: odd (A) and even (B) multiples of 1/8 so A ± B is odd, never zero, and exact in BF16.
A = bf16_tile(lambda i: exact(Fraction(2 * ((7 * i) % 61) - 61, 8)))
B = bf16_tile(lambda i: exact(Fraction(2 * ((5 * i + 3) % 31) - 30, 8)) if (5 * i + 3) % 31 != 15 else 0x3e00)
BINARY_KINDS = ('add', 'sub', 'max', 'min')
BINARY = (add(6, 0), *(vload(m, 6, 8 * m) for m in range(4)),
          *(('vpu_binary', f'kind = "{k}", dst = {4 + 2 * n} : i32, lhs = 0 : i32, rhs = 2 : i32') for n, k in enumerate(BINARY_KINDS)),
          *(vstore(4 + m, 6, 128 + 8 * m) for m in range(8)))

def binary(vm, dram):
    for n, kind in enumerate(BINARY_KINDS):
        mnemonic = {'add': 'vadd.bf16', 'sub': 'vsub.bf16', 'max': 'vmaximum.bf16', 'min': 'vminimum.bf16'}[kind]
        put(vm, 512 + 64 * n, b''.join(elementwise(mnemonic, vm[r], vm[64 + r]) for r in range(64)))

# Unary transcendental and polynomial lanes on their exact domains.
EXP_CODES = (0x0000, 0x8000, 0x7f80, 0xff80, 0x4300, 0x4380, 0x4f00, 0x7f00)
EXP_IN = bf16_tile(lambda i: EXP_CODES[(3 * i + i // 16) % len(EXP_CODES)])
EXP2_IN = bf16_tile(lambda i: exact(Fraction((11 * i) % 189 - 100)) if (11 * i) % 189 != 100 else 0x8000)
SQUARE_IN = bf16_tile(lambda i: exact(Fraction((i % 13) - 6 or 5, 1 << (i % 5))))
UNARY_OPS = (('exp', 0, 6), ('exp2', 2, 8), ('square', 4, 10), ('cube', 4, 12), ('mov', 2, 14))
UNARY = (add(6, 0), *(vload(m, 6, 8 * m) for m in range(6)),
         *(('vpu_unary', f'kind = "{k}", dst = {d} : i32, src = {s} : i32') for k, s, d in UNARY_OPS),
         *(vstore(m, 6, 128 + 8 * (m - 6)) for m in range(6, 16)))
UNARY_EXPECT = chain(*(unary({'mov': 'vmov'}.get(k, f'v{k}.bf16'), 32 * s, 512 + 32 * (d - 6)) for k, s, d in UNARY_OPS))

# Reductions over one pair of small multiples of 1/4: row, column and total magnitudes stay far below 2^24 units.
REDUCE_IN = bf16_tile(lambda i: exact(Fraction((i * 37) % 9 - 4, 4)) if (i * 37) % 9 != 4 else (0x3e80 if i % 2 else 0xbf00))
REDUCE_OPS = (('row_sum', 4), ('row_max', 6), ('row_min', 8), ('col_sum', 10), ('col_max', 12))
REDUCE = (add(6, 0), vload(0, 6), vload(1, 6, 8),
          *(('vpu_reduce', f'kind = "{k}", dst = {d} : i32, src = 0 : i32') for k, d in REDUCE_OPS),
          *(vstore(m, 6, 128 + 8 * (m - 4)) for m in range(4, 14)))

def reduce(vm, dram):
    rows = [vm[r] for r in range(64)]
    for n, (kind, _) in enumerate(REDUCE_OPS):
        mnemonic = {'row_sum': 'vredsum.row.bf16', 'row_max': 'vredmax.row.bf16', 'row_min': 'vredmin.row.bf16',
                    'col_sum': 'vredsum.bf16', 'col_max': 'vredmax.bf16'}[kind]
        if kind.startswith('row'): result = [row_reduce(mnemonic, rows[r], rows[32 + r]) for r in range(32)] * 2
        else: result = [column_reduce(mnemonic, rows)] * 64
        put(vm, 512 + 64 * n, b''.join(result))

# Immediate loads: all and row fill a pair, col and one a single register; every written row is checked.
VLI_OPS = (('all', 0, 0x3f80), ('row', 2, 0xc0a0), ('col', 4, 0x4049), ('one', 5, 0xbe00))
VLI = (add(6, 0), *(('vli', f'mode = "{m}", dst = {d} : i32, immediate = {i} : i32') for m, d, i in VLI_OPS),
       *(vstore(m, 6, 128 + 8 * m) for m in range(6)))

def immediates(vm, dram):
    for mode, dst, immediate in VLI_OPS:
        put(vm, 512 + 32 * dst, b''.join(vli(mode, immediate, n) for n in range(64 if mode in ('all', 'row') else 32)))

# FP8 pack then unpack with E8M0 scale 128 (pack halves, unpack doubles), loaded by SELD from line 0 byte 0.
SCALE = 128
PACK_IN = bf16_tile(lambda i: 0 if i % 17 == 0 else exact(Fraction((-1) ** (i % 3) * (8 + (5 * i) % 8), 8) * Fraction(2) ** ((i // 7) % 12 - 5)))
PACK = (add(6, 0), ('scalar_load', 'kind = "seld", dst = 3 : i32, base = 6 : i32, offset = 0 : i32'),
        add(8, 256), vload(0, 8), vload(1, 8, 8),
        ('vpu_pack', 'direction = "bf16_to_fp8", dst = 4 : i32, src = 0 : i32, scale_reg = 3 : i32'),
        ('vpu_pack', 'direction = "fp8_to_bf16", dst = 6 : i32, src = 4 : i32, scale_reg = 3 : i32'),
        vstore(4, 6, 128), vstore(6, 6, 128 + 8), vstore(7, 6, 128 + 16))

def packed(vm, dram):
    rows = [vm[32 + r] for r in range(64)]
    fp8 = [fp8_pack(rows[2 * n], rows[2 * n + 1], SCALE) for n in range(32)]
    put(vm, 512, b''.join(fp8))
    put(vm, 544, b''.join(fp8_unpack(fp8[n // 2], n % 2, SCALE) for n in range(64)))

CASES = {
    'vpu_relu': Case(RELU + END, {0: RELU_IN, 512: filler(6144)}, unary('vrelu.bf16', 0, 512), {'vload': 2, 'vpu': 1, 'vstore': 2}),
    'vpu_binary': Case(BINARY + END, {0: A + B, 512: filler(8192)}, binary, {'vload': 4, 'vpu': 4, 'vstore': 8}),
    'vpu_unary': Case(UNARY + END, {0: EXP_IN + EXP2_IN + SQUARE_IN, 512: filler(10240)}, UNARY_EXPECT, {'vload': 6, 'vpu': 5, 'vstore': 10}),
    'vpu_reduce': Case(REDUCE + END, {0: REDUCE_IN, 512: filler(10240)}, reduce, {'vload': 2, 'vpu': 5, 'vstore': 10}),
    'vpu_vli': Case(VLI + END, {512: filler(8192)}, immediates, {'vpu': 4, 'vstore': 6}),
    'vpu_pack': Case(PACK + END, {0: bytes([SCALE]) + filler(1023, 1), 32: PACK_IN, 512: filler(5120)}, packed,
                     {'vload': 2, 'vpu': 2, 'vstore': 3}),}

def vpu(kind, dst, src): return ('vpu_unary', f'kind = "{kind}", dst = {dst} : i32, src = {src} : i32')

SLOT_LOADS = (add(6, 0), *(vload(m, 6, 8 * m) for m in range(4)), vload(6, 6, 32), vload(7, 6, 40))
SLOT_PAIR = (vpu('mov', 4, 0), vpu('exp', 8, 4), vpu('relu', 0, 2))
SLOT_STORES = tuple(vstore(m, 6, 128 + 8 * k) for k, m in enumerate((4, 5, 8, 9, 0, 1)))
SLOT_INPUTS = {0: EXP_IN + RELU_IN + filler(2048), 512: filler(8192)}

def copied(source, destination):
    return lambda vm, dram: put(vm, destination, tile(vm, source, 64))

PAIR_EXPECT = chain(copied(0, 512), unary('vexp.bf16', 0, 576), unary('vrelu.bf16', 64, 640))

# add(m0, m2) -> m4, then relu(m4) -> m8 (RAW), mov(m6) -> m0 and vli.all -> m2 (WAR on the add's sources).
IMMEDIATE = 0x3fc0
SLOT_CHAIN = (add(6, 0), *(vload(m, 6, 8 * k) for k, m in enumerate((0, 1, 2, 3, 6, 7))),
         ('vpu_binary', 'kind = "add", dst = 4 : i32, lhs = 0 : i32, rhs = 2 : i32'), vpu('relu', 8, 4), vpu('mov', 0, 6),
         ('vli', f'mode = "all", dst = 2 : i32, immediate = {IMMEDIATE} : i32'),
         *(vstore(m, 6, 128 + 8 * k) for k, m in enumerate((4, 5, 8, 9, 0, 1, 2, 3))))

def chained(vm, dram):
    sums = [elementwise('vadd.bf16', vm[r], vm[64 + r]) for r in range(64)]
    put(vm, 512, b''.join(sums))
    put(vm, 576, b''.join(elementwise('vrelu.bf16', row) for row in sums))
    copied(128, 640)(vm, dram)
    put(vm, 704, b''.join(vli('all', IMMEDIATE, r) for r in range(64)))

SLOTS = {
    'slots_exp_relu': Case(SLOT_LOADS + SLOT_PAIR + SLOT_STORES + END, SLOT_INPUTS, PAIR_EXPECT, {'vload': 6, 'vpu': 3, 'vstore': 6},
                           check=lambda events: bool(issue_gaps(events, 'vpu')), pairs=True),
    'slots_chain': Case(SLOT_CHAIN + END, {0: A + B + filler(2048), 512: filler(8192)}, chained, {'vload': 6, 'vpu': 4, 'vstore': 8},
                        check=lambda events: len(issue_gaps(events, 'vpu')) >= 2, pairs=True),
    'slots_unary': replace(CASES['vpu_unary'], consumers=('delay',), check=lambda events: bool(issue_gaps(events, 'vpu')), pairs=True),
}
