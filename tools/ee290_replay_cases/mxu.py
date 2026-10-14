"""MXU and virtual cases with an independent exact fixed-point reference."""
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from . import END, Case, add, dma, dma_config, filler, issue_gaps, overlapping, put, tile, transpose, upper, vload, vstore, wait, xlu
from . import op, raw_tiles
from .vpu_ops import bf16_from_fp8, exact, fp8_from_bf16, lanes, scale_exponent, value, fp8_unpack, RELU_IN, elementwise, fp8_pack
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
def exact_bf16(fixed): return exact(Fraction(fixed, 1 << 16)) if fixed else 0
def fixture(phase=0):
    activation = bytes(fp8(1 + (3 * r + 5 * k + phase % 8) % 8, r // 8 - 1) for r in range(32) for k in range(32))
    weights = bytearray(1024)
    for c in range(32): weights[32 * c + (5 * c + phase % 32) % 32] = fp8(1, c // 8 - 2, (c + phase) % 2 != 0)
    return activation, bytes(weights)

def reference(activation, weights, accumulator=None):
    """C[r,j] = acc[32*r+j] + sum_k A[r,k] W[j,k], emitted as two 32x16 BF16 panels.
    Terms must fit the anchor tree's 17-bit window; accumulated totals must be nonzero."""
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
def push(unit, src=2, slot=0): return op('mxu_push', kind='weight_fp8', unit=unit, src=src, slot=slot)
def matmul(unit, src=0, slot=0, acc=0, accumulate=False):
    return op('mxu_matmul', unit=unit, src=src, weight_slot=slot, acc_slot=acc, accumulate=accumulate)
def pop(unit, dst=4, acc=0): return op('mxu_pop', format='bf16', unit=unit, dst=dst, slot=acc, scale_reg=0)
def tile_ops(unit, slot):
    return (add(6, 0), vload(0, 6), add(6, 512), vload(2, 6), push(unit, slot=slot),
            matmul(unit, slot=slot, acc=slot), pop(unit, acc=slot),
            add(8, 1024), vstore(4, 8), add(8, 1280), vstore(5, 8))
def result(vm, dram): put(vm, 128, reference(ACTIVATION, WEIGHTS))

# Accumulate: the pushed accumulator is a multiple of each column's single product, so every total is exact
# (BF16 pop: C0 (1 + f) has at most 8 significant bits; FP8 pop: 1 + f is a power of two and C0 has 4 bits).
PRODUCTS = accumulator_codes(reference(ACTIVATION, WEIGHTS))
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
    format = 'fp8' if fp8 else 'bf16'
    load = op('mxu_push', kind=f'acc_{format}', unit=unit, src=4, slot=slot)
    pop = op('mxu_pop', format=format, unit=unit, dst=6, slot=slot, scale_reg=3 if fp8 else 0)
    scale = (op('scalar_load', kind='seld', dst=3, base=6, offset=1024),) if fp8 else ()
    return (add(6, 0), *scale, vload(0, 6), vload(2, 6, 16), vload(4, 6, 32), *(() if fp8 else (vload(5, 6, 40),)),
            push(unit, slot=slot), load, matmul(unit, slot=slot, acc=slot, accumulate=True),
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
BANK2, BANK3 = 2 * 8192, 3 * 8192  # first VMEM line of each bank
A, O = 0x90000000, 0x90002000
SIZE = 1024
TILE = bytes((37 * i + 11 + 3 * (i // 32)) & 255 for i in range(SIZE))
GUARD = b'\xa5' * 32

# x2 size, x5 DMA base 0, x9/x10 word addresses of bank 3 / bank 2 line 0, x1 DRAM tile / x3 DRAM output.
SETUP = (add(5, 0), dma_config(0, 5), add(2, SIZE), upper(9, BANK3 * 8 >> 12), upper(10, BANK2 * 8 >> 12),
         add(6, 0), vload(0, 6), add(6, 512), vload(2, 6))
def under_transfer(unit, direction):
    """The MXU tile with a transfer launched while its matmul runs: a load into bank 3 once m8 has read the old
    bank 3 tile, or a store of bank 3 once m8 (from bank 2) has been stored there."""
    if direction == 'load':
        body = (upper(1, A >> 12), vload(8, 9), push(unit), matmul(unit), dma('load', 0, 9, 1), pop(unit), wait(0), vstore(8, 10))
    else:
        body = (upper(3, O >> 12), vload(8, 10), push(unit), matmul(unit), vstore(8, 9), dma('store', 0, 9, 3), pop(unit), wait(0))
    return (*SETUP, *body, add(8, 1024), vstore(4, 8), add(8, 1280), vstore(5, 8))
OLD = filler(SIZE, 4096)
def loaded(vm, dram): result(vm, dram); put(vm, BANK2, OLD); put(vm, BANK3, TILE)
def stored(vm, dram):
    result(vm, dram); put(vm, BANK3, TILE)
    for i in range(SIZE): dram[O + i] = TILE[i]

# Under the MXU, unpack m0 to bank 2 lines 128..191, transpose m2 to line 32, and load line 64.
# Unpack uses scale register 3 from byte 1024.
def beside_vector(unit):
    return (add(6, 0), op('scalar_load', kind='seld', dst=3, base=6, offset=1024), vload(0, 6),
            add(6, 512), vload(2, 6), upper(10, BANK2 * 8 >> 12), push(unit), matmul(unit),
            op('vpu_pack', direction='fp8_to_bf16', dst=12, src=0, scale_reg=3), xlu(17, 2),
            vload(18, 10, 16), pop(unit), vstore(12, 10, 32), vstore(13, 10, 40), vstore(17, 10, 8),
            add(8, 1024), vstore(4, 8), add(8, 1280), vstore(5, 8))
def vector_result(vm, dram):
    result(vm, dram)
    rows = [vm[r] for r in range(32)]
    put(vm, BANK2 + 128, b''.join(fp8_unpack(rows[n // 2], n % 2, IMAGE[1024]) for n in range(64)))
    put(vm, BANK2 + 32, transpose(tile(vm, 64)))

# The tile on both MXUs at once: MXU1 reads its own copies (m8, m10) and pops to m12:13, stored at lines 192..255.
BOTH = (add(6, 0), vload(0, 6), vload(8, 6), add(6, 512), vload(2, 6), vload(10, 6), push(0), push(1, 10), matmul(0), matmul(1, 8),
        pop(0), pop(1, 12), add(8, 1024), vstore(4, 8), vstore(5, 8, 8), vstore(12, 8, 16), vstore(13, 8, 24))
def both_results(vm, dram): result(vm, dram); put(vm, 192, reference(ACTIVATION, WEIGHTS))
TILE_COUNTS = {'vload': 3, 'dma': 1}
OVERLAP = {}
for unit in (0, 1):
    engine = f'mxu{unit}'
    OVERLAP[f'{engine}_under_dma_load'] = Case(under_transfer(unit, 'load') + END, {0: IMAGE, BANK2: filler(SIZE), BANK3: OLD},
                                              loaded, {**TILE_COUNTS, engine: 3, 'vstore': 3}, dram={A: TILE}, dma=(32, 0),
                                              check=lambda events, e=engine: overlapping(events, 'dma', e))
    OVERLAP[f'{engine}_under_dma_store'] = Case(under_transfer(unit, 'store') + END, {0: IMAGE, BANK2: TILE, BANK3: filler(SIZE)},
                                               stored, {**TILE_COUNTS, engine: 3, 'vstore': 3},
                                               dram={O - 32: GUARD, O: filler(SIZE, 2048), O + SIZE: GUARD}, dma=(0, 32),
                                               check=lambda events, e=engine: overlapping(events, 'dma', e))
    OVERLAP[f'{engine}_beside_vector'] = Case(beside_vector(unit) + END, {0: IMAGE, BANK2: raw_tiles()},
                                            vector_result, {'vload': 3, engine: 3, 'vpu': 1, 'xlu': 1, 'vstore': 5},
                                            check=lambda events, e=engine: all(overlapping(events, e, x) for x in ('vpu', 'xlu', 'vload', 'vstore')))
OVERLAP['mxu_both_units'] = Case(BOTH + END, {0: IMAGE}, both_results, {'vload': 4, 'mxu0': 3, 'mxu1': 3, 'vstore': 4},
                               check=lambda events: overlapping(events, 'mxu0', 'mxu1'))

# Lines 0..31 activation 1, 32..63 activation 2, 64..95 weights 1, 96..127 weights 2; results from line 128.
A1, W1 = fixture(0)
A2, W2 = fixture(3)
PIPELINE_IMAGE = A1 + A2 + W1 + W2 + filler(4096)
PIPELINE_LOADS = (add(6, 0), vload(0, 6), vload(4, 6, 8), vload(8, 6, 16), vload(12, 6, 24))
def stores(*pairs):
    """Store each popped BF16 pair (two registers) to consecutive 64-line panels from line 128."""
    return tuple(vstore(dst + half, 6, 32 + 16 * k + 8 * half) for k, dst in enumerate(pairs) for half in (0, 1))
def two_tiles(unit):
    return (*PIPELINE_LOADS, push(unit, 8, 0), push(unit, 12, 1), matmul(unit, 0, 0, 0), matmul(unit, 4, 1, 1),
            pop(unit, 16, 0), pop(unit, 20, 1), *stores(16, 20))
def two_results(vm, dram):
    put(vm, 128, reference(A1, W1)); put(vm, 192, reference(A2, W2))
def k2(unit):
    return (*PIPELINE_LOADS, push(unit, 8, 0), matmul(unit, 0, 0, 0), push(unit, 12, 1), matmul(unit, 4, 1, 0, True),
            pop(unit, 16, 0), *stores(16))
def k2_result(vm, dram): put(vm, 128, reference(A2, W2, accumulator_codes(reference(A1, W1))))

def pipelined():
    cases = {}
    for unit in (0, 1):
        engine = f'mxu{unit}'
        overlap = lambda events, e=engine: bool(issue_gaps(events, e))
        for name in ('tile', 'acc_bf16', 'acc_fp8'):
            cases[f'{engine}_{name}_pipelined'] = replace(CASES[f'{engine}_{name}'], check=overlap, pairs=True)
        cases[f'{engine}_two_tiles'] = Case(two_tiles(unit) + END, {0: PIPELINE_IMAGE}, two_results, {'vload': 4, engine: 6, 'vstore': 4},
                                            check=overlap, pairs=True)
        cases[f'{engine}_k2'] = Case(k2(unit) + END, {0: PIPELINE_IMAGE}, k2_result, {'vload': 4, engine: 5, 'vstore': 2},
                                     check=overlap, pairs=True)
    cases['virtual_dma_mxu_pipelined'] = replace(DMA['virtual_dma_mxu'], consumers=('delay', 'schedule'), pairs=True,
                                                 check=lambda events: bool(issue_gaps(events, 'mxu0')))
    for name in ('virtual_fp8_mlp', 'virtual_fp8_mlp_bias'):
        base = MLP[name]
        cases[f'{name}_pipelined'] = replace(base, pairs=True, check=lambda events, c=base.check: c(events) and any(issue_gaps(events, f'mxu{u}') for u in (0, 1)))
    return cases
RELU_INPUT, RELU_OUTPUT = 0x90000000, 0x90001000
RELU_IN_LINE, RELU_OUT_LINE = 0, 8192
RELU_PROGRAM = """module {
  func.func @relu() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %x = "atlas.virtual_input_bf16"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %y = "atlas.virtual_vpu_unary"(%x) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %io2 = "atlas.virtual_output_bf16"(%io1, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io2 : !atlas.virtual_state
  }
}
"""
PIPELINE = ('--verify-atlas-virtual-stream', '--schedule-atlas-virtual', '--lower-atlas-virtual-to-machine', '--insert-atlas-dma-waits')
RELU_RESULT = b''.join(elementwise('vrelu.bf16', RELU_IN[32 * r:32 * r + 32]) for r in range(64))
def relu_output(vm, dram):
    put(vm, RELU_IN_LINE, RELU_IN); put(vm, RELU_OUT_LINE, RELU_RESULT)
    for i, byte in enumerate(RELU_RESULT): dram[RELU_OUTPUT + i] = byte
RELU = {
    'virtual_relu': Case((), {RELU_IN_LINE: filler(8192, 7), RELU_OUT_LINE: filler(8192, 9)}, relu_output, {'dma': 4, 'vload': 2, 'vpu': 1, 'vstore': 2},
                         dram={RELU_INPUT: RELU_IN, RELU_OUTPUT - 32: b'\xa5' * 32 + b'\x5a' * 2048 + b'\xa5' * 32}, dma=(64, 64),
                         source=RELU_PROGRAM, pipeline=PIPELINE),}
DMA_PROGRAM = (Path(__file__).resolve().parents[2] / 'test/examples/virtual_dma_mxu.mlir').read_text()
VIRTUAL_X, WEIGHT, OUT = 0x90000000, 0x90000400, 0x90001000
STAGE = 16384  # first VMEM line of bank 2
DMA_SCALE = 129
DMA_ACTIVATION = bytes(fp8(1 + (3 * r + 5 * k) % 8, r // 8 - 2) for r in range(32) for k in range(32))
DMA_WEIGHTS = bytearray(1024)
for c in range(32): DMA_WEIGHTS[32 * c + (5 * c) % 32] = fp8(1, c // 8 - 3, c % 2 != 0)
DMA_WEIGHTS = bytes(DMA_WEIGHTS)
PRODUCT = reference(DMA_ACTIVATION, DMA_WEIGHTS)
TOTAL = accumulator_codes(reference(DMA_ACTIVATION, DMA_WEIGHTS, accumulator_codes(PRODUCT)))
DMA_RESULT = b''.join(quantize(b''.join(c.to_bytes(2, 'little') for c in TOTAL[32 * r:32 * r + 32]), DMA_SCALE) for r in range(32))
def readout(vm, dram):
    put(vm, STAGE, DMA_RESULT)
    for i, byte in enumerate(DMA_RESULT): dram[OUT + i] = byte

# The second staging window ends holding whichever tile was loaded last, which depends on load order, so it is
# scratch; the store's window and the two unused windows are checked, as are every DMA launch and the DRAM result.
DMA_STAGE_IMAGE = filler(8192, 7)
COMMON = dict(vmem={STAGE: DMA_STAGE_IMAGE[:2048], STAGE + 64: DMA_STAGE_IMAGE[2048:4096], STAGE + 128: DMA_STAGE_IMAGE[4096:]},
              scratch=(STAGE + 64,), expect=readout, commands={'dma': 3, 'vload': 2, 'mxu0': 4, 'vstore': 1},
              dram={VIRTUAL_X: DMA_ACTIVATION, WEIGHT: DMA_WEIGHTS, OUT - 32: b'\xa5' * 32 + b'\x5a' * 1024 + b'\xa5' * 32}, dma=(64, 32),
              source=DMA_PROGRAM, pipeline=PIPELINE)
DMA = {
    'virtual_dma_mxu': Case((), consumers=('delay',), check=lambda events: overlapping(events, 'dma', 'mxu0'), **COMMON),
    'virtual_dma_mxu_stream': Case((), consumers=('schedule',), **COMMON),}
EXAMPLES = Path(__file__).resolve().parents[2] / 'test/examples'
INPUT = 0x90000000
X_LINE, W0_LINE, W1_LINE, B0_LINE, B1_LINE, PACKED, ROWS, OUT_LINE = 0, 64, 128, 192, 256, 4096, 4128, 8192
MLP_SCALE = 127
VIRTUAL_X, MLP_W0 = fixture()
MLP_W1 = bytearray(1024)
for c in range(32): MLP_W1[32 * c + 2 * ((5 * c + 1) % 16)] = fp8(1, c // 8 - 2, c % 2 != 0)
MLP_W1 = bytes(MLP_W1)
def rows(data): return [data[32 * r:32 * r + 32] for r in range(len(data) // 32)]
def add_tiles(lhs, rhs): return b''.join(elementwise('vadd.bf16', a, b) for a, b in zip(rows(lhs), rows(rhs)))
def layers(bias):
    """Every intermediate tile; with bias, B0 = H0 and B1 = the second product, so each add doubles exactly."""
    h0 = reference(VIRTUAL_X, MLP_W0)                               # BF16 panels: columns 0..15, then 16..31
    b0 = h0 if bias else None
    h1 = [elementwise('vrelu.bf16', row) for row in rows(add_tiles(h0, b0) if bias else h0)]
    packed = b''.join(fp8_pack(h1[2 * n], h1[2 * n + 1], MLP_SCALE) for n in range(32))
    # Row r joins the FP8 columns 0..15 (packed bytes 16r..) and 16..31 (bytes 512 + 16r..) of matrix row r.
    h = b''.join(packed[16 * r:16 * r + 16] + packed[512 + 16 * r:528 + 16 * r] for r in range(32))
    y = reference(h, MLP_W1)
    return b0, (y if bias else None), packed, h, add_tiles(y, y) if bias else y
def relayout_moves(events):
    """The loop body ran 32 times, each instance entering with its own x10, and moved 256 words each way."""
    loop = [e for e in events['block_entries'] if e['registers'].get('x14') == 32 and e['registers'].get('x13', 32) < 32]
    return sorted(e['registers']['x10'] for e in loop) == [0x20000 + 16 * k for k in range(32)] and \
        sum(a['kind'] == 'lw' for a in events['scalar_memory']) == 256 and sum(a['kind'] == 'sw' for a in events['scalar_memory']) == 256

def mlp(bias, pipeline):
    b0, b1, packed, h, y = layers(bias)
    output = 0x90003000 if bias else 0x90002000
    vmem = {X_LINE: filler(1024, 1), W0_LINE: filler(1024, 2), W1_LINE: filler(1024, 3), PACKED: filler(1024, 4), ROWS: filler(1024, 5),
            OUT_LINE: filler(3072, 6)}  # 32 guard lines follow the output pair
    inputs = VIRTUAL_X + filler(1024, 7) + MLP_W0 + filler(1024, 8) + MLP_W1
    if bias:
        vmem.update({B0_LINE: filler(2048, 9), B1_LINE: filler(2048, 10)})
        inputs += filler(1024, 11) + b0 + b1
    def expect(vm, dram):
        put(vm, X_LINE, VIRTUAL_X); put(vm, W0_LINE, MLP_W0); put(vm, W1_LINE, MLP_W1)
        if bias: put(vm, B0_LINE, b0); put(vm, B1_LINE, b1)
        put(vm, PACKED, packed); put(vm, ROWS, h); put(vm, OUT_LINE, y)
        for i, byte in enumerate(y): dram[output + i] = byte
    commands = {'dma': 9 if bias else 5, 'vload': 8 if bias else 4, 'mxu0': 3, 'vpu': 4 if bias else 2, 'vstore': 3, 'mxu1': 3}
    source = (EXAMPLES / ('virtual_fp8_two_layer_mlp_bias.mlir' if bias else 'virtual_fp8_two_layer_mlp.mlir')).read_text()
    return Case((), vmem, expect, commands, dram={INPUT: inputs, output - 32: b'\xa5' * 32 + b'\x5a' * 2048 + b'\xa5' * 32},
                dma=(224 if bias else 96, 64), source=source, pipeline=pipeline, check=relayout_moves)
LOWER = ('--verify-atlas-virtual-stream', '--lower-atlas-virtual-to-machine', '--insert-atlas-dma-waits')
MLP = {'virtual_fp8_mlp': mlp(False, LOWER), 'virtual_fp8_mlp_scheduled': mlp(False, PIPELINE), 'virtual_fp8_mlp_bias': mlp(True, LOWER)}
