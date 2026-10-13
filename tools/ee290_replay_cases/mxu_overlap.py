"""Compiler-scheduled MXU work overlapping a pending DMA transfer and the other vector engines.

Each case runs the FP8 tile of mxu.py (exact fixed-point reference) on one MXU; `check` requires that the overlap
occurred in the decoded events. Under a transfer, the MXU tile's operands and results stay on VMEM bank 0 while a
1 KiB tile moves between DRAM and bank 3, launched once the VLS work on that tile is done, so the launch falls
inside the matmul under both consumers; beside the vector engines, an FP8 unpack of the activation tile, a transpose
of the weight tile, VSTOREs and an unrelated VLOAD run on tiles the MXU reads but does not write; and the tile runs on both MXUs at once.
"""
from . import END, Case, add, dma, dma_config, filler, overlapping, put, tile, transpose, upper, vload, vstore, wait, xlu
from .compute import raw_tiles
from .mxu import ACTIVATION, IMAGE, WEIGHTS, reference
from .vpu_ops import fp8_unpack

BANK2, BANK3 = 2 * 8192, 3 * 8192  # first VMEM line of each bank
A, O = 0x90000000, 0x90002000
SIZE = 1024
TILE = bytes((37 * i + 11 + 3 * (i // 32)) & 255 for i in range(SIZE))
GUARD = b'\xa5' * 32


def push(unit, src=2): return ('mxu_push', f'kind = "weight_fp8", unit = {unit} : i32, src = {src} : i32, slot = 0 : i32')
def matmul(unit, src=0): return ('mxu_matmul', f'unit = {unit} : i32, src = {src} : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false')
def pop(unit, dst=4): return ('mxu_pop', f'format = "bf16", unit = {unit} : i32, dst = {dst} : i32, slot = 0 : i32, scale_reg = 0 : i32')


def mxu_result(vm, dram): put(vm, 128, reference(ACTIVATION, WEIGHTS))


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


def loaded(vm, dram): mxu_result(vm, dram); put(vm, BANK2, OLD); put(vm, BANK3, TILE)


def stored(vm, dram):
    mxu_result(vm, dram)
    put(vm, BANK3, TILE)
    for i in range(SIZE): dram[O + i] = TILE[i]


# Vector work issued while the MXU tile runs, on tiles it has already loaded: m12:13 <- FP8 unpack of the activation
# tile m0 (scale register 3 from byte 1024) -> bank 2 lines 128..191; m17 <- transpose of the weight tile m2 -> bank 2
# line 32; an unrelated m18 <- bank 2 line 64.
def beside_vector(unit):
    return (add(6, 0), ('scalar_load', 'kind = "seld", dst = 3 : i32, base = 6 : i32, offset = 1024 : i32'), vload(0, 6),
            add(6, 512), vload(2, 6), upper(10, BANK2 * 8 >> 12), push(unit), matmul(unit),
            ('vpu_pack', 'direction = "fp8_to_bf16", dst = 12 : i32, src = 0 : i32, scale_reg = 3 : i32'), xlu(17, 2),
            vload(18, 10, 16), pop(unit), vstore(12, 10, 32), vstore(13, 10, 40), vstore(17, 10, 8),
            add(8, 1024), vstore(4, 8), add(8, 1280), vstore(5, 8))


def vector_result(vm, dram):
    mxu_result(vm, dram)
    rows = [vm[r] for r in range(32)]
    put(vm, BANK2 + 128, b''.join(fp8_unpack(rows[n // 2], n % 2, IMAGE[1024]) for n in range(64)))
    put(vm, BANK2 + 32, transpose(tile(vm, 64)))


# The tile on both MXUs at once: MXU1 reads its own copies (m8, m10) and pops to m12:13, stored at lines 192..255.
BOTH = (add(6, 0), vload(0, 6), vload(8, 6), add(6, 512), vload(2, 6), vload(10, 6), push(0), push(1, 10), matmul(0), matmul(1, 8),
        pop(0), pop(1, 12), add(8, 1024), vstore(4, 8), vstore(5, 8, 8), vstore(12, 8, 16), vstore(13, 8, 24))


def both_results(vm, dram): mxu_result(vm, dram); put(vm, 192, reference(ACTIVATION, WEIGHTS))


TILE_COUNTS = {'vload': 3, 'dma': 1}
CASES = {}
for unit in (0, 1):
    engine = f'mxu{unit}'
    CASES[f'{engine}_under_dma_load'] = Case(under_transfer(unit, 'load') + END, {0: IMAGE, BANK2: filler(SIZE), BANK3: OLD},
                                              loaded, {**TILE_COUNTS, engine: 3, 'vstore': 3}, dram={A: TILE}, dma=(32, 0),
                                              check=lambda events, e=engine: overlapping(events, 'dma', e))
    CASES[f'{engine}_under_dma_store'] = Case(under_transfer(unit, 'store') + END, {0: IMAGE, BANK2: TILE, BANK3: filler(SIZE)},
                                               stored, {**TILE_COUNTS, engine: 3, 'vstore': 3},
                                               dram={O - 32: GUARD, O: filler(SIZE, 2048), O + SIZE: GUARD}, dma=(0, 32),
                                               check=lambda events, e=engine: overlapping(events, 'dma', e))
    CASES[f'{engine}_beside_vector'] = Case(beside_vector(unit) + END, {0: IMAGE, BANK2: raw_tiles()},
                                            vector_result, {'vload': 3, engine: 3, 'vpu': 1, 'xlu': 1, 'vstore': 5},
                                            check=lambda events, e=engine: all(overlapping(events, e, x) for x in ('vpu', 'xlu', 'vload', 'vstore')))
CASES['mxu_both_units'] = Case(BOTH + END, {0: IMAGE}, both_results, {'vload': 4, 'mxu0': 3, 'mxu1': 3, 'vstore': 4},
                               check=lambda events: overlapping(events, 'mxu0', 'mxu1'))
