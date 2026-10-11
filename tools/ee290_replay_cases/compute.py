"""Straight-line VLS, XLU, VPU and DMA compositions scheduled by the compiler."""
from . import END, Case, add, dma, dma_config, filler, put, tile, transpose, upper, vload, vmul, vstore, wait, xlu


def raw_tiles(): return bytes((37 * i + 11 + 3 * (i // 32)) & 255 for i in range(1024)) + filler(7168, 1024)


def power(i, rhs):
    exponent = i % 3 - 1 if rhs else i % 5 - 2
    negative = i % 11 == 0 if rhs else i % 7 == 0
    return (0x8000 if negative else 0) | ((127 + exponent) << 7)


def product(i):  # closed form of power(i, False) * power(i, True)
    return (0x8000 if (i % 11 == 0) != (i % 7 == 0) else 0) | ((127 + i % 5 - 2 + i % 3 - 1) << 7)


def bf16_pairs():
    return b''.join(power(i, False).to_bytes(2, 'little') for i in range(1024)) + \
           b''.join(power(i, True).to_bytes(2, 'little') for i in range(1024)) + filler(4096, 4096)


def transposed(vm, dram): put(vm, 32, transpose(tile(vm, 0)))
def products(vm, dram): put(vm, 128, b''.join(product(i).to_bytes(2, 'little') for i in range(1024)))


A, B, O = 0x90000000, 0x90001000, 0x90000400
def source(i): return (37 * i + 11) & 255


def composed(vm, dram):
    put(vm, 0, bytes(source(i) for i in range(128)))
    transposed(vm, dram)
    for i, value in enumerate(tile(vm, 32, 4)): dram[O + i] = value


TRANSPOSE = (add(6, 0), vload(0, 6), xlu(1, 0), add(8, 256), vstore(1, 8), add(0, 0))
MUL = tuple(op for bank, words in enumerate([0, 256, 512, 768]) for op in (add(6, words), vload(bank, 6))) + \
      (vmul(4, 0, 2), add(8, 1024), vstore(4, 8), add(8, 1280), vstore(5, 8))
MIXED = (add(5, 0), dma_config(0, 5), add(6, 0), upper(1, 589824), add(2, 128), dma('load', 0, 6, 1), wait(0)) + TRANSPOSE + \
        (upper(3, 589824), add(3, 1024, 3), dma('store', 1, 8, 3), wait(1))

CASES = {
    'xlu': Case(TRANSPOSE + END, {0: raw_tiles()}, transposed, {'vload': 1, 'xlu': 1, 'vstore': 1}),
    'vmul': Case(MUL + END, {0: bf16_pairs()}, products, {'vload': 4, 'vpu': 1, 'vstore': 2}),
    'dma_xlu': Case(MIXED + END, {0: raw_tiles()}, composed, {'dma': 2, 'vload': 1, 'xlu': 1, 'vstore': 1},
                    dram={A: bytes(source(i) for i in range(128)), B: bytes((53 * i + 79) & 255 for i in range(128)),
                          A + (1 << 32): b'\xe7' * 128, O - 32: b'\xa5' * 192}, dma=(4, 4)),
}
