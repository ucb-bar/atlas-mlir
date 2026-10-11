"""Drained basic blocks: counted loops and both arms of an if/else diamond."""
from . import END, Case, add, branch, dma, dma_config, jal, nop, put, tile, upper, vload, vstore, wait
from .compute import A, raw_tiles, source


def copy(first, to):
    return lambda vm, dram: put(vm, to, tile(vm, first))


# Three iterations over loop-invariant tiles; only the x13 counter varies.
LOOP = (add(6, 0), add(8, 256), add(13, 0), add(14, 3), vload(0, 6), vstore(0, 8), add(13, 1, 13), branch('blt', 13, 14, -3), nop())


# Each iteration launches and waits a DMA load, so every block instance restarts its epochs.
DMA_LOOP = (add(5, 0), dma_config(0, 5), add(6, 0), upper(1, 589824), add(2, 128), add(8, 256), add(13, 0), add(14, 3),
            dma('load', 0, 6, 1), wait(0), vload(0, 6), vstore(0, 8), add(13, 1, 13), branch('blt', 13, 14, -5), nop())


def loaded_copy(vm, dram): put(vm, 0, bytes(source(i) for i in range(128))); put(vm, 32, tile(vm, 0))


def diamond(condition):
    """x5 == 0 takes the branch to the else arm (tile 1); otherwise the then arm (tile 0) jumps to the join."""
    return (add(6, 0), add(7, 256), add(8, 512), add(5, condition), branch('beq', 5, 0, 5), nop(),
            vload(0, 6), jal(3), nop(), vload(0, 7), vstore(0, 8))


CASES = {
    'loop': Case(LOOP + END, {0: raw_tiles()}, copy(0, 32), {'vload': 3, 'vstore': 3}),
    'dma_loop': Case(DMA_LOOP + END, {0: raw_tiles()}, loaded_copy, {'dma': 3, 'vload': 3, 'vstore': 3},
                     dram={A: bytes(source(i) for i in range(128))}, dma=(12, 0)),
    'diamond_taken': Case(diamond(0) + END, {0: raw_tiles()}, copy(32, 64), {'vload': 1, 'vstore': 1}),
    'diamond_fall': Case(diamond(1) + END, {0: raw_tiles()}, copy(0, 64), {'vload': 1, 'vstore': 1}),
}
