"""Hand-scheduled final streams in which engines overlap on disjoint registers and VMEM banks.

The selected compiler rules still serialize these engines, so the streams skip scheduling and
bind each command's events to the footprint of a serialized reference instead.
"""
from . import END, Case, add, delay, overlapping, put, tile, upper, vload, vmul, vstore
from .compute import bf16_pairs, products, raw_tiles

BANK1 = 8192  # first line of VMEM bank 1; x7 = 16 << 12 words addresses it
SECOND = bytes((29 * i + 5) & 255 for i in range(1024))

# VLOAD into m1 from bank 1, then one cycle later VSTORE of m0 to bank 0.
LOAD_STORE = (add(6, 0), vload(0, 6), delay(34), upper(7, 16), add(8, 256), add(9, 512),
              vload(1, 7), vstore(0, 8), delay(40), vstore(1, 9), delay(40))
# VMUL on m0..m5, then one cycle later VLOAD into m6 from bank 1.
VPU_LOAD = tuple(op for bank, words in enumerate([0, 256, 512, 768]) for op in (add(6, words), vload(bank, 6), delay(34))) + \
           (upper(7, 16), add(8, 1024), add(9, 1280), add(10, 1536), vmul(4, 0, 2), vload(6, 7), delay(70),
            vstore(4, 8), delay(34), vstore(5, 9), delay(34), vstore(6, 10), delay(40))


def load_store(vm, dram): put(vm, 32, tile(vm, 0)); put(vm, 64, tile(vm, BANK1))
def vpu_load(vm, dram): products(vm, dram); put(vm, 192, tile(vm, BANK1))


CASES = {
    'overlap_vls': Case(LOAD_STORE + END, {0: raw_tiles(), BANK1: SECOND}, load_store, {'vload': 2, 'vstore': 2},
                        final=True, consumers=('final',), check=lambda events: overlapping(events, 'vload', 'vstore')),
    'overlap_vpu': Case(VPU_LOAD + END, {0: bf16_pairs(), BANK1: SECOND}, vpu_load, {'vload': 5, 'vpu': 1, 'vstore': 3},
                        final=True, consumers=('final',), check=lambda events: overlapping(events, 'vpu', 'vload')),
}
