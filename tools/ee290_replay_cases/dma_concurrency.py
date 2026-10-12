"""Hand-scheduled final streams with DMA transfers in flight on two channels, and vector work under a pending transfer.

The selected `dma=wait` rules allow one pending transfer and no VMEM access while it is pending, so these streams
skip scheduling. Each DMA tile is 1 KiB (32 beats); DMA windows sit in VMEM banks 3 and 4, vector work on the same
bank uses other lines, and DRAM outputs carry 32-byte guards on both sides. `check` expects the decoded events to
list DMA commands with `edge`/`release_edge` like the other engines.
"""
from . import END, Case, add, delay, dma, dma_config, filler, put, tile, transpose, upper, vload, vstore, wait, xlu

BANK1, BANK3, BANK4 = 8192, 3 * 8192, 4 * 8192          # first VMEM line of each bank
A, B, O1, O2 = 0x90000000, 0x90001000, 0x90002000, 0x90003000
SIZE = 1024
TILE_A = bytes((37 * i + 11 + 3 * (i // 32)) & 255 for i in range(SIZE))
TILE_B = bytes((53 * i + 79 + (i // 32)) & 255 for i in range(SIZE))
SOURCE = bytes((29 * i + 5) & 255 for i in range(SIZE))
GUARD = b'\xa5' * 32

# x2 size; x6/x7 word addresses of bank 3 / bank 4 line 0; x1/x3 DRAM tiles A/B; x8/x9 DRAM outputs O1/O2;
# x10/x11/x12 word addresses of bank 3 line 1024, bank 3 line 2048 and bank 1 line 0.
SETUP = (add(5, 0), dma_config(0, 5), add(2, SIZE), upper(6, BANK3 * 8 >> 12), upper(7, BANK4 * 8 >> 12),
         upper(1, A >> 12), upper(3, B >> 12), upper(8, O1 >> 12), upper(9, O2 >> 12),
         upper(10, (BANK3 + 1024) * 8 >> 12), upper(11, (BANK3 + 2048) * 8 >> 12), upper(12, BANK1 * 8 >> 12))


def guarded(address, data):
    return {address - 32: GUARD, address: data, address + len(data): GUARD}


def concurrent_dma(events):
    """Check: two DMA commands were live on a common edge."""
    spans = [(c['edge'], c['release_edge']) for c in events['commands'] if c['engine'] == 'dma']
    return any(a < d and c < b for i, (a, b) in enumerate(spans) for c, d in spans[i + 1:])


def under_dma(engine):
    """Check: some `engine` command issued while a DMA command was live."""
    def check(events):
        dmas = [(c['edge'], c['release_edge']) for c in events['commands'] if c['engine'] == 'dma']
        return any(a <= c['edge'] < b for c in events['commands'] if c['engine'] == engine for a, b in dmas)
    return check


# Two loads in flight on channels 0 and 1, waited in launch order, then two stores in flight.
TWO_CHANNELS = SETUP + (dma('load', 0, 6, 1), dma('load', 1, 7, 3), wait(0), wait(1),
                        dma('store', 0, 6, 8), dma('store', 1, 7, 9), wait(0), wait(1))


def two_channels(vm, dram):
    put(vm, BANK3, TILE_A)
    put(vm, BANK4, TILE_B)
    for i in range(SIZE):
        dram[O1 + i] = TILE_A[i]
        dram[O2 + i] = TILE_B[i]


# A load into bank 3 lines 0..31 stays pending while a VLOAD reads bank 3 lines 1024..1055 and a VSTORE writes
# bank 3 lines 2048..2079 during the transfer's write phase (the LSU is granted first); a transpose of the loaded
# tile is then stored to bank 1.
LOAD_UNDER_VLS = SETUP + (dma('load', 0, 6, 1), delay(36), vload(4, 10), delay(34), vstore(4, 11), delay(34), xlu(5, 4),
                          delay(66), vstore(5, 12), delay(34), wait(0))


def load_under_vls(vm, dram):
    put(vm, BANK3, TILE_A)
    put(vm, BANK3 + 2048, tile(vm, BANK3 + 1024))
    put(vm, BANK1, transpose(tile(vm, BANK3 + 1024)))


# A store from bank 3 lines 0..31 reads VMEM at once while a VLOAD reads bank 3 lines 1024..1055 (DMA reads wait for
# the LSU) and the loaded tile is stored to bank 3 lines 2048..2079.
STORE_UNDER_VLS = SETUP + (dma('store', 0, 6, 8), vload(4, 10), delay(34), vstore(4, 11), delay(34), wait(0))


def store_under_vls(vm, dram):
    put(vm, BANK3 + 2048, tile(vm, BANK3 + 1024))
    for i in range(SIZE):
        dram[O1 + i] = SOURCE[i]


CASES = {
    'dma_two_channels': Case(TWO_CHANNELS + END, {BANK3: filler(SIZE), BANK4: filler(SIZE, SIZE)}, two_channels, {'dma': 4},
                             dram={A: TILE_A, B: TILE_B, **guarded(O1, filler(SIZE, 2048)), **guarded(O2, filler(SIZE, 3072))},
                             dma=(64, 64), final=True, consumers=('final',), check=concurrent_dma),
    'dma_load_under_vls': Case(LOAD_UNDER_VLS + END, {BANK3: filler(SIZE), BANK3 + 1024: SOURCE, BANK3 + 2048: filler(SIZE, SIZE),
                                                      BANK1: filler(SIZE, 2048)},
                               load_under_vls, {'dma': 1, 'vload': 1, 'xlu': 1, 'vstore': 2}, dram={A: TILE_A},
                               dma=(32, 0), final=True, consumers=('final',), check=under_dma('vload')),
    'dma_store_under_vls': Case(STORE_UNDER_VLS + END, {BANK3: SOURCE, BANK3 + 1024: TILE_B, BANK3 + 2048: filler(SIZE)},
                                store_under_vls, {'dma': 1, 'vload': 1, 'vstore': 1}, dram=guarded(O1, filler(SIZE, 3072)),
                                dma=(0, 32), final=True, consumers=('final',), check=under_dma('vload')),
}
