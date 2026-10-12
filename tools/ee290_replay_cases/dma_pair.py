"""Smoke case for several DMA transfers in flight: a load on channel 0 and a store on channel 1, launched back to back.

The hand-scheduled stream binds footprints to a reference in which each wait follows its own launch; the compiled
variants keep both transfers in flight under the provider's queued-transfer rules.
"""
from . import END, Case, add, dma, dma_config, filler, put, upper, wait

A, O = 0x90000000, 0x90000400
PAIR = (add(5, 0), dma_config(0, 5), upper(1, 589824), upper(3, 589824), add(3, 1024, 3), add(2, 128), add(6, 0), add(7, 256),
        dma('load', 0, 6, 1), dma('store', 1, 7, 3), wait(0), wait(1))
LOADED = bytes((19 * i + 7) & 255 for i in range(128))


def moved(vm, dram):
    stored = b''.join(vm[32 + r] for r in range(4))
    put(vm, 0, LOADED)
    for i, byte in enumerate(stored): dram[O + i] = byte


def both_in_flight(events):
    commands = events['commands']
    return any(commands[i]['engine'] == commands[j]['engine'] == 'dma' for i, j in events['concurrent_commands'])


CASES = {
    'dma_pair_in_flight': Case(PAIR + END, {0: filler(8192)}, moved, {'dma': 2}, dram={A: LOADED, O - 32: b'\xa5' * 192}, dma=(4, 4),
                               final=True, consumers=('final',), check=both_in_flight),
    'dma_pair_compiled': Case(PAIR + END, {0: filler(8192)}, moved, {'dma': 2}, dram={A: LOADED, O - 32: b'\xa5' * 192}, dma=(4, 4),
                              check=both_in_flight),
}
