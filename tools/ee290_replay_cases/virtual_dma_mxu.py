"""#15's virtual_dma_mxu program: two FP8 DMA loads, a weight push, an overwrite and an accumulating matmul, an FP8
readout (scale 129) and an FP8 DMA store, verified, scheduled, allocated and lowered untimed by the compiler, then
timed from the facts.

Its virtual schedule pushes the weights while the activation load is still pending; delay insertion keeps that order
(the stream scheduler may wait first, so that consumer runs without the overlap check). The lowering stages both
tiles in VMEM bank 2 (weights at line 16384, activations at 16448) and stores the readout from line 16384. The inputs
follow mxu.py's fixture with smaller exponents, so the readout 4 * (C + C) of every single-product output is an exact
E4M3 value.
"""
from pathlib import Path

from . import Case, filler, overlapping, put
from .mxu import accumulator_codes, fp8, quantize, reference

PROGRAM = (Path(__file__).resolve().parents[2] / 'test/examples/virtual_dma_mxu.mlir').read_text()
PIPELINE = ('--verify-atlas-virtual-stream', '--schedule-atlas-virtual', '--lower-atlas-virtual-to-machine')
X, WEIGHT, OUT = 0x90000000, 0x90000400, 0x90001000
STAGE = 16384  # first VMEM line of bank 2
SCALE = 129

ACTIVATION = bytes(fp8(1 + (3 * r + 5 * k) % 8, r // 8 - 2) for r in range(32) for k in range(32))
WEIGHTS = bytearray(1024)
for column in range(32):
    WEIGHTS[32 * column + (5 * column) % 32] = fp8(1, column // 8 - 3, column % 2 != 0)
WEIGHTS = bytes(WEIGHTS)
PRODUCT = reference(ACTIVATION, WEIGHTS)
TOTAL = accumulator_codes(reference(ACTIVATION, WEIGHTS, accumulator_codes(PRODUCT)))
RESULT = b''.join(quantize(b''.join(c.to_bytes(2, 'little') for c in TOTAL[32 * r:32 * r + 32]), SCALE) for r in range(32))


def readout(vm, dram):
    put(vm, STAGE, RESULT)
    put(vm, STAGE + 64, ACTIVATION)
    for i, byte in enumerate(RESULT): dram[OUT + i] = byte


COMMON = dict(vmem={STAGE: filler(8192, 7)}, expect=readout, commands={'dma': 3, 'vload': 2, 'mxu0': 4, 'vstore': 1},
              dram={X: ACTIVATION, WEIGHT: WEIGHTS, OUT - 32: b'\xa5' * 32 + b'\x5a' * 1024 + b'\xa5' * 32}, dma=(64, 32),
              source=PROGRAM, pipeline=PIPELINE)
CASES = {
    'virtual_dma_mxu': Case((), consumers=('delay',), check=lambda events: overlapping(events, 'dma', 'mxu0'), **COMMON),
    'virtual_dma_mxu_stream': Case((), consumers=('schedule',), **COMMON),
}
