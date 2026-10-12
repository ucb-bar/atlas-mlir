"""The two-layer FP8 MLP virtual programs (test/examples/virtual_fp8_two_layer_mlp*.mlir), lowered by the compiler.

MXU0(X, W0) -> BF16 ReLU -> FP8 pack (scale code 127) -> 32-iteration LW/SW relayout loop whose addresses move every
iteration -> MXU1(H, W1) -> BF16 output. The lowering stages X, W0 and W1 at VMEM bytes 0, 2048 and 4096, the packed
tile at 0x20000, its row-major relayout at 0x20400 and the output pair at 0x40000; those windows and 1 KiB of guard
lines after the output are initialized. Each weight column has one nonzero entry (mxu.py's fixture), so every product
is exact; W1 picks an even, non-negative column of H. Expected bytes come from the same exact references the
validator checks writes against. The bias program adds B0 after MXU0 and B1 after MXU1 (staged at VMEM bytes 6144
and 8192); each bias equals the product it is added to, so every sum is exact.
"""
from pathlib import Path

from . import Case, filler, put
from .mxu import fixture, fp8, reference
from .vpu_ops import elementwise, fp8_pack

EXAMPLES = Path(__file__).resolve().parents[2] / 'test/examples'
INPUT = 0x90000000
X_LINE, W0_LINE, W1_LINE, B0_LINE, B1_LINE, PACKED, ROWS, OUT_LINE = 0, 64, 128, 192, 256, 4096, 4128, 8192
SCALE = 127

X, W0 = fixture()
W1 = bytearray(1024)
for column in range(32):
    W1[32 * column + 2 * ((5 * column + 1) % 16)] = fp8(1, column // 8 - 2, column % 2 != 0)
W1 = bytes(W1)


def rows(data): return [data[32 * r:32 * r + 32] for r in range(len(data) // 32)]
def add(lhs, rhs): return b''.join(elementwise('vadd.bf16', a, b) for a, b in zip(rows(lhs), rows(rhs)))


def layers(bias):
    """Every intermediate tile; with bias, B0 = H0 and B1 = the second product, so each add doubles exactly."""
    h0 = reference(X, W0)                               # BF16 panels: columns 0..15, then 16..31
    b0 = h0 if bias else None
    h1 = [elementwise('vrelu.bf16', row) for row in rows(add(h0, b0) if bias else h0)]
    packed = b''.join(fp8_pack(h1[2 * n], h1[2 * n + 1], SCALE) for n in range(32))
    # Row r joins the FP8 columns 0..15 (packed bytes 16r..) and 16..31 (bytes 512 + 16r..) of matrix row r.
    h = b''.join(packed[16 * r:16 * r + 16] + packed[512 + 16 * r:528 + 16 * r] for r in range(32))
    y = reference(h, W1)
    return b0, (y if bias else None), packed, h, add(y, y) if bias else y


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
    inputs = X + filler(1024, 7) + W0 + filler(1024, 8) + W1
    if bias:
        vmem.update({B0_LINE: filler(2048, 9), B1_LINE: filler(2048, 10)})
        inputs += filler(1024, 11) + b0 + b1

    def expect(vm, dram):
        put(vm, X_LINE, X); put(vm, W0_LINE, W0); put(vm, W1_LINE, W1)
        if bias: put(vm, B0_LINE, b0); put(vm, B1_LINE, b1)
        put(vm, PACKED, packed); put(vm, ROWS, h); put(vm, OUT_LINE, y)
        for i, byte in enumerate(y): dram[output + i] = byte
    commands = {'dma': 9 if bias else 5, 'vload': 8 if bias else 4, 'mxu0': 3, 'vpu': 4 if bias else 2, 'vstore': 3, 'mxu1': 3}
    source = (EXAMPLES / ('virtual_fp8_two_layer_mlp_bias.mlir' if bias else 'virtual_fp8_two_layer_mlp.mlir')).read_text()
    return Case((), vmem, expect, commands, dram={INPUT: inputs, output - 32: b'\xa5' * 32 + b'\x5a' * 2048 + b'\xa5' * 32},
                dma=(224 if bias else 96, 64), source=source, pipeline=pipeline, check=relayout_moves)


LOWER = ('--verify-atlas-virtual-stream', '--lower-atlas-virtual-to-machine')
SCHEDULED = ('--verify-atlas-virtual-stream', '--schedule-atlas-virtual', '--lower-atlas-virtual-to-machine')
CASES = {'virtual_fp8_mlp': mlp(False, LOWER), 'virtual_fp8_mlp_scheduled': mlp(False, SCHEDULED), 'virtual_fp8_mlp_bias': mlp(True, LOWER)}
