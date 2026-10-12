"""The two-layer FP8 MLP virtual program (test/examples/virtual_fp8_two_layer_mlp.mlir), lowered by the compiler.

MXU0(X, W0) -> BF16 ReLU -> FP8 pack (scale code 127) -> 32-iteration LW/SW relayout loop whose addresses move every
iteration -> MXU1(H, W1) -> BF16 output. The lowering stages X, W0 and W1 at VMEM bytes 0, 2048 and 4096, the packed
tile at 0x20000, its row-major relayout at 0x20400 and the output pair at 0x40000; those windows and 1 KiB of guard
lines after the output are initialized. Each weight column has one nonzero entry (mxu.py's fixture), so every product
is exact; W1 picks an even, non-negative column of H. Expected bytes come from the same exact references the
validator checks writes against.
"""
from pathlib import Path

from . import Case, filler, put
from .mxu import fixture, fp8, reference
from .vpu_ops import elementwise, fp8_pack

PROGRAM = (Path(__file__).resolve().parents[2] / 'test/examples/virtual_fp8_two_layer_mlp.mlir').read_text()
INPUT, OUTPUT = 0x90000000, 0x90002000
X_LINE, W0_LINE, W1_LINE, PACKED, ROWS, OUT_LINE = 0, 64, 128, 4096, 4128, 8192
SCALE = 127

X, W0 = fixture()
W1 = bytearray(1024)
for column in range(32):
    W1[32 * column + 2 * ((5 * column + 1) % 16)] = fp8(1, column // 8 - 2, column % 2 != 0)
W1 = bytes(W1)


def rows(data): return [data[32 * r:32 * r + 32] for r in range(len(data) // 32)]


H0 = reference(X, W0)                                   # BF16 panels: columns 0..15, then 16..31
H1 = [elementwise('vrelu.bf16', row) for row in rows(H0)]
PACKED_TILE = b''.join(fp8_pack(H1[2 * n], H1[2 * n + 1], SCALE) for n in range(32))
# Row r joins the FP8 columns 0..15 (packed bytes 16r..) and 16..31 (bytes 512 + 16r..) of matrix row r.
H = b''.join(PACKED_TILE[16 * r:16 * r + 16] + PACKED_TILE[512 + 16 * r:528 + 16 * r] for r in range(32))
Y = reference(H, W1)


def mlp_output(vm, dram):
    put(vm, X_LINE, X); put(vm, W0_LINE, W0); put(vm, W1_LINE, W1)
    put(vm, PACKED, PACKED_TILE); put(vm, ROWS, H); put(vm, OUT_LINE, Y)
    for i, byte in enumerate(Y): dram[OUTPUT + i] = byte


VMEM = {X_LINE: filler(1024, 1), W0_LINE: filler(1024, 2), W1_LINE: filler(1024, 3), PACKED: filler(1024, 4), ROWS: filler(1024, 5),
        OUT_LINE: filler(3072, 6)}  # 32 guard lines follow the output pair
DRAM = {INPUT: X + filler(1024, 7) + W0 + filler(1024, 8) + W1, OUTPUT - 32: b'\xa5' * 32 + b'\x5a' * 2048 + b'\xa5' * 32}
COMMANDS = {'dma': 5, 'vload': 4, 'mxu0': 3, 'vpu': 2, 'vstore': 3, 'mxu1': 3}
PIPELINES = {'virtual_fp8_mlp': ('--verify-atlas-virtual-stream', '--lower-atlas-virtual-to-machine'),
             'virtual_fp8_mlp_scheduled': ('--verify-atlas-virtual-stream', '--schedule-atlas-virtual', '--lower-atlas-virtual-to-machine')}


def relayout_moves(events):
    """The loop body ran 32 times, each instance entering with its own x10 and binding one entry-context group."""
    loop = [e for e in events['block_entries'] if e['registers'].get('x14') == 32 and e['registers'].get('x13', 32) < 32]
    return sorted(e['registers']['x10'] for e in loop) == [0x20000 + 16 * k for k in range(32)] and \
        sum(a['kind'] == 'lw' for a in events['scalar_memory']) == 256 and sum(a['kind'] == 'sw' for a in events['scalar_memory']) == 256


CASES = {name: Case((), VMEM, mlp_output, COMMANDS, dram=DRAM, dma=(96, 64), source=PROGRAM, pipeline=pipeline, check=relayout_moves)
         for name, pipeline in PIPELINES.items()}
