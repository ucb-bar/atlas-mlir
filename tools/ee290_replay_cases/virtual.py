"""A virtual BF16 program: verified, scheduled, allocated and lowered untimed by the compiler, then timed from the facts.

The lowering stages the input pair at VMEM bytes 0..2047 and the result pair at the start of bank 1 (byte 262144), as its
tile contract records; those windows and their banks' first 8 KiB are initialized so every DMA and VLS access is modeled. The output DRAM tile sits between
guard bytes and the input tile must survive unchanged.
"""
from . import Case, filler, put
from .vpu_ops import RELU_IN, elementwise

INPUT, OUTPUT = 0x90000000, 0x90001000
IN_LINE, OUT_LINE = 0, 8192
PROGRAM = """module {
  func.func @relu() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %x = "atlas.virtual_input_bf16"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %y = "atlas.virtual_vpu_unary"(%x) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %io2 = "atlas.virtual_output_bf16"(%io1, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io2 : !atlas.virtual_state
  }
}
"""
PIPELINE = ('--verify-atlas-virtual-stream', '--schedule-atlas-virtual', '--lower-atlas-virtual-to-machine')
RESULT = b''.join(elementwise('vrelu.bf16', RELU_IN[32 * r:32 * r + 32]) for r in range(64))


def relu_output(vm, dram):
    put(vm, IN_LINE, RELU_IN)
    put(vm, OUT_LINE, RESULT)
    for i, byte in enumerate(RESULT): dram[OUTPUT + i] = byte


CASES = {
    'virtual_relu': Case((), {IN_LINE: filler(8192, 7), OUT_LINE: filler(8192, 9)}, relu_output, {'dma': 4, 'vload': 2, 'vpu': 1, 'vstore': 2},
                         dram={INPUT: RELU_IN, OUTPUT - 32: b'\xa5' * 32 + b'\x5a' * 2048 + b'\xa5' * 32}, dma=(64, 64),
                         source=PROGRAM, pipeline=PIPELINE),
}
