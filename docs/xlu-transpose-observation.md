# XLU transpose observation

This note records a bounded, ACT-independent check of the hand-authored Atlas
OOT machine dialect. It does not qualify the complete XLU scheduling contract
or an integrated SoC execution path.

The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`.
`src/main/scala/atlas/scalar/Instructions.scala:157-158` assigns
`VTRPOSE_XLU` opcode `0x6b`, function field zero, and three six-bit VR fields.
`IDecode.scala:149` dispatches the word to XLU transpose, and
`ScalarCore.scala:524-527` supplies `vd` as the destination matrix register
and `vs1` as the source. `xlu/XLU.scala:66-69,146-188` reads a complete
32-by-32 matrix of eight-bit elements into an internal buffer, then writes
output element `(j,i)` from input `(i,j)`. The full read phase precedes the
write phase in that FSM, allowing this test to examine both different-register
and same-register source/destination choices. The RTL's active-read and
active-write reports are distinct (`XLU.scala:120-128`); a state token alone
does not establish their safe overlap with other instructions.

The examined `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.
`npu_model/configs/isa_definition.py:796-802` expresses the same 32-by-32
FP8 register transpose via a tensor view, transpose, and contiguous flatten.
It labels the instruction `EXU.VECTOR`, whereas the RTL routes it to XLU.
The test oracle is an independent byte-index permutation; the model is not
used to calculate expected output or encode instructions.

`test/examples/xlu_transpose.mlir` is a typed 29-word hand-authored program.
It moves 1,024 input bytes from DRAM `0x90000000` into matrix register 4,
transposes them into register 8, stores the result at `0x90000400`, and stores
the original register 4 at `0x90000800`. The `test/examples/xlu_transpose.S`
transcription is assembled independently with the selected Atlas assembler.
All emitted words are also compared with an ELF32 RISC-V object produced from
the dialect's LLVM lowering. This object-word comparison checks lowering and
encoding; the void LLVM function is not called as a host function.

`test/test_xlu_reference.py` checks the entire 1,024-byte output using
`out[j*32+i] = in[i*32+j]`. Directed panels include all 256 byte values
repeated four times and a nonsymmetric panel of positive and negative finite
FP8 encodings. A same-register variant changes only the XLU destination and
result store source; it checks that the original register is overwritten with
the transpose after the read phase. For the different-register cases, the
second store checks source-register preservation. All runs check the DRAM
input and an unrelated DRAM guard. Invalid matrix-register operands are
rejected before emission.

The 33- and 64-cycle diagnostic delays follow an existing selected public
assembly example (`baremetal/assembly/xlu_fanout_branch_reuse.S`). These
successful finite streams cannot establish a safe general instruction latency,
resource overlap policy, DMA completion protocol, or full Atlas launch ABI.
Those are separate target-compiler obligations.

The focused check passed three selected-source-linked standalone-core
executions: all 1,024 result bytes matched each reference panel, the input
and 32-byte guard remained unchanged, and each run observed 32 DMA reads and
64 DMA writes. The distinct-register cases also matched all 1,024 source
mirror bytes; the same-register case matched the overwritten source register
to the transpose. Static checks passed for 29/29 selected-assembler words,
29/29 LLVM object words, parser/printer round-trip, and two invalid register
fields. The full OOT Python suite passed 39/39 tests in 174.274 seconds with
the selected ARC paths configured and no skips.
CTest passed 1/1 outer test in 153.08 seconds; its inner Python suite passed
39/39 in 152.824 seconds with no skips.
