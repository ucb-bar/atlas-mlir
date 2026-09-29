# Bounded VREDMAX.ROW.BF16 observation

This is a selected-RTL standalone-core diagnostic for row-wise BF16 maximum
in the hand-authored, ACT-independent OOT Atlas dialect. The selected
`atlas-npu` revision is `0079c0541111197741a231c002e3843fa6f545b2`;
the inspected `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.

`Instructions.scala` fixes `VREDMAX_ROW_BF16` opcode `0x57` and funct7
`0x26`; `IDecode.scala` selects `VPU_RMAX`. `RowMax.scala` reduces each
16-element half with `VectorParam.scala::compareReturnMax`, and
`VectorEngine.scala` combines the half winners and broadcasts the result to
both halves. The comparison uses a sign-dependent unsigned ordering of raw
BF16 encodings. The inspected model calls Torch row `max`; it was not used
to compute expected outputs.

The [typed 36-word stream](../test/examples/vpu_row_max_pair.mlir) loads a
32-by-32 BF16 tile into one register pair, reduces each row, and stores both
1,024-byte output halves. Its [selected-assembler transcription](../test/examples/vpu_row_max_pair.S),
OOT emitter, and RISC-V LLVM-object `.text` words agree. The registered
Atlas-to-LLVM pass emits ordered side-effecting inline assembly with a memory
clobber. Parser/printer round-trip and verifier negatives cover odd and
out-of-range pair bases plus an unknown reduction mode.

`test/test_vpu_row_max_reference.py` uses an independent integer-order
oracle on two complete 32-row panels. It places winners in both physical
halves and checks full 32-lane broadcast. Directed cases include positive
versus negative zero, positive NaN versus infinity, negative NaN versus
negative finite values, and ordinary signed finite values. Both panels
matched the selected-source-linked standalone `AtlasCore` bit for bit.
Each run halted, observed 64 DMA reads and 64 writes, preserved the entire
2,048-byte input and a 32-byte guard, and overwrote output preloads. A
first-half-only reduction produces a different expected output and is
explicitly rejected by the test.

The diagnostic `DELAY 33` and `DELAY 64` instructions follow the public VPU
example; they are not qualified minimum availability bounds. This evidence
covers the directed row values, one tile geometry and stream, and the
standalone core with ModeLIR's external TileLink driver. Other BF16
encodings, in-place use, general timing, and integrated SoC execution remain
unqualified. The mode remains outside a frozen executable software contract.
Local logs are under `out/qualifications/oot-vpu-row-max-r1/`.
