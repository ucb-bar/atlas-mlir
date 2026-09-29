# MXU1 continuation observation

This is bounded evidence for `VMATMUL_ACC_MXU1` in the hand-authored Atlas
OOT reference. It does not admit the instruction's full numerical or temporal
domain.

The selected RTL is `atlas-npu` commit
`0079c0541111197741a231c002e3843fa6f545b2`.
`src/main/scala/atlas/mxu/ipt/InnerProductTreesSequencer.scala:468-475`
selects the previous accumulator buffer value as `psum` for `MatmulAcc` and
zero for reset. `AnchorAccumulationTree.scala` chooses an exponent anchor from
32 products and that BF16 addend, reduces aligned signed integers, then
converts the result to BF16. This supports a **per-tile** rounding boundary;
one exact reduction over all K tiles is a different numerical contract.

The separately pinned `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.
`npu_model/configs/isa_definition.py::_vmatmul` uses an FP16 matrix product,
adds a BF16 accumulator in FP16 for continuation, and converts to BF16.
That is not the selected RTL's anchor-aligned integer tree and is not the
oracle for this test. Equality on the tested values does not reconcile the
two implementations across their full domains.

`test/examples/mxu1_k2.mlir` is a 57-word typed stream with two 32-wide K
tiles, separate MXU1 weight pushes, reset then continuation into one local
accumulator slot, BF16 pop, and stores of both 16-column register halves.
Its emitted words match the selected standalone assembler transcription and
the RISC-V LLVM object. The parser/printer round trip passes; unit 2 and
accumulator slot 2 are rejected.

`test/test_mxu1_continuation_reference.py` computes each tile from exact
rational FP8 values and applies BF16 round-to-nearest-even after the first
tile and after adding the second tile to the rounded previous result. It uses
neither `npu_model` nor the OOT emitter for expected tensor values. The sparse
witness makes tile 0 exactly `1 + 1/256`, which rounds to `1`, and tile 1
`1/256`, which ties back to `1`. A single final rounding of the exact
64-product sum would produce `0x3f81`; resetting on tile 1 yields `0x3b80`.
Witnesses are placed in both BF16 register halves, with a negative second
tile at another cell. Dense all-one tiles yield 64 for continuation and 32
for reset. A fixed-seed mixed panel uses only zero and signed finite-normal
FP8 `1` and `1/16`.

Six selected standalone `AtlasCore` runs passed: three panels times
continuation and a second-tile reset mutation. Each checks all 1,024 BF16
outputs, 128 DMA reads, 64 DMA writes, four preserved input tiles and a
DRAM guard. This supports the specified finite values, geometry, fixed
schedule, and preloaded reset-entry DRAM. It does not qualify arbitrary FP8
exponents, anchor truncation, exceptional values, independent seeded
accumulators, scheduling bounds, or an integrated SoC launch ABI.
