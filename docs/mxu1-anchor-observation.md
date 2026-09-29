# MXU1 anchor-tree arithmetic observation

This is bounded evidence for the hand-authored OOT reference on the selected
standalone `AtlasCore`. It does not establish arithmetic equivalence across
MXU0 and MXU1 or qualify all MXU1 inputs and schedules.

The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`.
`src/main/scala/atlas/scalar/Instructions.scala:100-113` gives MXU1 its own
weight-push, matmul, continuation, and BF16-pop function fields.
`src/main/scala/atlas/mxu/ipt/InnerProductTrees.scala:58-69` uses one
`AnchorAccumulationTree` per output column, each receiving 32 activation and
weight values. `AnchorAccumulationTree.scala:48-118` multiplies E4M3 values,
chooses an exponent anchor, converts products into aligned signed integers,
reduces them as integers, then converts the sum to BF16. The default geometry
and work width come from `common/InnerProductTreeParams.scala:9-45`.
`mxu/FPUtils.scala:187-297` contains the aligned converters and uses
round-to-nearest-even on integer-to-BF16 conversion. These are distinct from
MXU0's systolic ordered product-add path.

The separately pinned local `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.
`npu_model/configs/isa_definition.py:94-107` routes both MXUs through an
FP16 matrix product followed by BF16 conversion. This was inspected but was
not used as the test oracle; it does not model MXU1's anchor alignment or
MXU0's per-product BF16 rounding.

`test/examples/mxu1_pair.mlir` is a hand-maintained typed 42-word stream for
one reset contraction. It accepts N-by-K-oriented FP8 weight rows at DRAM
`0x90000000`, M-by-K activation rows at `0x90000400`, and stores the two
BF16 column halves at `0x90000800` and `0x90001000`. Its words match
`test/examples/mxu1_pair.S` assembled by the selected independent assembler
and the LLVM-lowered RISC-V object. Parser/printer round-trip succeeds;
invalid unit `2` and a BF16 pair starting at physical register `63` are
rejected. The three MXU1 opcodes are checked individually against the
selected assembler.

`test/test_mxu1_reference.py` uses exact rational FP8 decoding and
BF16 round-to-nearest-even for a **single rounding after the 32-product sum**.
It does not call `npu_model` or the OOT emitter to obtain expected values.
The checked execution cases are:

- All-one activation and weight tiles: all 1,024 outputs are BF16 32.
- A sparse tie: `1 + (1/16)^2 + (1/16)^2` is BF16 `0x3f81` on MXU1.
  The same inputs run through the separately checked MXU0 typed stream
  produce `0x3f80` under ordered per-product BF16 rounding. An output in
  column 20 checks the second BF16 register half and signed orientation.
- A fixed-seed mixed tile (`0xA71A61`), using zero, positive and negative
  finite-normal FP8 encodings `{0x00, 0x38, 0xb8, 0x18, 0x98}`:
  all 1,024 output cells match the independent round-once reference.

Four selected-source-linked standalone-core executions passed: three MXU1
cases and the MXU0 tie comparison. Each observed 64 DMA reads and 64 writes;
both input tiles, both BF16 output register halves, and an output guard were
checked. The observed mixed-tile agreement supports this finite input set,
geometry, and schedule. The anchor shifters can discard low terms or clamp
outside this range; the test does not establish exact-sum behavior for all
FP8 values, accumulator states, exceptional values, or exponent ranges.
The delays in the diagnostic stream are not qualified general availability
bounds, and the external driver is not an Atlas launch ABI or integrated SoC
runner.
