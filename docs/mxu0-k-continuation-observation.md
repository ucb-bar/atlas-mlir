# MXU0 reset and two-K-tile continuation observation

This is a bounded standalone-core observation for the hand-authored OOT
reference. It does not qualify arbitrary scheduling, the full Atlas SoC, or
all MXU0 numerical inputs.

The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`.
`src/main/scala/atlas/scalar/Instructions.scala:110-112` distinguishes
`VMATMUL.MXU0` from `VMATMUL.ACC.MXU0` by the function field, and
`IDecode.scala:99-100` maps those to separate commands.
`src/main/scala/atlas/mxu/sa/SystolicArraySequencer.scala:435-441` sets the
compute beat's `accumulate` flag only for `MatmulAcc` and supplies the selected
accumulator contents as `psum`. `SystolicArray.scala:66-69` feeds zero for
reset and that `psum` for continuation. The selected public
`baremetal/assembly/smolvla_matmul_k_chain_mxu0.S:3-4,54-55,80-81` uses the same
reset-then-continuation sequence and diagnostic 95-cycle waits.

The separately pinned local `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`. Its
`npu_model/configs/isa_definition.py:94-107,996-1033` describes reset and
continuation, but computes a FP16 matrix product before BF16 conversion.
The test does not use that function as its numerical oracle. It uses the
existing independent exact-rational test oracle in `test/test_mxu_numerics.py`:
each FP8 product is added in K order to the BF16 accumulator, then rounded to
BF16 round-to-nearest-even. It starts the second 32-element K tile from the
first tile's stored BF16 result. This matches the selected numerical policy
for the finite values exercised here; it is not a general equivalence proof.

`test/examples/mxu0_k2.mlir` loads two activation tiles and two already
N-by-K-oriented weight tiles, then issues reset, continuation, and a BF16
pair pop. The selected public assembly transposes K-by-N weights before push;
this fixture instead declares the transposed orientation at its input
boundary. `test/examples/mxu0_k2.S` is an independent transcription for the
selected assembler. All 57 words match the OOT emitter and the LLVM-lowered
RISC-V object's extracted words. The parser/printer round-trips both distinct
matmul operations.

`test/test_mxu_k2_reference.py` checks two complete 32-by-32 output tiles:

- Dense all-one FP8 tiles: continuation produces BF16 64 (`0x4280`) in all
  1,024 cells; changing only the second instruction to reset produces 32
  (`0x4200`) in all cells.
- A fixed-seed mixed finite FP8 case: every 32-by-32 cell is checked against
  the exact ordered reference. Two isolated cells, one in each BF16 output
  register half, distinguish continuation from reset. The first is `0x3f7f`
  after a positive half-ULP tie and a negative half-ULP contribution in the
  second K tile; rounding only once after both would produce `0x3f80`.

Four selected-source-linked standalone `AtlasCore` ARC executions passed:
dense and mixed continuation, plus dense and mixed reset mutations. Each
reported 128 DMA reads and 64 writes. Both BF16 output register halves,
all four 1,024-byte input tiles, and an output guard were checked. The
mutated reset program matched the independent second-tile-only reference
and differed from the continuation reference.

The tested FP8 encodings are zero and finite normals drawn from a small
declared set. Subnormals, NaNs, saturation/overflow, all possible accumulator
values, and arbitrary initial accumulator state remain unqualified. The
program uses conservative delays copied from a selected public diagnostic;
no timing bound or asynchronous-lifetime proof follows from this run.
