# VPU BF16 row-minimum observation

This is a bounded ACT-independent check of one already parameterized
operation in the hand-authored Atlas OOT machine dialect. Its execution
evidence is limited to the selected-source-linked standalone `AtlasCore`.

The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`.
`src/main/scala/atlas/scalar/Instructions.scala:130-133` assigns
`VREDMIN_ROW_BF16` opcode `0x57` and function field `0x24`.
`IDecode.scala:123` routes it to the VPU row-minimum operation.
`vector/laneBoxes/RowMin.scala:27-52` compares adjacent pairs over each
16-lane half, and `vector/VectorEngine.scala:280-290` compares the two
half minima and broadcasts the selected code to both result halves.
`vector/laneBoxes/VectorParam.scala:98-103` transforms BF16 bits to an
ordered key, chooses the lesser key, and chooses its second argument on
an equal-key tie. This agrees with numeric minimum for the tested
finite-normal, distinct-winner domain. It does not establish IEEE NaN
propagation or signed-zero behavior.

The examined local `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.
`npu_model/configs/isa_definition.py:511-522` reads a BF16 pair,
applies a row-wise Torch minimum, expands each row's value, and writes a
BF16 pair. The model is not used as the numerical oracle here.

`test/examples/vpu_row_min_pair.mlir` is a typed 36-word stream using the
existing `atlas.vpu_reduce` family with `kind = "row_min"`. The existing
verifier requires even, in-range source/destination BF16 pair bases;
the test rejects two illegal pair bases and an unknown reduction kind.
All 36 emitted words match both the independently selected Atlas assembler
from `test/examples/vpu_row_min_pair.S` and an ELF32 RISC-V object lowered
through the registered Atlas-to-LLVM pass. Parser/printer round-trip passes.

`test/test_vpu_row_min_reference.py` decodes finite-normal BF16 values to
exact rationals and computes the minimum of each 32-lane logical row.
Two deterministic 32-row panels place unique negative minima in both
physical register halves, vary their position and magnitude by row, and
check the expected 2,048-byte pair-broadcast representation. A first-half
only reduction differs from the full-row reference on directed rows.
The independent oracle, emission, and verifier checks passed **3/3** static
test methods. Two selected-core panels then checked all 1,024 BF16 result
cells each across both register halves. The full-row expected result differs
from a first-half-only reduction on directed rows; the observed output
matched the full-row oracle. Each run preserved the 2,048-byte DRAM input
and an unrelated 32-byte guard, and observed 64 DMA reads and 64 DMA writes.
This finite execution does not establish general temporal validity or an
integrated SoC launch path. Exceptional values, signed zeros, and equal
minima are outside the bounded comparison.
The direct OOT Python suite passed 51/51 tests in 111.389 seconds with
selected ARC paths configured and no skips.
CTest passed 1/1 outer test in 102.66 seconds; its inner Python suite passed
51/51 in 102.415 seconds with no skips.
