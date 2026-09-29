# VPU BF16 row-sum observation

This note records a bounded, ACT-independent reference check of the
hand-authored Atlas OOT machine dialect. It does not qualify all BF16
encodings, all VPU schedules, or the integrated Atlas SoC.

The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`.
`src/main/scala/atlas/scalar/Instructions.scala:131` assigns
`VREDSUM_ROW_BF16` opcode `0x57` and function field `0x21`;
`IDecode.scala:122` routes it to `VPU_RSUM`.
`vector/VectorEngineTop.scala:160-172` requires even BF16 source and
destination pair bases. `VectorFSM.scala:117-140,200-215,350-409`
reads the two 16-lane register halves together for each of 32 logical
rows. `VectorEngine.scala:168-171,292-300,405-430,450-474` concatenates
the two 16-lane inputs, passes all 32 lanes to `ReduSumRec`, takes the
broadcast result, and writes both output register halves.

`vector/laneBoxes/SumRedu.scala:83-163` widens each BF16 lane to FP32,
reduces adjacent pairs in a five-level binary tree, rounding each addition
in the FP32 format with rounding mode zero, then rounds the final result
to BF16 and broadcasts it. For the input row
`[2^25, 1, -2^25, 1, 2, 0, ..., 0]`, this tree yields BF16 `2`;
left-to-right FP32 addition yields BF16 `3`. The separate row
`[1, 0, ..., 0, 3/512, 0, ..., 0]` yields BF16 `0x3f81` after the
final nearest-even conversion. That differs from the selected VADD
operation's final bit chop on the same two nonzero values.

The examined local `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.
`npu_model/configs/isa_definition.py:502-508` uses Torch's row-wise sum,
casts to BF16, and expands the result across each row. This model was not
used to calculate expected bits. Its reduction order and numerical
agreement with the selected RTL were not established by this test.

`test/examples/vpu_row_sum_pair.mlir` is a typed 36-word stream: it loads
two 1,024-byte source halves, reduces one BF16 matrix-register pair,
and stores both 1,024-byte result halves. Its independent selected-assembler
transcription and the ELF32 RISC-V object from LLVM lowering are checked
word for word. The resulting object words are supplied to the selected
standalone `AtlasCore`; the void LLVM function is not called through a
host ABI.

`test/test_vpu_row_sum_reference.py` decodes the selected finite-normal
BF16 and positive-zero inputs to exact rational numbers. It implements
nearest-even rounding after every adjacent-tree addition at FP32 precision,
then one nearest-even BF16 conversion. Two deterministic 32-row panels
exercise the order witness, final-rounding ties, both source halves, both
result halves, broadcast layout, and multiple finite row patterns. The
oracle excludes subnormals, infinities, NaNs, signed-zero behavior, and
overflow/underflow. The verifier rejects odd/out-of-range pair bases and
an unknown reduction kind.

The stream uses 33- and 64-cycle diagnostic delays, at least as long as
those in selected public `baremetal/assembly/vpu_row_reduce.S`. Passing
these finite schedules would not establish general availability or safe
cross-family overlap.

The focused selected-source-linked standalone-core test passed two panels:
all 1,024 BF16 output cells in each panel matched the independent tree
oracle, including the order and final-rounding rows. Both 2,048-byte input
panels and the unrelated 32-byte DRAM guards were preserved. Each run
observed 64 DMA reads and 64 DMA writes. The selected assembler and LLVM
object each matched all 36 emitted words; parser/printer round-trip and
three invalid verifier cases passed. Source matrix-register preservation
and in-place reduction were not exercised. The direct OOT Python suite passed
47/47 tests in 140.605 seconds with selected ARC paths and no skips.
CTest passed 1/1 outer test in 124.34 seconds; its inner Python suite passed
47/47 in 124.089 seconds with no skips.
