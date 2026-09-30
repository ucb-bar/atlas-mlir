# Bounded VREDSUM.BF16 observation on selected AtlasCore

This ACT-independent hand OOT check uses selected `atlas-npu` revision
`0079c0541111197741a231c002e3843fa6f545b2` and the source-linked
standalone `AtlasCore` ARC model. The inspected `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.

`Instructions.scala` and `IDecode.scala` bind `VREDSUM_BF16` to opcode `0x57`,
funct7 `0x01`, and `VPU_CSUM`. `VectorFSM.scala` reads a BF16 register pair as
64 physical rows of 16 lanes. `ColAddVec.scala` converts each BF16 input to a
widened 24-bit-significand format and adds it to a per-lane accumulator with
nearest-even rounding at each row. `VectorEngine.scala` converts the final
recoded result to 32-bit IEEE bits, takes bits 31:16, and writes those same 16
results to every row in both destination registers. The inspected software
model instead sums a logical 32-by-32 BF16 tile by its 32 columns and converts
the result to BF16. Its actual outputs were not used as a golden here.

The [typed 36-word program](../test/examples/vpu_col_sum_pair.mlir) loads both
source halves, executes `col_sum`, and stores both destination halves. Its
[independent selected-assembler transcription](../test/examples/vpu_col_sum_pair.S),
the OOT emitter, and LLVM RISC-V object `.text` agree word for word. The parser
and printer round trip, and the verifier rejects odd/out-of-range register
pairs and an unknown reduction kind.

The [test](../test/test_vpu_col_sum_reference.py) independently decodes finite
normal BF16 values as exact rationals, rounds each serial addition to normal
binary32 with nearest-even ties, then takes the upper 16 output bits. It tests
two complete 2,048-byte input panels. The first distinguishes final truncation
from BF16 rounding, binary32 steps from BF16 steps, serial loss of 30 small
addends before cancellation from exact regrouping, a negative result, and a
value located only in physical row 63. The second exercises every lane with
both source halves and alternating upper-bit results. Both selected-core runs
matched all 1,024 BF16 output cells, halted, observed 64 DMA reads and 64
writes, preserved all 2,048 input bytes and a 32-byte guard, and overwrote
the output preload. The focused suite passed 4/4 methods with the freshly
built OOT tools. After updating the source-bound census assertion, the full
source-linked CTest passed 137/137 Python methods with no skips or failures
in 125.61 seconds. Its retained summary is under
`out/qualifications/oot-vpu-col-sum-r1/full-source-linked-ctest-r2.log`.

The `DELAY 256` in the diagnostic program supplies slack for this observed
stream; it is not a qualified minimum latency. This result covers two finite
normal/positive-zero panels, one source and destination register pair, one
flat launch, and the selected standalone core. Exceptional values, arbitrary
pairs, dynamic timing, and integrated SoC behavior remain open. The hand OOT
reference still has no callable Atlas ABI, and this observation does not
complete gate D.
