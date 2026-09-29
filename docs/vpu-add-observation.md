# VPU BF16 addition observation

This is a bounded, ACT-independent numerical and emission check of the
hand-authored Atlas OOT machine dialect. It does not qualify every BF16
encoding, VPU resource schedule, or the integrated Atlas SoC.

The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`.
`src/main/scala/atlas/scalar/Instructions.scala:117` assigns `VADD_BF16`
opcode `0x57` and function field zero. `IDecode.scala:110` selects `VPU_ADD`.
`vector/VectorEngineTop.scala:73-85,160-172` maps the VR source fields to
two physical BF16 register pairs and requires even pair bases; it also
requires an even destination pair base. `VectorFSM.scala:117-140,195-215`
uses both read ports and traverses 64 rows for this binary pair operation.
`vector/VectorEngine.scala:178-188` sends two 16-lane rows to `AddSubSumVec`.

`vector/laneBoxes/AddSubSumVec.scala:30-50,121-138` widens each BF16 input
to a 32-bit floating format, adds with rounding mode zero, converts the
recoded result to a 32-bit floating word, and writes its upper 16 bits.
The final step is a bit chop; it is not BF16 round-to-nearest-even. For the
tested exact-FP32 sum `1 + 3/512`, the selected standalone core produced
`0x3f80`; BF16 nearest-even would produce `0x3f81`.

The examined local `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.
`npu_model/configs/isa_definition.py:114-121,420-426` reads two BF16 pairs,
uses Torch addition, and writes a BF16 pair. That implementation was not
used to calculate the expected bits. Its numerical agreement with the
selected RTL outside the checked domain remains unresolved.

`test/examples/vpu_add_pair.mlir` is a hand-authored typed 49-word stream.
It loads A and B as two 1,024-byte halves each from DRAM, performs one
`atlas.vpu_binary` add on matrix-register pairs 0/1 and 2/3, and stores
both result halves from pair 4/5. The independent selected-assembler
transcription is `test/examples/vpu_add_pair.S`; all emitted words are
checked against that assembler and an ELF32 RISC-V object lowered through
the OOT LLVM pass. The object words are extracted for standalone-core
execution; the void LLVM function is not called through a host ABI.

`test/test_vpu_add_reference.py` decodes finite-normal BF16 operands to exact
rationals and sums them. It asserts that every selected sum is exactly
representable in FP32 before chopping to BF16. Its 12 directed operand
pairs include signs, partial cancellation, both output halves, and the
rounding discriminator above. Two deterministic 1,024-element panels vary
those pairs across the full pair layout. Signed zero, subnormals, NaNs,
infinities, FP32-inexact sums, overflow, and underflow are excluded from
this bounded oracle. The selected-core results matched all 1,024 output
cells per panel, with both 2,048-byte input panels and an unrelated guard
preserved. Each run observed 128 DMA reads and 64 DMA writes.

The verifier rejects odd/out-of-range BF16 pair bases and an unknown VPU
kind. The diagnostic waits of 33 and 64 cycles follow the selected public
`baremetal/assembly/vpu_binary.S`; these successful executions do not
establish general availability or cross-family overlap rules. Input matrix
register preservation and same-register VPU hazards are not checked here.

The selected-assembler transcription and LLVM object each matched all
49 emitted words. Parser/printer round-trip, four invalid verifier cases,
and the exact-rational arithmetic checks passed. The focused standalone-core
test passed both full panels. The direct OOT Python suite passed 43/43 tests
in 165.247 seconds with selected ARC paths configured and no skips.
CTest passed 1/1 outer test in 100.88 seconds; its inner Python suite passed
43/43 in 100.608 seconds with no skips.
