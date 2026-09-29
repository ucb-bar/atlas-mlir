# VPU BF16 subtraction observation

This is a bounded, ACT-independent check of the hand-authored Atlas OOT
machine dialect on one selected RTL revision. It does not qualify all BF16
encodings, VPU overlap schedules, or the integrated Atlas SoC.

The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`.
`src/main/scala/atlas/scalar/Instructions.scala:118` assigns
`VSUB_BF16` opcode `0x57` and function field `0x02`; `IDecode.scala:111`
routes it to `VPU_SUB`. `vector/VectorEngineTop.scala:160-172` requires
even matrix-register pair bases. `vector/VectorEngine.scala:178-188`
selects the add/sub lane box with subtraction enabled. In
`vector/laneBoxes/AddSubSumVec.scala:30-50,121-138`, the lanes widen BF16
operands to the FP32-width format, subtract, convert to FP32, and retain
bits `[31:16]` for the BF16 output. The independent oracle covers the
finite-normal subset where each exact rational difference is nonzero and
exactly representable as a normal FP32 number; its final BF16 step is a
bit chop, not nearest-even rounding.

The examined local `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.
`npu_model/configs/isa_definition.py:438-444` reads BF16 pairs and uses
Torch elementwise subtraction followed by a BF16 cast. The model is not
used to calculate expected values or to encode the stream. Its rounding
behavior may differ from the selected RTL and remains a source discrepancy.

`test/examples/vpu_sub_pair.mlir` is a typed 49-word program that loads
two 2,048-byte BF16 register pairs from DRAM, subtracts them into a third
pair, and stores both result halves. The independent selected-assembler
transcription is `test/examples/vpu_sub_pair.S`. All 49 emitted words
match that assembler and an ELF32 RISC-V object lowered through the OOT
LLVM pass. The test extracts the object words for standalone-core
execution; it does not call the void LLVM function through a host ABI.
Parser/printer round-trip succeeds and invalid pair bases or unknown
operation kinds are rejected.

`test/test_vpu_sub_reference.py` decodes finite-normal BF16 inputs to
exact rationals, subtracts them, checks exact FP32 representability, and
chops the result to BF16. Directed cases check operand order and sign.
For example, `1 - (-3/512)` produces `0x3f80` on the tested RTL, whereas
BF16 nearest-even rounding of the exact result would produce `0x3f81`.
Two deterministic 1,024-element panels place directed cases in both BF16
register halves. Zero, subnormal and exceptional operands/results,
FP32-inexact differences, overflow, underflow, and signed-zero behavior
are outside this oracle. The diagnostic 33- and 64-cycle waits follow
selected public `baremetal/assembly/vpu_binary.S`; they do not establish
general VPU availability or cross-family overlap safety.

The focused selected-source-linked standalone-core test passed two panels:
all 1,024 BF16 result cells in each panel matched the independent exact
oracle, including both register halves and the directed rounding case.
The four 1,024-byte DRAM input halves and an unrelated 32-byte guard were
preserved; each run observed 128 DMA reads and 64 DMA writes. The static
oracle, assembler/object, parser, and invalid-verifier checks passed 3/3
test methods; the standalone-core execution is a fourth test method.
Input matrix-register preservation and in-place subtraction were not
exercised.
