# VPU BF16 multiply observation

This is a bounded, ACT-independent check of the hand-authored Atlas OOT
machine dialect on one selected RTL revision. It does not qualify all BF16
encodings, VPU overlap schedules, or the integrated Atlas SoC.

The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`.
`src/main/scala/atlas/scalar/Instructions.scala:116-119` assigns
`VMUL_BF16` opcode `0x57` and function field `0x03`; `IDecode.scala:112`
routes it to `VPU_MUL`. `vector/VectorEngineTop.scala:160-172` requires
even matrix-register pair bases for both operands and the result.
`vector/VectorEngine.scala:190-198` sends both 16-lane rows to the
multiply lane box with all lanes enabled. In
`vector/laneBoxes/MulRec.scala:81-124`, `MulRawFN` multiplies BF16
operands and `RoundRawFNToRecFN` rounds the product once at BF16
precision with rounding mode zero. The independent oracle models the
selected finite-normal, normal-result subset as an exact rational
product followed by nearest-even BF16 rounding.

The examined local `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.
`npu_model/configs/isa_definition.py:447-453` reads BF16 pairs and uses
Torch elementwise multiplication. The model is not used to calculate
expected values or to encode the stream.

`test/examples/vpu_mul_pair.mlir` is a typed 49-word program that loads
two 2,048-byte BF16 register pairs from DRAM, multiplies them into a
third pair, and stores both result halves. The independent selected
assembler transcription is `test/examples/vpu_mul_pair.S`. All 49
emitted words match that assembler and an ELF32 RISC-V object lowered
through the OOT LLVM pass; the object words are extracted for standalone
core execution, rather than calling the void LLVM function through a
host ABI. Parser/printer round-trip succeeds and invalid pair bases or
unknown operation kinds are rejected.

`test/test_vpu_mul_reference.py` decodes finite-normal BF16 inputs to
exact rationals and rounds each exact product to BF16 nearest-even. Its
directed cases cover positive and negative products, exact products,
and halfway cases rounding both down and up:

- `0x3f82 × 0x3fa0` yields `0x3fa2` (lower even significand).
- `0x3f81 × 0x3fc0` yields `0x3fc2` (upper even significand).
- Negating one or both operands checks sign handling independently.

Two deterministic 1,024-element panels place these cases in both BF16
register halves and vary the remaining finite inputs. The oracle excludes
zero, subnormal, infinity, NaN, overflow, underflow, and other exceptional
behavior. The diagnostic 33- and 64-cycle waits follow selected public
`baremetal/assembly/vpu_binary.S`; they do not establish general VPU
availability or cross-family overlap safety.

The focused selected-source-linked standalone-core test passed two panels:
all 1,024 BF16 result cells in each panel matched the independent exact
oracle, including both register halves and directed halfway products. The
four 1,024-byte DRAM input halves and an unrelated 32-byte guard were
preserved; each run observed 128 DMA reads and 64 DMA writes. The static
oracle, assembler/object, parser, and invalid-verifier checks passed 3/3
test methods. Input matrix-register preservation and in-place VMUL were
not exercised.
The direct OOT Python suite passed 55/55 tests in 157.239 seconds with
selected ARC paths configured and no skips. CTest was not rerun for this
increment; the allocated run budget allowed the focused execution and one
full suite repetition.
