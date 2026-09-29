# VPU BF16 pairwise maximum observation

This is an ACT-independent hand Atlas OOT reference check for one selected
machine instruction mode. It does not qualify the entire VPU, general timing,
or the integrated Atlas SoC.

The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`.
`src/main/scala/atlas/scalar/Instructions.scala:123` gives `VMAX_BF16`
opcode `0x57` and function field `0x06`; `IDecode.scala:114` selects
`VPU_PAIRMAX`. `vector/VectorEngineTop.scala:160-172` requires even pair
bases for both inputs and the output. `vector/VectorEngine.scala:302-312`
routes the two input vectors to `PairWiseMax`.
`vector/laneBoxes/PairwiseMax.scala:29-45` applies the comparator per
enabled lane and registers the output. The comparator in
`vector/laneBoxes/VectorParam.scala:105-110` maps a raw 16-bit value to
`~bits` for a negative sign, or `bits XOR 0x8000` otherwise; unsigned
comparison chooses the greater key, with ties choosing the second operand.
This is a raw-encoding rule. Its ordering for NaNs and signed zero must not
be replaced by a generic floating-point maximum operation.

The examined local `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.
`npu_model/configs/isa_definition.py:479-485` uses `torch.maximum` on
BF16 pairs. That is not used as the oracle or encoder here. In particular,
Torch's floating-point NaN behavior is not assumed equivalent to the
selected RTL's raw-bit ordering.

`test/examples/vpu_max_pair.mlir` is a typed 49-word stream that loads
two 2,048-byte BF16 register pairs, computes pairwise maximum into a third
pair, and stores both result halves. The separate selected-assembler
transcription is `test/examples/vpu_max_pair.S`. All 49 words emitted by
the OOT dialect match the assembler and the ELF32 RISC-V object lowered
through the OOT LLVM pass. Parser/printer round-trip succeeds; odd or
out-of-range pair bases and an unknown operation kind are rejected.
The object words are extracted for the diagnostic standalone-core driver;
the void LLVM function is not invoked through a host ABI.

`test/test_vpu_max_reference.py` independently calculates the bit-order
key and expected output for two deterministic 1,024-element panels. It
checks both operand orders for signed zero, positive and negative NaN bit
patterns, infinities, ordinary finite values, and sampled raw encodings.
This is a bounded test of the selected encoding rule, not a proof of all
65,536 × 65,536 input pairs. The 33- and 64-cycle waits in the diagnostic
program follow selected public `baremetal/assembly/vpu_binary.S`; they are
not general completion bounds or overlap qualification.

The focused selected-source-linked standalone-core test passed both panels:
all 1,024 raw BF16 output encodings per panel matched the separate bit-order
calculation, including both register halves and the directed signed-zero and
NaN cases. The four 1,024-byte DRAM input halves and an unrelated 32-byte
guard were preserved; each run observed 128 DMA reads and 64 DMA writes.
The static oracle, selected-assembler/LLVM-object, parser, and invalid
verifier checks passed 3/3 methods; standalone-core execution is a fourth
method. Input matrix-register preservation and in-place maximum were not
exercised.
The full direct OOT suite passed 63/63 Python methods in 119.077 seconds,
with selected ARC paths configured and no skips. This is standalone-core
diagnostic evidence; it is not an integrated SoC or general temporal result.
