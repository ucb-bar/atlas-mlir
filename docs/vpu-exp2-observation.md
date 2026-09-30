# Bounded VEXP2 observation on selected AtlasCore

This is an ACT-independent hand OOT diagnostic for selected `atlas-npu`
`0079c0541111197741a231c002e3843fa6f545b2` and its `sp26-fp-units`
gitlink `9a0cc09c41ab918a3548580185f39cca8d559e0d`. The inspected
`npu_model-atlas` is `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.
The local receipt under `out/qualifications/oot-vpu-exp2-r1/` records source,
model, ARC, OOT tool, and program hashes. The arithmetic mirror has broken
Git metadata; its checked source bytes are recorded separately from the
selected Gitlink.

`Instructions.scala` and `IDecode.scala` assign `VEXP2` opcode `0x57`,
funct7 `0x43`, and `VPU_EXP2`. `VectorEngine.scala` selects the base-two
branch of `ExpLane.scala`. For the tested normal integer inputs, the BF16
value becomes an exact Q9.12 integer, `Qmn.getKR` yields integer `k` and
zero residual, `ExLUT[0]` is exactly Q1.16 one, and conversion returns
the exact BF16 power of two. Signed zeros return one, positive infinity
returns infinity, and negative infinity returns zero through explicit early
results. The selected BF16 parameter guard (`maxXExp=0x85`,
`maxXSig=0x31`) returns positive infinity for input 89 even though 2^89 is
representable as BF16; input 88 remains finite. The inspected model calls
`torch.exp2` and has no corresponding source-level early guard. It was
used as discrepancy context, never as the expected-output oracle.

The [typed 36-word program](../test/examples/vpu_exp2_pair.mlir) loads two
BF16 source halves, computes `m4/m5`, and stores both halves. Its
[selected-assembler transcription](../test/examples/vpu_exp2_pair.S), OOT
emitter, and LLVM RISC-V object `.text` agree word for word. The LLVM pass
emits ordered side-effecting inline assembly with a memory clobber. The
parser/printer round-trips and verifier rejects odd/out-of-range pair bases
and an unknown mode.

The [bounded reference](../test/test_vpu_exp2_reference.py) derives exact
integer input values from BF16 bits and computes power-of-two result
exponents. It accepts 17 directed codes: positive/negative zero,
positive/negative infinity, positive/negative integers 1, 2, 4, 8,
positive 88, 89, 100, and negative 89, 100. It rejects fractional,
subnormal, NaN, and out-of-range cases rather than assigning an unverified
result. Two permuted 1,024-cell panels exercised both physical register
halves on the selected-source-linked standalone ARC core. Each run matched
all 2,048 output bytes, halted, observed 64 DMA reads and 64 writes,
preserved all 2,048 input bytes and a 32-byte guard, and overwrote the
output preloads. Focused testing passed 4/4 methods with no skips.
The full source-linked CTest passed **154/154 Python methods**, with no skips
or failures, in 98.18 seconds. The source-bound census checker passed at
56/99 bounded modes, 0 software-admitted modes, and 99 full-qualification
blockers. The build, focused, full, and census logs are retained under
`out/qualifications/oot-vpu-exp2-r1/`.

The `DELAY 33` and `DELAY 64` instructions provide diagnostic slack, not
minimum latency rules. Fractional inputs, NaN bit policy, subnormals,
underflow, other register pairs, concurrent operations, integrated SoC
behavior, and a callable LLVM ABI remain open. The mode is not admitted to
a frozen software contract and does not close D/F/M/N.
