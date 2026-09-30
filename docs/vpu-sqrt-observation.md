# Selected-source VSQRT observation

This ACT-independent hand OOT diagnostic targets `atlas-npu`
`0079c0541111197741a231c002e3843fa6f545b2`, its `sp26-fp-units`
gitlink `9a0cc09c41ab918a3548580185f39cca8d559e0d`, and the inspected
`npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.
The local `out/qualifications/oot-vpu-sqrt-all-codes-r1/receipt.json`
records SHA-256 for the selected `Sqrt.scala` (`195d8057…`), `SqrtLUT.scala`
(`490cf82e…`), `LUTParams.scala` (`684aac05…`), BF16 type (`21182c13…`),
model ISA file (`67a06de0…`), executable core, state JSON, assembler, typed
program words, and diagnostic. The local arithmetic dependency mirror has
broken Git metadata; checked bytes and the selected Gitlink are recorded,
but the mirror's own checkout revision was not independently established.

`Instructions.scala` and `IDecode.scala` assign `VSQRT` opcode `0x57`,
funct7 `0x4d`, and `VPU_SQRT`. `VectorEngine.scala` sends 16-lane physical
rows to `Sqrt.scala`. Its selected `SqrtLUT` samples
`sqrt(1 + fraction/128)` into Q1.16, rounding the table entries upward at
ties. For even encoded exponents, it multiplies by a separately rounded
Q1.16 `sqrt(2)` and truncates the product. `LUTParams.scala` chops seven
fraction bits and shifts the input exponent. The selected lane box ignores
the input sign. Zero, subnormal, and NaN encodings map to positive zero;
infinities map to positive infinity. Thus negative finite inputs yield the
same positive result as their positive counterparts. The inspected software
model calls `torch.sqrt` on the BF16 pair and was not used as the oracle.

The [36-word typed program](../test/examples/vpu_sqrt_pair.mlir) loads both
source halves, computes `m4/m5`, and stores both halves. Its [selected
assembler transcription](../test/examples/vpu_sqrt_pair.S), OOT emitter,
and LLVM RISC-V object `.text` agree word for word. The pass emits ordered
side-effecting inline assembly with a memory clobber. Parser/printer
round-trip and negative pair-base/mode verifier checks pass.

The [unit reference](../test/test_vpu_sqrt_reference.py) derives the Q1.16
table with integer square roots and rational half-up rounding; host floating
point and `npu_model` do not produce its expected bits. Two permuted
1,024-cell panels cover exact powers, fractions, both input signs, zeros,
subnormals, infinities, NaNs, and the normal exponent extremes. Both selected
standalone-core runs matched all 2,048 output bytes, halted after 64 DMA
reads and 64 writes, and preserved input and guard memory. Focused testing
passed 4/4 methods with no skips.

The separate [all-code diagnostic](../test/diagnostics/characterize_vpu_sqrt.py)
checked every 16-bit BF16 input encoding over 64 panels on one persistent
selected standalone core with one reset. It checked each output cell, both
physical register halves, input preservation, guard, ECALL termination, and
DMA drain after every launch. The source-derived checker found **0 mismatches
in 65,536 encodings** in 11.70 seconds of model driving. The range covers
input encodings, not register pairs, operation overlap, or timing schedules.
The full source-linked CTest passed **150/150 Python methods**, with no skips
or failures, in 95.01 seconds. The source-bound census checker passed at
55/99 bounded modes, 0 software-admitted modes, and 99 remaining blockers.
The full log is under `out/qualifications/oot-vpu-sqrt-r1/`.

The diagnostic `DELAY 33` and `DELAY 64` instructions provide observed slack;
they are not minimum latency rules. This is a transcription of selected
source behavior checked against one source-linked ARC model, not an
independent mathematical `sqrt` contract, a physical RTL simulation,
integrated SoC test, callable LLVM ABI, or D/F/M/N certificate. The mode
remains software-unadmitted and blocked on those obligations.
