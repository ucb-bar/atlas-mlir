# Bounded DELAY frontend-stall observation

This ACT-independent hand OOT test observes one standalone selected-source
`AtlasCore`. It does not establish timing for every instruction family or
qualify an integrated SoC.

| Source role | Exact revision |
| --- | --- |
| Selected RTL | `atlas-npu` `0079c0541111197741a231c002e3843fa6f545b2` |
| Examined software model | local `npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` |
| Hand OOT starting point | `atlas-mlir` `84a41e67a633c4ddb3d65eb40d3967754d07b669` |
| MLIR/LLVM source | `a47bddccec30255619bb8c37fa59700e661d4e66` (23.0.0git) |

`ScalarCore.scala` sets `delayCounter` to `dec.imm(11,0)` when DELAY
fires, stalls while the counter is nonzero, and decrements it once per
cycle. `PcControl.scala` holds the frontend PC on stall. The inspected
`npu_model/configs/isa_definition.py::DELAY.exec` is a no-op; it is not
the oracle for selected-RTL timing.

The authored [six-word program](../test/examples/delay_timing.mlir) writes
`dbg0=7`, issues DELAY, then computes `x2=8` and writes `dbg1=8`.
Both DELAY 0 and DELAY 4 variants pass through typed MLIR emission, an
independently authored selected-assembler transcription, and
`MLIR -> LLVM dialect -> LLVM IR -> ELF32 RISC-V object` extraction. All
six physical words agree in each variant. Parser/printer round-trip passes;
the verifier rejects counts -1 and 4096 outside the declared 0..4095
unsigned field.

In selected-core cycle traces, DELAY at word 2 fired once in each run.
The successor at word 3 fired one cycle later for DELAY 0, and five
cycles later for DELAY 4. Only the latter trace held word 3 without firing
for four cycles while `delayCounter` read 4, 3, 2, 1. During the hold,
`dbg0` remained 7 and `dbg1` remained 0. Both runs completed with
`x2=8`, `dbg0=7`, and `dbg1=8`, zero DMA traffic, two preserved DRAM
guards, and a total-cycle difference of four. This independently
distinguishes the selected frontend stall from a no-op model DELAY.

The test covers only counts 0 and 4 in a short scalar stream. It does not
establish behavior at all 4096 count values, overlapping DMA or tensor
activity, branch-delay-slot interactions, clock-domain behavior, or the
full integrated SoC. DELAY remains unadmitted and blocked for gate D.
