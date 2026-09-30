# Bounded FENCE scalar progression and no-wait observation

This ACT-independent test observes a canonical FENCE on the selected
standalone `AtlasCore`. It does not qualify a general memory barrier or
integrated SoC execution.

| Source role | Exact revision |
| --- | --- |
| Selected RTL | `atlas-npu` `0079c0541111197741a231c002e3843fa6f545b2` |
| Examined software model | local `npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` |
| Hand OOT starting point | `atlas-mlir` `650ad3dbbcd6180db955f01884fd3006a40c7ff7` |
| MLIR/LLVM source | `a47bddccec30255619bb8c37fa59700e661d4e66` (23.0.0git) |

`IDecode.scala` decodes FENCE with no ALU, branch, register write, CSR,
DMA, memory, tensor, or LSU command. `ScalarCore.scala` forms its frontend
stall only from `delay_stall || dma_wait_stall`; FENCE is absent from that
condition. The examined `npu_model/configs/isa_definition.py::FENCE.exec`
also has an empty body. These source facts do not prove arbitrary memory
ordering behavior; the selected core execution below checks a narrower case.

The authored [six-word typed program](../test/examples/fence_scalar.mlir)
writes `dbg0=7`, issues canonical FENCE, then computes `x2=8` and writes
`dbg1=8`. Its independent selected-assembler transcription agrees with
all six emitted words, including canonical `0x0000000f`. The
`MLIR -> LLVM dialect -> LLVM IR -> ELF32 RISC-V object` path yields the
same words, and parser/printer round-trip succeeds. On the selected core,
FENCE at word 2 fires once; the successor at word 3 fires on the next
cycle. `delayCounter` remains zero. The trace has `dbg0=7`, `dbg1=0`
at FENCE issue and ends with `x1=7`, `x2=8`, `dbg0=7`, `dbg1=8`.
Replacing FENCE by canonical scalar NOP gives the same run-cycle count
and checked final state. Both variants halt, issue no DMA, and preserve
two unrelated DRAM guards.

The existing [scalar LW/JALR discriminator](jalr-load-delay-observation.md)
now also emits its FENCE variant from a typed `atlas.fence` replacement
and compares its 14 words with an independent selected-assembler
transcription. That selected-core run observes a pending scalar load and
its response at JALR issue; the JALR sees stale `x1=8` and reaches the
stale-target path, while DELAY 8 reaches the loaded target. FENCE does
not discharge the LLVM JALR target proof. The 14-word dynamic-JALR stream
cannot be lowered to the current LLVM inline-assembly block because the
memory-derived target is unproved, so LLVM object agreement is claimed
only for the six-word FENCE scalar program.

These are bounded negative completion and scalar-progress observations.
They do not cover DMA completion, all memory-ordering patterns, concurrent
tensor operations, noncanonical FENCE encodings, or integrated execution.
The FENCE ledger row remains unadmitted and blocked for gate D.
