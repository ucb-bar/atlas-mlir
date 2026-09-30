# Bounded direct JAL target and link observation

This is a diagnostic of the selected standalone `AtlasCore`, not an integrated
SoC control-flow or callable-function qualification. It is independent of ACT.

| Source role | Exact revision |
| --- | --- |
| Selected RTL | `atlas-npu` `0079c0541111197741a231c002e3843fa6f545b2` |
| Examined software model | local `npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` |
| Hand OOT starting point | `atlas-mlir` `628c7cee` |
| MLIR/LLVM source | `a47bddccec30255619bb8c37fa59700e661d4e66` (23.0.0git) |

`ScalarCore.scala` computes a direct JAL target as `s1_pc + (dec.imm >> 1)`
and a link as `s1_pc + 1`. `PcControl.scala` uses a word-index PC and one
delay slot. The inspected `npu_model/configs/isa_definition.py::JAL.exec`
instead uses a byte-address PC and pipeline-latency correction. The selected
RTL defines this test's expected behavior; the model is not its oracle.

The authored [eight-word program](../test/examples/jal_direct_target.mlir)
places JAL at word 1 with encoded signed byte displacement 8. It reaches
word 5, writes word index 2 into `x10`, executes word 2 as a delay slot
(`x12=1`), and skips words 3 and 4 (`x12=2` and `x12=4`). Word 5 computes
`x11=x10+x12=3`; a CSR write exposes 3 in `dbg0`. The changed-target
variant uses displacement 4, reaches word 3, and exposes 6 after the later
`x12=4` instruction. These values distinguish target units, link, and
delay-slot count in the tested straight-line stream.

The test compares all eight typed-emitter words with an independently
authored selected-assembler transcription and with the ELF32 RISC-V object
from `MLIR -> LLVM dialect -> LLVM IR -> object`. Parser/printer round-trip
passes. Negative checks reject an invalid destination register, nonzero JAL
base, odd or out-of-range displacement, and a target outside the single LLVM
inline-assembly block. Two selected-core runs check halt, link and marker
registers, `dbg0`, zero DMA traffic, and two unchanged DRAM guards. The
object words are loaded by ModeLIR's reset-entry driver; the LLVM function
is not called through a host ABI.

The single test does not establish arbitrary jumps, register-preservation
rules, interacting branches or stalls, a general return protocol, full
instruction-domain legality, or integrated SoC execution. The JAL row
remains unadmitted and blocked for gate D.
