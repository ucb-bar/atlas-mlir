# Bounded JALR word-target and link observation

This is a selected-standalone-`AtlasCore` diagnostic, not a general
register-indirect control-flow contract or a callable LLVM function ABI.

| Inspected role | Exact revision |
| --- | --- |
| Selected Atlas RTL | `atlas-npu` `0079c0541111197741a231c002e3843fa6f545b2` |
| Examined model | `npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` |
| Hand OOT starting point | `atlas-mlir` `ee227b1` |
| MLIR/LLVM build | LLVM source `a47bddccec30255619bb8c37fa59700e661d4e66` (23.0.0git) |

`scalar/PcControl.scala:1-12,40-78` makes the PC an IMEM **word index**
and executes exactly one branch delay slot. `ScalarCore.scala:269-279`
forms JALR's target as `rs1_data + signed I-immediate`, with no low-bit
mask or byte-to-word shift, and writes the link as `s1_pc + 1`. The selected
assembler's `JALR(rd, rs1, off)` puts `off` directly in the signed I field.
The examined `npu_model/configs/isa_definition.py::JALR.exec` instead
computes with its byte-address PC and a `PIPELINE_LATENCY` correction.
We preserve both observations rather than use the model as the selected
RTL oracle. The inherited v2 plan cited an earlier model revision
`6c8601015a39b98e54ca1f9aeaac7536685ae43d`; this test is explicitly
about the locally pinned `5bb08624` model and selected RTL.

The eight-word typed
[`jalr_word_target.mlir`](../test/examples/jalr_word_target.mlir)
uses `ADDI x1,x0,6` followed by `JALR x10,x1,-1`. The target is the **odd**
word index 5. Index 2 is the delay slot and writes `x12=1`; indices 3 and 4
would write 2 and 3 but must be skipped. At index 5, `x11=x10+x12` must
equal 3 because the link is `1+1=2`, and a CSR write exposes 3 in `dbg0`.
If a byte-address JALR clears the target's low bit, it goes to index 4 and
produces 5. Missing or extra delay slots, or a link pointing beyond the
slot, also change the checked register/CSR state. A mutation setting `x1=5`
deliberately targets even index 4 and produces 5.

The selected assembler's independent transcription, OOT emitter, and
`MLIR -> LLVM dialect -> LLVM IR -> ELF32 RISC-V object` path agree on all
eight physical words. Parser/printer round-trip succeeds. Typed verifier
tests reject invalid scalar registers and signed immediate. LLVM-pass
tests reject an unknown base, a target outside the assembly block, a prior
redirect, and a prior asynchronous scalar load. The selected standalone
core executes both target variants, checks link and delay-slot registers,
halt, zero DMA traffic, and two preserved DRAM guards. The tested ELF object
words are loaded through the reset-entry standalone driver; the generated
LLVM function is not invoked through a host calling convention.

The LLVM pass now proves a JALR target only from a straight-line prefix
using known `LUI`/`ADDI` scalar constants, rejecting unmodeled preceding
effects and other control flow. It checks the resolved word target against
the single inline-assembly block before emitting. General runtime-computed
targets, effects of asynchronous scalar loads, external register state,
link-register preservation under an ABI, relocations, branch interaction,
and integrated SoC execution remain blocked. The 99-mode ledger now records
one bounded JALR LLVM/core route, **99/99 modes with at least one LLVM-word
route**, **30/99 modes with bounded semantic/core tests**, **0 admitted**, and
**99 full-D blockers**. The JALR row's full-domain dynamic target obligation
is still open.
