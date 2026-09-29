# Branch delay positions and backward loop observation

This is a bounded source-linked observation for the hand-authored OOT Atlas
reference. It does not certify branch timing on the integrated SoC.

The selected RTL is `atlas-npu`
`0079c0541111197741a231c002e3843fa6f545b2`.
`src/main/scala/atlas/scalar/PcControl.scala` describes and implements **one**
architecturally visible delay slot. When a branch resolves, the already
in-flight instruction at `branch_pc+1` executes and the next fetch redirects
to the target. `ScalarCore.scala` computes the target from its word-index PC
plus the signed encoded byte displacement shifted right by one. The selected
assembler doubles a label distance in words when encoding BEQ/BLT.

The separately pinned `npu_model-atlas` architectural text at
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`,
`npu_spec/04_functional_units/README.md` and
`npu_spec/06_instruction_set/README.md`, requires **two** delay slots. The
selected standalone RTL result below disagrees with that intent. No source
document was changed to erase the discrepancy.

`test/examples/branch_two_positions_loop.mlir` is a 14-word typed program.
A taken BEQ writes debug CSR `0xc10` in the first post-branch position and
would write `0xc11` in the second. Its target then enters a backward BLT loop
that increments a counter three times. The BLT is taken twice. Its first
post-branch instruction increments another counter on all three evaluations;
its second position increments a third counter only on the final, not-taken
fallthrough. The independent selected assembler, `atlas-emit`, and LLVM
RISC-V object agree on all 14 words, including BEQ displacement `+6` and
BLT displacement `-2` bytes. Typed parser/printer round-trip passes.

The selected standalone `AtlasCore` execution halted with debug CSR values
`0xc10=17`, `0xc11=0`, and loop counters `(x12,x13,x15)=(3,3,1)`.
A checked target mutation redirects BEQ to the second post-branch position;
then `0xc11=34`, proving that the second CSR write is observable when reached.
The source program and mutation preserve preloaded DRAM input and guard
regions, with no DMA reads or writes. These observations distinguish the
selected one-slot behavior from the architectural two-slot text for these
commands. They do not establish all branch hazards, illegal control flow in
a slot, interrupts, branch-delay timing under stalls, or integrated execution.

The test also exposed an OOT LLVM pass bug: it treated a negative I32
displacement as unsigned when checking whether a target stays in the inline
assembly block. The pass now sign-extends the attribute for branch and JAL
target checks. Regressions require the legal backward BLT to lower and match
the assembler, while odd positive/negative offsets, forward/backward
out-of-block targets, a redirect in a delay slot, and an unmapped CSR address
are rejected. This correction does not change the selected RTL's delay-slot
semantics or qualify a callable LLVM function ABI.
