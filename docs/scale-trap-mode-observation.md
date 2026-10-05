# Bounded scale-register and trap mode observation

The hand OOT dialect emitted SELI, SELD, ECALL, and EBREAK in four selected-core
programs: two E8M0 immediate/memory byte panels, each ending in one of the two
traps. The selected RTL revision was
`0079c0541111197741a231c002e3843fa6f545b2`.

Each program wrote a nonzero scalar word to VMEM through `SW`, then loaded its
low byte through `SELD e4, x6, 8`. The two effective VMEM addresses were 8
and 12 bytes; source words had nonzero upper bytes. `SELI` wrote a distinct
immediate to `e3`, while a third `SELI` initialized `e5` as a preservation
guard. A delay separated the scalar store, scale load, and trap so the bounded
test did not assume immediate memory completion.

| Check | Actual bounded result |
| --- | --- |
| Typed Atlas words versus selected assembler | 4/4 programs matched |
| Typed Atlas words versus unmodified LLVM RISC-V object text | 4/4 matched |
| Standalone `AtlasCore` scale state | `e3` equaled each immediate, `e4` equaled each VMEM low byte, `e5` stayed `0x55`, `e2` stayed zero |
| Termination | ECALL reason 2 and EBREAK reason 3; the instruction after each trap did not replace scalar `x9`'s prior value |

The executable test is
[`test_scale_trap_reference.py`](../test/test_scale_trap_reference.py).
It requires selected assembler, matching LLVM/MLIR tools, a selected-source
standalone ARC model and state map, ModeLIR, and the pinned RTL checkout.
The test checks the RTL checkout revision; the local prebuilt ARC artifact is
not independently re-attested to that revision by this test. The result is
bounded standalone execution evidence. Full register/code/address domains,
read/write collision timing, exception interactions, integrated EE290
execution, and software admission remain open.

The census now uses the existing `e8m0_scale_reg_0_31` domain for SELI and
SELD destinations. ECALL and EBREAK have empty parameter-domain lists because
their selected BitPats are fixed 32-bit words. The source-bound checker still
requires parameter domains for variable encodings.
