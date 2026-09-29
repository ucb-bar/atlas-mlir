# DMA scalar-pointer capture observation

This is a bounded, ACT-independent control/effect check on one selected
Atlas standalone core. It does not establish all DMA scheduling rules,
cross-channel overlap behavior, or integrated SoC execution.

The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`.
`src/main/scala/atlas/scalar/ScalarCore.scala:489-509` drives the DMA
command's DRAM address from scalar register data when `s1_fire` launches
the command. `src/main/scala/diplomatic/memory/DMA.scala:223-229` copies
the command bits into a queue, and lines 237-245 use the queued address
for TileLink requests. The examined local `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`; its
`npu_model/configs/programs/asm/matmul.S:56-60` warns about XRF operands
being read at execution time. That comment does not establish whether
registers remain live after launch on this selected RTL. The test below
examines that narrower boundary directly.

`test/examples/dma_pointer_capture.mlir` is a typed 16-word stream. It
configures DMA, sets scalar `x1` to DRAM A (`0x90000000`), issues a 128-byte
DMA load into VMEM, immediately overwrites `x1` with DRAM B
(`0x90001000`), waits for the DMA, and stores VMEM to C (`0x90000400`).
`test/examples/dma_pointer_capture.S` is a separate transcription for the
selected assembler. All 16 words match both that assembler and an ELF32
RISC-V object lowered through the hand OOT LLVM pass. Parser/printer
round-trip passes; four illegal scalar-register or channel variants fail
verification. The object words are passed to the existing standalone-core
driver, not called through a host ABI.

The selected-source-linked standalone-core test made two runs. In the
ordinary stream, an immediate post-launch overwrite of `x1` to B left
the transfer sourced from A. In a mutation moving B into `x1` before
the launch, the transfer sourced from B. Each run checked the complete
128-byte output, both distinct 128-byte input regions, four guard
regions, halt, four DMA reads, and four DMA writes. This independently
discriminates the launch boundary for the tested 128-byte transfer; it
does not make later scalar overwrites universally safe under different
core revisions, command-queue conditions, or scheduling contexts.

The hand OOT dialect conservatively puts every machine operation on one
state token and declares read/write physical effects. It does not yet
encode a precise scalar-register lifetime or prove DMA queue capacity,
completion, or source preservation for arbitrary programs. The observed
command capture is a fact to reconcile with Merlin's selected target
contract, not permission to weaken its current conservative scheduler.
The full direct OOT suite passed 66/66 Python methods in 221.023 seconds
with the selected ARC paths configured and no skips.
