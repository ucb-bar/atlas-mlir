# DMA pointer capture and wait/reuse observation

This is bounded evidence for the selected standalone `AtlasCore`, not a
full-domain DMA timing contract or integrated SoC qualification. The selected
RTL is `atlas-npu` `0079c0541111197741a231c002e3843fa6f545b2`;
the inspected model is `npu_model-atlas`
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.

In the selected RTL, `ScalarCore.scala:215-217` reads scalar operands in
stage 1. `ScalarCore.scala:489-509` asserts the DMA command on `s1_fire`
and forms its DRAM address from the current register value.
`AtlasCore.scala:172-191` forwards the command and combines the DMA's
`channelBusy` with a one-cycle `dmaLaunched` latch for `DMA.WAIT`.
`DMA.scala:223-229` copies all command bits into its queue; lines 237-245
make later TileLink requests using the queued address. The inspected
`npu_model/configs/programs/asm/matmul.S:56-60` warns that scalar DMA
operands are read at execution time. For this RTL, the tested execution
point is the scalar launch, not the later memory request. We do not infer a
universal latency or a safe scheduling rule for other revisions or queues.

The new 25-word
[`dma_pointer_lifetime.mlir`](../test/examples/dma_pointer_lifetime.mlir)
and independently transcribed
[`dma_pointer_lifetime.S`](../test/examples/dma_pointer_lifetime.S)
perform two 128-byte loads into the same VMEM location and two stores to
disjoint DRAM outputs. The first load reads address A, then the program
immediately replaces scalar `x1` with B before `DMA.WAIT 0`. The first
output must contain A. After a store and `DMA.WAIT 1`, the program reuses
VMEM for a second load from B and a second store. A mutation placing B in
`x1` before the first launch must change the first output to B. Both variants
must leave their inputs and six DRAM guards intact.

The test checks all 25 words against the selected assembler and the
`MLIR -> LLVM dialect -> LLVM IR -> ELF32 RISC-V object` path. It checks a
parser/printer round trip and rejects illegal DMA channels, scalar register
indices, and signed immediates. The resulting object words, not a callable
LLVM function, are loaded into the selected standalone core. The two
executions each require eight DMA read beats and eight write beats, both
outputs, both input regions, and six guards to match. Per-cycle traces of
the core's DMA-busy signals and scalar marker registers require that a
post-wait marker appears only after the corresponding busy signal clears.
The test also requires `x1=B` while the first channel remains busy, so a
later read of the scalar register would be distinguishable.

These observations support a launch-time address snapshot and waited reuse
for this stream. The trace uses an internal busy signal and cannot by itself
prove every externally visible completion condition, arbitration behavior,
or the safety of omitting waits. The hand dialect conservatively threads a
state token through DMA commands and waits; it does not yet encode a
precise general temporal lifetime. `DMA_WAIT_ANY` now has bounded semantic
and standalone-core evidence in the 99-mode ledger, with its full-domain and
integrated-execution blockers retained. No DMA mode is software admitted.
