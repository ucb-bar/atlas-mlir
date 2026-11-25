# Atlas MLIR docs

Start with the [repository README](../README.md) for the build, and the
[examples](../examples/handoff/README.md) for runnable commands.

| Doc | Read it for |
| --- | --- |
| [Design](design.md) | How a model reaches Atlas, who decides what, numerics, and the optional schedules. |
| [Dialect reference](dialect-reference.md) | Virtual and machine operations, verifiers, passes, tools, and how to add a pass. |
| [Compiler pipeline](compiler-pipeline.md) | Virtual lowering, the SSA interpreter, the physical program format, atomic DMA, and the LLVM handoff. |
| [DMA across PACK](pack-dma-overlap.md) | Why explicit DMA may stay in flight through a PACK relayout loop, how each stage proves it, and the measured effect. |
| [PACK relayout loop](pack-relayout-loop.md) | The software-pipelined copy that repairs PACK's row layout, and the measured effect. |
| [Native Merlin binding](native-merlin-binding.md) | Compiling graphs through Merlin: supported graphs, CLI, outputs, precision policies, and limits. |
| [CFG DMA timing](timing-dma-cfg.md) | Generic control-flow DMA proof and explicit immutable-input assumptions. |
| [Host launch](host-launch.md) | Host C runtime, memory map, mailbox, cache hooks, reset entry, and board builds. |
| [Deployment](deployment.md) | `atlascc deploy`: the on-chip window, board DDR, staged plans, and kits. |
| [Chip runner](../python/atlas_mlir/chiprunner/README.md) | `atlascc run`: live, simulated, and resident runs of a kit. |
| [Baremetal-IDE integration](baremetal-c-integration.md) | Generated C/H, embedded program/data, platform callbacks and the host build. |
| [Instruction memory](instruction-memory.md) | Final image size, branch ranges, reduction unrolling and profiling. |
| [FP8 numerical contract](fp8-numerical-contract.md) | Codes, exponents, the admitted MXU1 code set, and the declared arithmetic. |
| [Already-quantized intake](already-quantized-intake.md) | Consuming FP8 codes and int32 exponents unchanged: capture, source operations, admission, and masks. |
| [Already-quantized K32 domain](already-quantized-k32-domain.md) | The BF16 readout domain that K32 partial sums stay within on MXU1. |
| [PACK scale domain](pack-scale-domain.md) | Raw PACK scale codes, their source evidence, and the guarded conversion. |
| [Single-program profiling](single-program-profiling.md) | Connected graphs in one image, model execution and measurement scope. |
| [Hardware status](hardware-status.md) | Per-instruction status, RTL/model differences, and how to rerun the RTL core tests. |
| [`selected-variant-inventory.json`](selected-variant-inventory.json) | Machine-readable list of decoder modes; read by the build and tests. |
