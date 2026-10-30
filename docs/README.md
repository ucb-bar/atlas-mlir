# Atlas MLIR docs

Start with the [repository README](../README.md) for the build, and the
[examples](../examples/handoff/README.md) for runnable commands.

| Doc | Read it for |
| --- | --- |
| [Dialect reference](dialect-reference.md) | Virtual and machine operations, verifiers, passes, tools, and how to add a pass. |
| [Compiler pipeline](compiler-pipeline.md) | Virtual lowering, the SSA interpreter, the physical program format, atomic DMA, and the LLVM handoff. |
| [DMA across PACK](pack-dma-overlap.md) | Why explicit DMA may stay in flight through a PACK relayout loop, how each stage proves it, and the measured effect. |
| [PACK relayout loop](pack-relayout-loop.md) | The software-pipelined copy that repairs PACK's row layout, and the measured effect. |
| [Native Merlin binding](native-merlin-binding.md) | Compiling graphs through Merlin: supported graphs, CLI, outputs, precision policies, and limits. |
| [Host launch](host-launch.md) | Host C runtime, memory map, mailbox, cache hooks, reset entry, and board builds. |
| [Traffic reuse](traffic-reuse.md) | Opt-in DMA reuse recipes, admission and validation. |
| [Full-model contracts](full-model-contracts.md) | Bundles, placement, UART results, and Tiny segment inputs. |
| [Full Smol host](smol-full-host.md) | Policy inputs, program selection, and the phase archive. |
| [Model tile runner](model-tile-runner.md) | Source-bound model arithmetic, contraction providers and workload execution. |
| [Complete ResNet50 host](resnet-full-host.md) | Generate programs, replay the graph, and build a full-host bundle. |
| [Task accuracy](task-accuracy.md) | How bit-exact Atlas results differ from PyTorch, and possible opt-in precision modes. |
| [RV64 host ISS](host-iss.md) | Execute a compatible ResNet ELF and external fixture bundle with a model device. |
| [Hardware status](hardware-status.md) | Per-instruction status, RTL/model differences, and how to rerun the RTL core tests. |
| [`selected-variant-inventory.json`](selected-variant-inventory.json) | Machine-readable list of decoder modes; read by the build and tests. |
