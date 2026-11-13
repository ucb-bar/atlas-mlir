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
| [CFG DMA timing](timing-dma-cfg.md) | Generic control-flow DMA proof and explicit immutable-input assumptions. |
| [Host launch](host-launch.md) | Host C runtime, memory map, mailbox, cache hooks, reset entry, and board builds. |
| [Baremetal-IDE integration](baremetal-c-integration.md) | Generated C/H, embedded program/data, platform callbacks and the host build. |
| [Instruction memory](instruction-memory.md) | Final image size, branch ranges, reduction unrolling and profiling. |
| [Quantized operand reuse](quantized-operand-reuse.md) | Optional MXU1 activation residency, shared Merlin planning, limits, and validation. |
| [Scratch output priority](scratch-output-priority.md) | Native scheduling through cache dependencies and complete-layer measurements. |
| [Panel readout](panel-readout.md) | Keeping completed MXU tiles on chip through scaling and epilogues. |
| [Row dispatch budget](row-dispatch-budget.md) | Batching complete softmax rows under the selected execution budget. |
| [Private staging](private-staging.md) | Keeping private BF16 intermediates and repeated immutable inputs in invocation-local VMEM. |
| [Quantizer input permutations](quantizer-input-permutation.md) | Folding private layouts into byte-preserving consumer packing. |
| [Model compilation followups](model-compilation-followups.md) | Additional optimizations, prerequisite PRs, and feature switches. |
| [Single-program profiling](single-program-profiling.md) | Connected graphs in one image, model execution and measurement scope. |
| [Hardware status](hardware-status.md) | Per-instruction status, RTL/model differences, and how to rerun the RTL core tests. |
| [`selected-variant-inventory.json`](selected-variant-inventory.json) | Machine-readable list of decoder modes; read by the build and tests. |
