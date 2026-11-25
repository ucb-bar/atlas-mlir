# Atlas compiler design

This page explains how a model reaches Atlas, who decides what, and which optional schedules exist. It replaces the earlier per-optimization notes.

## Flow

1. An atlas-torchao export carries FP8 codes and int32 exponents.
2. model2MLIR captures it, and Merlin's application compiler partitions the graph into host and device regions.
3. The Atlas native provider (`python/atlas_mlir_native`) admits each device region under the [FP8 numerical contract](fp8-numerical-contract.md) and lowers it to Atlas virtual SSA.
4. `atlas-opt` lowers virtual SSA to machine IR, and `atlas-emit` writes the instruction words ([compiler pipeline](compiler-pipeline.md)).
5. Merlin emits the host C that launches each image; Baremetal-IDE links the host ELF ([C integration](baremetal-c-integration.md), [host launch](host-launch.md)).
6. `atlascc deploy` runs these steps, checks the result against an independent reference, and writes a kit ([deployment](deployment.md)).

Exports that are already quantized keep their codes and exponents unchanged ([already-quantized intake](already-quantized-intake.md)).

## Who decides what

Merlin owns the target-neutral parts: capture, the semantic graph, partitioning, liveness, placement, residency planning, host math, and the host C.

Atlas owns the device facts: the device families and their admission rules, the FP8 contract, tile sizes and memory capacities, the lowering of each admitted region, and every instruction word.

Atlas supplies the region plan today through a priority order and a selected fusion list. That grouping is temporary. Its replacement is declared implementation families with Merlin selecting the cover.

## Numerics

Each admitted device route declares its arithmetic, and admission refuses inputs outside the declared domain. MXU1 contractions keep the ascending K32 order, the BF16 readout after each K32 step, and the final exponent restoration. The [K32 domain proof](already-quantized-k32-domain.md) bounds every intermediate readout, and the [PACK scale audit](pack-scale-domain.md) covers the raw scale codes.

No schedule on this page changes arithmetic. Each one changes only where operands live or how often they are loaded, and each candidate still passes the instruction-memory, allocation, DMA, timing, and publication checks that the default schedule passes. A candidate that fails a check is refused, not relaxed.

## Pipelines

`--native-pipeline mandatory` is the default and the only pipeline deploy uses. `--native-pipeline optimized` reschedules the program; only the compiler verifiers and encoding checks examine the rescheduled program, while the native semantic checks run on the mandatory lowering of the same virtual program. See [native binding](native-merlin-binding.md) for the full option list.

## Optional schedules

These are off by default. Each is a field of `CodegenOptions` in `python/atlas_mlir_native/options.py`, so each enters the compilation identity and the materialization cache key. Pass the same flags to `native-verify` as to `native-compile`.

| Option | What it changes |
| --- | --- |
| `--quantized-operand-reuse row-cache` | Keeps an FP8 activation row in VMEM across output-column tiles, up to 64 K32 tiles. |
| `--quantized-operand-reuse prefix-cache` | As `row-cache`, and for wider rows keeps the first 64 tiles and streams the rest in order. |
| `--quantized-operand-reuse wide-cache` | Keeps up to 192 K32 activation tiles (192 KiB); longer rows cache a bounded prefix. |
| `--quantized-operand-reuse paired-cache` | Processes pairs of M32 tile rows so each weight column loads once for both rows (at most 96 KiB of slots). |
| `--quantized-operand-reuse panel-cache` | Chooses a panel of 2, 4, 8, 16, or 32 tile rows by weight and partial-result traffic. |
| `--looped-operand-reuse resident-a` | Keeps up to 24 activation tiles in registers across the N loop; requires the mandatory pipeline. |
| `--staging-storage vmem` or `vmem-inputs` | Moves private BF16 intermediates, and with `vmem-inputs` repeated immutable inputs, into invocation-local VMEM. |

The selection rules use tensor shapes, layouts, and resource limits only. No model name, calibration value, or observed input enters a selection.

## Recipe

Atlas declares no recipe. Deploy uses the mandatory pipeline and places every window in Merlin's memory budget of a declared region (see [deployment.md](deployment.md)).

## Measured effect

These numbers come from Rust-model runs with RTL numerics on earlier branches. They were not rerun on this head, and none ran on silicon.

| Fixture | Baseline | With the earlier dispatch-reduction recipe and later folds |
| --- | --- | --- |
| One TinyLlama decoder layer | 226,725,080 cycles, 586 dispatches | 82,772,819 cycles, 59 dispatches |
| One SmolVLA vision block | not recorded on the same basis | 766,046,001 cycles, 147 dispatches |

Every output word of both fixtures matched the independent selected-arithmetic reference in those runs.

## Open work

One dispatch per layer is not reached. The distinct images of one layer total more than the 131,072-byte IMEM, and adjacent kernels disagree on physical layout and lifetime, so they cannot be concatenated without a shared layout contract.

The optimized pipeline and its switches follow separately, once the board spill limit is fixed.

The priority order and fusion list are replaced by declared families with Merlin selecting the cover, as stated above.
