# Generator reuse: coverage, target limits and qualification

The opt-in Python generator admits 57 exact physical shapes: 53 qualified full-model shapes plus four separate selected-layer Smol shapes. The default generator remains unchanged. No compiler lowering, dialect, launch ABI or hardware protocol change is required. G0-G7 remain shelved.

```python
from atlas_program import assembly, code_sources, dma_load_bytes

m, k, n = 128, 3072, 768
sources = code_sources(traffic_reuse=True)
source = assembly(m, k, n, m*k + k*n + 256, -15, traffic_reuse=True)
loads = dma_load_bytes(m, k, n, traffic_reuse=True)
assert code_sources(traffic_reuse=True) == sources
```

`atlas_program._TRAFFIC_REUSE_MODULES` defines the exact allowlist. Five helpers implement resident, split-bank streaming, Tiny residency, Tiny prefetch, and shared A-block/B-column recipes. The blocked helper covers ResNet and Smol, including selected-layer down projection. `code_sources(traffic_reuse=True)` binds all five helpers; the default source set is unchanged. Legacy residency/pretranspose flags cannot be combined with reuse. Admission requires the original guarded output offset and a finite integer BF16 rescale. Unknown shapes fail rather than falling back to an unqualified recipe.

## Measured kernels and launch-weighted estimates

Retained native pairs use **NPU Model curve**, revision `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`, RTL numerics, profile `e5bc2419675e91138045da25f54cf3c4b071b99ab39471fe1d75219d94ab70c9`. Cycles start at the first issued instruction and end at drained ECALL, inclusive. Captured inputs, output addresses and the original 4-GiB DRAM base remain unchanged. No timing constants were fitted.

The combined Launch-Weighted Curve Transfer Model applies measured captured-shape deltas to matching launches, leaving other kernels unchanged. Each delta comes from a measured receipt that records its inputs, cycles, DMA bytes, DMA.WAIT stalls and program hash.

| Scope | Qualified shapes / total | Qualified launches / total | Estimated kernel-cycle reduction |
|---|---:|---:|---:|
| ResNet full launch ledger, raw and graph capacity | 21 / 21 | 54 / 54 | 57.423% |
| Tiny full launch ledger | 10 / 10 | 840 / 840 | 55.136% |
| Smol full-policy launch ledger | 22 / 24 | 2703 / 2715 | 61.658% of measured 2713-launch subtotal |
| Separate selected Smol layer | 4 / 4 | 7 / 7 | 68.896% |

These are representative contraction-kernel transfer estimates, excluding host and noncontraction work, not full-model executions. The selected layer is separate from full-policy totals. Smol `64/736/32` improved only 1.848% and retains its original kernel. Smol `64/12288/960` has two unmeasured launches: its recorded short-run admission exceeded the wall budget, so no full run or candidate was started. The complete-policy Smol percentage remains unknown.

**All 54 ResNet launches fit the graph copy buffer with compact loops and next-tile prefetch together.** For exactly `12544/160/64`, contiguous row pairs within one VMEM bank share a scalar loop. Cross-bank pairs and odd tails retain explicit bodies. The weight-column preloader also applies here: its channel-1 DMA drains before the row-pair loop reuses scalar register x24. Blocks, DMA byte counts, ordered accumulations and guarded output layout remain unchanged. The combined program is 1,374 words, below the 8,192-word graph limit, compared with 1,337 words for compact reuse before prefetch and 13,777 for prefetch before compaction.

Fresh combined full-image checks measure 3,500,239 curve cycles and 1,274,370 FIFO43 cycles. R1's compact receipts measured 3,504,686 and 1,278,154, so prefetch saves 4,447 curve cycles (0.127%) and 3,784 FIFO43 cycles (0.296%). Relative to R2's raw-only prefetch recipe, compaction adds 3,554 curve cycles (0.102%) and 4,213 FIFO43 cycles (0.332%). Original graph-fallback receipts measured 5,174,282 curve and 1,693,078 FIFO43 cycles. Both fresh runs check every byte of the 3,623,424-byte original DRAM image, including outputs, inputs and guards, with zero mismatches.

The combined Launch-Weighted Curve Transfer Model admits all 21 ResNet shapes and 54 of 54 launches. Conditional reuse adoption changes the kernel subtotal from 195,740,569 to 83,340,360 cycles, a 57.423% reduction. This incorporates R2's representative deltas and the fresh compact-prefetch timing and replaces the historical 53-launch graph scenario. The estimate excludes host and noncontraction work; no full graph was executed, and graph builds do not enable reuse automatically.

The original three-shape and eight-shape checkpoints are historical; their smaller tables are not the current allowlist. [Initial prototype pairs](generator-traffic-reuse.md) retain the original measurements.

## R2 DRAM-floor follow-up

R2 derives unique A + W + BF16 output floors for all 57 shapes from the pinned curve backend, including read/write history pacing and overlappable request-slot service. These optimistic bounds relax compute, VMEM capacity and dependencies. The eight DMA channels do not multiply the one-channel DRAM bandwidth. Tiny 32/2048/2048 and 32/5632/1024 are only 3.94% and 2.16% above their floors and remain unchanged.

A shared resident-column preloader queues the next disjoint 1-KiB W tile on channel 1 during the current transpose, waits before use, and suppresses the final speculative load. It uses existing storage and established delays. Single-K-tile columns omit the unused configuration. No DMA byte count, K order, rescale site, public admission or default-generator program changes.

| Highest-gap physical shape | Prior reuse curve | R2 curve | Reduction |
|---|---:|---:|---:|
| 128/2048/2048 | 7,191,210 | 6,630,763 | 7.794% |
| 128/768/768 | 1,153,869 | 1,077,142 | 6.650% |
| 128/768/3072 | 4,410,927 | 4,104,016 | 6.958% |
| 128/3072/768 | 4,162,791 | 3,845,872 | 7.613% |
| 64/736/2048 | 2,110,160 | 1,914,449 | 9.275% |

In R2, all 50 changed shapes passed full-image native curve and FIFO43 checks with no cycle regression on either profile. The launch-weighted estimates above include these deltas and the fresh combined compact recipe and remain conditional adoption models. The retained live-layer/QEMU and SoC evidence below uses pre-R2 programs; three changed representatives are selected for fast-check ELF handoff and owner-run SoC execution. R2 makes no new hardware or live-network timing claim.

## Exactness and tests

Qualified representatives match the original expected full DRAM image, including output, inputs and guards. Recipes preserve increasing K order, first overwrite then ordered accumulation, up to two MXU0 slots, one BF16 pop and the original power-of-two rescale. They do not pop partial reductions or reassociate arithmetic. Exactly one static `vli.all m4` patch site changes with the exponent.

Shared in-repo symbolic tests exercise every public shape: operand origins and transpose, accumulator ordering, output halves/scatter, pending DMA ownership on both channels and bank-contained accesses. Mutation controls remove required waits, transpose, accumulation, rescale or correct output placement. Receipt-derived DMA counts anchor dispatch accounting. Default-generator compatibility controls remain separate.

Execution qualifies the recorded captures. Other exponents use the structural argument that only the unique rescale literal changes, unless a separate captured run is recorded. Source proof is not additional numerical execution. The morning receipt rebind documents the pre-R2 binaries after merging helpers. R2 binds 50 changed programs to fresh curve/FIFO43 full-image receipts, with seven other programs byte-identical to the retained reuse programs. The combined stack regenerates all 57 capture programs: 56 are byte-identical to R2, including seven also identical to R1. Only the compact-prefetch `12544/160/64` program differs from both and is freshly qualified on curve and FIFO43. The shared ownership mutations also cover this compact shape.

## VMEM and program limits

EE290 has 1.5 MiB VMEM in six 256 KiB banks. Recipes use banks 0-1, device bytes `[0,524288)`, CPU alias `[0x20000000,0x20080000)`. VMEM operands count four-byte words; DRAM operands count bytes. Each vector/DMA tile stays inside one bank.

Safety requires known quiescence and exclusive raw ownership. Raw metadata, program assets and 4-GiB DRAM staging are outside these ranges. Compiled clients can allocate a mailbox and tensors in the same banks; mixed compiled/raw clients are not admitted. Recipes neither reserve banks nor implement arbitration.

| Consumer | Program limit |
|---|---|
| Real raw host | Raw words + five marker words ≤ 32,768 |
| ResNet graph host and segments | Raw copy buffer ≤ 8,192 words, plus separate hardware limit |
| Retained standalone RTL driver | ≤ 8,192 loaded words including its eight-word wrapper |

The largest current recipe is `3136/576/64`: 3,787 raw words, or 3,792 with raw-host markers. All 57 admitted programs fit the graph capacity. Builders validate actual byte and descriptor counts. Unconditional tests conservatively bound every generated program against both hardware and graph limits.

## Host checking and selected-layer handoff

`ATLAS_FAST_CHECK` remains opt-in. The checker compares every stable output/input/guard byte with an aligned OR-of-word-XOR reduction, byte prefixes/tails and a failure-only byte scan. If pass one differed but the rescan cannot locate a mismatch, it still fails with a nonzero count and an unknown-offset sentinel. DMA completion and stable spans remain required. The default checker is unchanged.

`tools/chipyard/smol_layer.py` builds one live selected Smol layer in q, k, v, o, gate, up, down order. It retains checksum-bound nonlinear C and final output/key/value checks, adds exact full-DRAM checks after each reuse launch, and emits `ATLAS_KERNEL_*` lines plus an `ATLAS_LAYER_*` total. Current recipes require at most 380 wrapped words; shared DRAM is 3,625,472 bytes.

A combined live QEMU and NPU Model run of the layer was not attempted: the measured component run times already exceed the run time limit before host overhead. Whole-host QEMU checks with captured device outputs and separate bounded numerical components are distinct from combined live Model execution. QEMU intercepts HTIF console output, so real frontend transport is left to an SoC or FPGA run; the union bit-copy fix reduces the measured host work from 5.50 to 1.55 billion QEMU instructions with captured device outputs. These are not Rocket cycles or a whole-system timing estimate; see [host bit copies](host-bit-copies.md).

## Standalone RTL and SoC evidence

Standalone AtlasCore checks use the pinned assertion-enabled Verilator binary and synthetic FIFO43 responder. They validate exact outputs, guards, drainage and retained-context invocations. Observer counts include setup and are not NPU Model curve, NPU Model FIFO43 or EE290 timing. Earlier breadth checks are recorded separately.

On the EE290 SoC in VCS, the pre-R2 reuse recipes cut device cycles 57.9% for Smol, 54.8% for ResNet and 60.8% for Tiny, and the fast check cuts wall time 62-71%. Nine more reuse kernels, three per model, also pass on the SoC with device cycles within 0.6-1.9% of the NPU Model curve.

Agent simulations are capped at 3,490 seconds plus at most five seconds cleanup; probes have smaller recorded caps. Numerical threads are one and total affinity stays within 16 CPUs. Full Tiny live, full-policy Smol live and the 3,555-kernel batch remain outside scope. Qualification implies no default promotion, push or PR.