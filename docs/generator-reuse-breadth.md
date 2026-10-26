# Generator reuse: coverage, target limits and qualification

The opt-in Python generator admits 57 exact physical shapes: 53 qualified full-model shapes plus four separate selected-layer Smol shapes. The default generator remains unchanged. No compiler lowering, dialect, launch ABI or hardware protocol change is required. G0–G7 remain shelved.

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

Launch-Weighted Curve Transfer Model v2 applies measured captured-shape deltas to matching launches, leaving other kernels unchanged. Its named inputs, cycles, DMA bytes, DMA.WAIT stalls and program hashes are in `/tmp/reednicolas-atlas/sota-fast/overnight/catalog-final.json` and its referenced receipts.

| Scope | Qualified shapes / total | Qualified launches / total | Estimated kernel-cycle reduction |
|---|---:|---:|---:|
| ResNet full launch ledger, raw-host capacity | 21 / 21 | 54 / 54 | 55.548% |
| Tiny full launch ledger | 10 / 10 | 840 / 840 | 53.718% |
| Smol full-policy launch ledger | 22 / 24 | 2703 / 2715 | 58.379% of measured 2713-launch subtotal |
| Separate selected Smol layer | 4 / 4 | 7 / 7 | 66.736% |

These are representative contraction-kernel transfer estimates, excluding host and noncontraction work, not full-model executions. The selected layer is separate from full-policy totals. Smol `64/736/32` improved only 1.848% and retains its original kernel. Smol `64/12288/960` has two unmeasured launches: its recorded short-run admission exceeded the wall budget, so no full run or candidate was started. The complete-policy Smol percentage remains unknown.

**Graph eligibility differs from raw coverage.** ResNet `12544/160/64` emits 13,740 raw words: it fits the real raw host but exceeds the graph host's 8,192-word copy buffer. Both graph builders reject its actual encoded bytes before linking. If the other 53 ResNet launches adopt reuse while this launch stays original, Launch-Weighted Curve Transfer Model v3 (graph eligibility) estimates 54.693% fewer kernel cycles from the same v2 receipts. This is an adoption scenario; graph builds do not enable reuse automatically.

The original three-shape and eight-shape checkpoints remain historical evidence under `/tmp/reednicolas-atlas/sota-fast/generator-reuse/` and `generator-breadth/`. Their smaller tables are not the current allowlist. [Initial prototype pairs](generator-traffic-reuse.md) retain the original measurements.

## Exactness and tests

Qualified representatives match the original expected full DRAM image, including output, inputs and guards. Recipes preserve increasing K order, first overwrite then ordered accumulation, up to two MXU0 slots, one BF16 pop and the original power-of-two rescale. They do not pop partial reductions or reassociate arithmetic. Exactly one static `vli.all m4` patch site changes with the exponent.

Shared in-repo symbolic tests exercise every public shape: operand origins and transpose, accumulator ordering, output halves/scatter, pending DMA ownership on both channels and bank-contained accesses. Mutation controls remove required waits, transpose, accumulation, rescale or correct output placement. Receipt-derived DMA counts anchor dispatch accounting. Default-generator compatibility controls remain separate.

Execution qualifies the recorded captures. Other exponents use the structural argument that only the unique rescale literal changes, unless a separate captured run is recorded. Source proof is not additional numerical execution. The morning receipt rebind checks emitted binaries against retained numerical receipts after merging helpers; it supplies no new timing measurement.

## VMEM and program limits

EE290 has 1.5 MiB VMEM in six 256 KiB banks. Recipes use banks 0–1, device bytes `[0,524288)`, CPU alias `[0x20000000,0x20080000)`. VMEM operands count four-byte words; DRAM operands count bytes. Each vector/DMA tile stays inside one bank.

Safety requires known quiescence and exclusive raw ownership. Raw metadata, program assets and 4-GiB DRAM staging are outside these ranges. Compiled clients can allocate a mailbox and tensors in the same banks; mixed compiled/raw clients are not admitted. Recipes neither reserve banks nor implement arbitration.

| Consumer | Program limit |
|---|---|
| Real raw host | Raw words + five marker words ≤ 32,768 |
| ResNet graph host and segments | Raw copy buffer ≤ 8,192 words, plus separate hardware limit |
| Retained standalone RTL driver | ≤ 8,192 loaded words including its eight-word wrapper |

The largest recipe is `12544/160/64`: 13,740 + 5 = 13,745 words. It is raw-only under the current graph limit. Other admitted programs fit the graph capacity. Builders validate actual byte and descriptor counts. Unconditional tests conservatively bound all generated programs against hardware and identify the graph exception.

## Host checking and selected-layer handoff

`ATLAS_FAST_CHECK` remains opt-in. The checker compares every stable output/input/guard byte with an aligned OR-of-word-XOR reduction, byte prefixes/tails and a failure-only byte scan. If pass one differed but the rescan cannot locate a mismatch, it still fails with a nonzero count and an unknown-offset sentinel. DMA completion and stable spans remain required. The default checker is unchanged.

`tools/chipyard/smol_layer.py` builds one live selected Smol layer in q, k, v, o, gate, up, down order. It retains checksum-bound nonlinear C and final output/key/value checks, adds exact full-DRAM checks after each reuse launch, and emits `ATLAS_KERNEL_*` lines plus an `ATLAS_LAYER_*` total. Its maximum wrapped program is 361 words; shared DRAM is 3,625,472 bytes. The build-only handoff is `/tmp/reednicolas-atlas/sota-fast/soc-spot/MANIFEST-layer.json`.

Combined live QEMU+Model was not admitted: the named retained component elapsed-time sum in `morning/layer/admission.json` already exceeds the agent cap before host overhead. Whole-host QEMU checks with captured device outputs and separate bounded numerical components are distinct from combined live Model execution. QEMU intercepts HTIF console output; real frontend transport remains for the owner's SoC check.

## Standalone RTL and owner-run SoC evidence

Standalone AtlasCore checks use the pinned assertion-enabled Verilator binary and synthetic FIFO43 responder. They validate exact outputs, guards, drainage and retained-context invocations. Observer counts include setup and are not NPU Model curve, NPU Model FIFO43 or EE290 timing. Overnight checks are recorded in `/tmp/reednicolas-atlas/sota-fast/overnight/REPORT.md`; earlier breadth checks remain separate.

The owner separately ran the original/reuse and fast-check SoC handoffs. `/tmp/reednicolas-atlas/sota-fast/soc-spot/results/REPORT.md` records EE290 device-cycle reductions of 57.9% Smol, 54.8% ResNet and 60.8% Tiny, and fast-check wall reductions of 62–71%. These named owner-run SoC results are distinct from this agent's model/RTL runs and the earlier conditional checker-transfer scenario. This agent does not run VCS.

Agent simulations are capped at 3,490 seconds plus at most five seconds cleanup; probes have smaller recorded caps. Numerical threads are one and total affinity stays within 16 CPUs. Full Tiny live, full-policy Smol live and the 3,555-kernel batch remain outside scope. Qualification implies no default promotion, push or PR.
