# Remaining model compilation optimizations

Reconciled on 2026-11-16 against Atlas PR #50 at `0748e5a` and Merlin PR #63
at `cadc2080b`. The next implementation branch in each repository is
`perf/model-traffic-reduction`. The old `perf/model-dispatch-reduction` branches
and their measurement artifacts remain unchanged.

## What is already covered

The current PR heads include the prior dispatch-reduction stack: exact cast
boundary folding, dense storage views, operand layout folding, operand and
exponent-page caches, panel readout, private VMEM staging, per-scratch-buffer
scheduling, and execution-budget-aware quantizer and softmax dispatches.
Merlin common-value elimination is also already available. None of these is a
new opportunity to claim or reimplement.

The updated heads additionally make compiler recipes explicit options, repair
I/O placement and replay, and improve compiler implementation performance.
Historical environment-only launch scripts must be translated to those options.
Use a fresh native snapshot with the selected I/O capacity and current Merlin
sources; do not bypass snapshot checks to reuse an older snapshot.

## Evidence and order

The retained exact single-layer measurements at `ac077e3` are:

| Layer | Device cycles | Dispatches | DRAM bytes | Serial same-work resource bound |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama, B=1, 64 tokens | 74,114,788 | 56 | 79,481,664 | 54,724,767 |
| SmolVLA vision, B=1, 1,024 tokens | 491,010,121 | 113 | 459,809,376 | 353,461,374 |

These are Rust model measurements with RTL numerics, not hardware timings or
full-model results. The resource bound preserves the current work and serial
launch boundaries; it is not a speedup prediction. A fresh PR-head control is
required to isolate each new optimization from upstream changes.

| Priority | Work | Evidence / expected benefit | Ownership |
| --- | --- | --- | --- |
| 1 — implemented | Admit valid zeros through the vector check for exact BF16 addition/subtraction | Saves 2.60M TinyLlama and 50.85M SmolVLA cycles in matched full-layer runs; details below. | Atlas: proof against the selected VPU primitives and native route selection. |
| 2 | Fuse private softmax output with static activation quantization | SmolVLA writes a 25,165,824-byte BF16 probability tensor that its quantizer reads back. Removing that pair of transfers removes about 34.1M cycles of bandwidth demand at 1.477 B/cycle; actual saved cycles require simulation. | Merlin: reusable private producer/consumer ownership proofs. Atlas: row schedule, original BF16 rounding, exact FP8 conversion and exponent publication. |
| 3 | Fold exact bias/residual additions into producer epilogues | SmolVLA's six projection bias additions still require 54 launches. Their producer results total 14,155,776 BF16 bytes; eliminating one write/read pair would remove 28.31 MB of traffic. This requires the original post-projection rounding and safe fallback/publication, not substitution of the VPU's truncated add for exact RNE. | Merlin: private result and broadcast-layout proofs. Atlas: exact arithmetic and producer-store fusion. |
| 4 | Pipeline tile DMA with vector compute and validation | Before priority 1, exact-float and folded-layout kernels accounted for about 7.26M TinyLlama and 106.75M SmolVLA cycles in which DMA was inactive. Remeasure after scalar work removal before selecting schedules. Prefetch the next independent tile while processing the current one. | Atlas: VMEM lifetimes, register pressure, DMA slots and instruction timing. |
| 5 | Compose compatible regions while keeping private intermediates on device | The current 56/113 launches include real data dependencies as well as image/resource limits. Composition must remove repeated publication/packing or enable overlap to reduce device cycles; merely concatenating launches chiefly changes host overhead. | Merlin: generic composition, ownership and lifetime contracts. Atlas: admitted schedules, memory placement and image bounds. |
| 6 | Extend independent-request batching and weight reuse | B=2 already improves TinyLlama throughput. Further batching amortizes weights but increases activation storage and attention work per batch. Measure throughput separately from single-request latency. | Merlin: independent-request graph semantics. Atlas: panel selection and residency. |

TinyLlama's seven projections contain about 44 MB of FP8 weights. At the
selected model bandwidth, fetching them once already costs about 29.8M cycles
of service. More scheduling alone cannot remove that traffic. Higher DMA busy
percentage and higher MXU busy percentage are not objectives when achieved by
doing extra work. Optimize end-to-end cycles, useful MAC throughput and traffic.

General MXU0 substitution, approximation of the exact float regions, and changing
reduction/rounding order are separate numerical-policy projects. They are not
exact rewrites under the current comparison contract.

## First implementation: zero-aware vector admission

The opt-in `exact_binary_vector_preflight` route first retains its cheap min/max
check. If the minimum magnitude is zero, it additionally checks each operand
using `t = min(abs(x), 2^-32) * 2^32` and the probe `t * (1 - t)`.
Within the finite bounded domain, the probe is zero exactly when the original
operand is zero or has magnitude at least `2^-32`. The existing upper-magnitude
check rejects values above `2^32`, infinities and NaNs.

The smallest nonzero BF16 input becomes `2^-101`, so an inadmissible small value
cannot disappear through underflow in the probe. The nearest representable
value below one is `255/256`, so subtraction cannot turn it into zero either.
The arithmetic operands are never replaced by probe values. Scalar fallback,
whole-input validation, atomic output publication and resource checks remain.

This is a runtime domain check derived from the operation's numeric contract.
It does not depend on model names, captured values, calibration or reference
outputs. It needs no new Merlin optimization or target-specific shared pass.

### Matched full-layer results

Fresh builds at the PR heads reproduce the retained baseline cycle counts
exactly. Both variants use the same Merlin snapshot, inputs, weights, native
tools, compiler options and Rust executable. Only the Atlas vector admission
implementation differs. All 131,072 TinyLlama and 786,432 SmolVLA output words
match the independent original-FX Atlas-arithmetic references.

| Fixture | Before cycles | After cycles | Saved | Dispatches |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama B=1, 64 tokens | 74,114,788 | 71,519,269 | 2,595,519 (3.50%) | 56 → 56 |
| SmolVLA vision B=1, 1,024 tokens | 491,010,121 | 440,156,657 | 50,853,464 (10.36%) | 113 → 113 |

Instructions fall from 7,786,086 to 5,107,077 and from 77,654,613 to 25,165,229,
respectively. Every launch retains its exact DRAM read/write byte counts.
Every unchanged program image retains identical cycles, instruction histogram
and engine-busy counters. Changed images are confined to exact-float additions
and their folded-input-layout wrappers.

The nonzero route pays a small branch overhead: TinyLlama's entirely admitted
residual add grows from 1,641,238 to 1,642,006 cycles (+768, or 0.047%). Its other
three changed additions save 1,204,064, 150,508 and 1,241,715 cycles. The complete
layer improvement includes this regression. SmolVLA has 42 slightly slower
launches, totaling 32,256 additional cycles; its reported net saving includes
all of them.

DMA busy fraction increases from 79.49% to 82.37% for TinyLlama and from 68.24%
to 76.12% for SmolVLA. These are engine-busy fractions, not payload bandwidth or
useful MXU utilization. The total time decreases without adding DMA traffic.
The recalculated serial same-work resource bounds are 53,811,754 and
311,310,882 cycles. Further traffic elimination can lower those bounds.

The runs retain Rust `be1b521015ff54ec75699c2d5d63e567ab0ae22a`, CPU, RTL
numerics, `register_rows`, 64 MiB I/O, 128,000,000 dynamic instructions and the
unchanged 128 KiB physical IMEM limit. Host packing/dispatch time is excluded.
SmolVLA is the independently calibrated vision block, not an action-expert or
full-policy run.

Directed validation passes six tests and ten subtests, including exhaustive
65,536-encoding admission analysis and Rust add/subtract runs with partial and
multiple tiles, signed zeros, tiny values next to zero, both input positions,
NaNs, infinities, immutable inputs and atomic late rejection. Float/layout
regressions pass 36 tests and 20 subtests; two Python-simulator tests are
deliberately skipped because execution uses Rust. A rehashed instruction-bound
refusal test also passes. All full applications pass fresh native publication,
memory, timing, allocation and resource verification.

Local reproduction and receipts are in the Atlas worktree's
`out/traffic-audit/`: `env.sh`, `compile_layer.py`, `run_layer.sh`,
`compare_layer.py`, `zero-preflight-comparison.json`, and the
`rust-{tinyllama,smolvla}-{baseline-v2,zero-domain}` directories. Each result
binds its source, application, simulator and exact output by SHA-256.
The clean baseline source is retained at `out/traffic-audit/atlas-baseline`.

## Acceptance gate

For every feature, cover multiple shapes, partial tiles, signed zeros, domain
boundaries, malformed inputs and resource refusals. Compare all output words
and protected memory using Rust. Compile and execute both complete retained
layers with matched options and source/model hashes. Report cycles, dispatches,
instructions and DRAM bytes, including regressions. Commit and push only after
the feature's independent checks and full-layer comparisons pass.
