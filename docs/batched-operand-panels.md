# Reuse operand panels across independent batches

Set `ATLAS_QUANTIZED_OPERAND_REUSE=batch-panel-cache` to consider M/K panels
inside a counted batch loop. `ATLAS_EXPONENT_PAGE_CACHE=1` also retains scalar
and per-batch scale pages. The default operand policy is unchanged.

Selection depends on tensor geometry, storage layout and resource limits. No
model names, calibration values or observed input values participate. Shared
matrix weights retain the existing flattening optimization where it is legal;
batched weights and per-batch scales retain independent matrix products. The
batch index selects each matrix's operands, output and scale pages. This does
not concatenate attention contexts or mix independent requests.

The new policy also corrects the panel traffic ranking: an inline final
restoration does not need a BF16 partial-result write/read round trip. Historical
`panel-cache` and `paired-cache` selections keep their previous ranking and
virtual programs for publication replay.

Genuinely batched panels currently require divisible M tile groups, at least two
N tiles, one complete K reduction resident within the 192 KiB panel reservation,
and bounded inline restoration. Ineligible geometries retain the prior row
cache. Native allocation, memory initialization, runtime instruction bounds and
the unchanged 128 KiB encoded IMEM limit still decide admission. Proved image
or instruction-budget refusals may retry the older policies. Compilation errors,
failed proofs and timeouts are never converted into successful fallbacks.

## Atlas implementation boundary

This is target-specific scheduling and allocation, so the implementation lives
in Atlas. Merlin's source graph and numerical policy remain unchanged.

One guarded batch counter supplies aligned offsets instead of carrying five
buffer pointers through nested loops. Both the live SSA analysis and the
independent captured-CFG verifier prove the unsigned range, at least 32-byte
alignment of every interior value, and absence of arithmetic overflow. The
complete DMA extent must still fit its declared memory window.

When early scheduling of address arithmetic creates scalar register pressure,
the spiller can move a pure single-use expression and its single-use scalar
operands next to their consumer in the same block. It preserves the expression
needed by the address proof, performs no memory spill or extra arithmetic, and
uses the existing bounded rewrite budget. Each root is moved at most once per
spill pass to prevent oscillation between expressions sharing a consumer.

Ordered BF16 rounding, exponent validation, scalar restoration fallback, input
and guard memory, and atomic generated-C publication remain checked. Selection
and geometry are recorded and rederived on replay independently of environment.

## Validation

Directed Rust tests cover multi-axis batches, shared and batched weights, row,
scalar and batch scale combinations, partial M/N/K tiles, wide reductions,
late invalid scales and generated-C fused epilogues. Every output word is
compared with independent single-matrix invocations of the previous schedule.
Native tests reject unaligned interior addresses despite aligned endpoints,
wrapped shifts/additions, absent guards and tampered captured bounds; pressure
regressions check schedule verification and the unchanged rewrite budget.

The frozen combined suite passes 108 tests and 119 subtests; native scheduler
and spill regression tests add 28 tests and 364 subtests. Twelve representative
historical panel programs remain byte-identical at the virtual-source level.
The new pressure test schedules 32 DMA address expressions successfully while
still rejecting an insufficient rewrite budget.

## Complete-layer results

The baseline is the [invariant exponent cache](exponent-page-cache.md), commit
`61e4cc2`, with the same inputs, weights, arithmetic and Rust simulator.

| Fixture | Before cycles | After cycles | Saved | Dispatches |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama B=1, 64 tokens | 74,571,452 | 74,114,788 | 0.61% | 56 → 56 |
| SmolVLA vision block B=1, 1,024 tokens | 520,987,059 | 491,010,121 | 5.75% | 113 → 113 |
| TinyLlama B=2, independent 64-token requests | 130,207,038 | 125,452,250 | 3.65% | 64 → 64 |

All 131,072, 786,432 and 262,144 BF16 output words, respectively, match the
retained independent original-FX Atlas-arithmetic references. SmolVLA is an
independently calibrated vision block; this is not a full-policy measurement.
The B=2 result is 62,726,125 cycles per request, a throughput metric rather
than single-request latency. Its calibration and source graph are unchanged.

SmolVLA's attention-score contraction drops from 65,596,164 to 49,974,852 cycles;
its attention-value contraction drops from 30,764,024 to 16,997,506 cycles.
Each saves 18,874,368 bytes of reads. Its final MLP projection saves another
589,108 cycles: a changed panel geometry adds 4,718,592 read bytes and removes
4,718,592 write bytes. Thus the ranking is a candidate heuristic; the complete
measured schedule and exact output checks remain the acceptance criteria.

TinyLlama B=1 removes 262,144 read bytes; the allocator change also saves
213,954 cycles in the down projection without changing its DRAM traffic.
B=2 removes 524,288 read bytes and improves several projection schedules through
shorter scalar lifetimes. Every changed region retains its matrix-product count;
every unchanged image retains identical cycles, traffic and instruction counters.

| Fixture | Instructions | DRAM read bytes | DRAM write bytes |
| --- | ---: | ---: | ---: |
| TinyLlama B=1 | 7,786,086 | 66,400,256 | 13,081,408 |
| SmolVLA vision B=1 | 77,654,613 | 322,148,352 | 137,661,024 |
| TinyLlama B=2 | 14,947,866 | 95,916,032 | 32,525,696 |

Runs use Rust `be1b521015ff54ec75699c2d5d63e567ab0ae22a`, CPU, RTL numerics,
`register_rows`, 64 MiB I/O and a 128,000,000-instruction execution budget.
Device cycles exclude host packing and are simulator measurements. The 128 KiB
IMEM and native verification limits remain unchanged.

Artifacts under `perf-studies/smolvla-dispatch` are `tinyllama-batch-panels`,
`application-batch-panels` and `tinyllama-batch2-panels`; runs are
`rust-tinyllama-batch-panels`, `rust-batch-panels` and
`rust-tinyllama-batch2-panels`. Each retains the exact-output comparison,
executed-traffic roofline and per-region panel-comparison receipts.
