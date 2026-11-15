# Keep invariant exponent pages in VMEM

Set `ATLAS_EXPONENT_PAGE_CACHE=1` to load each scalar exponent page once per
invocation and each per-batch page once per batch. The default remains `0`.
Row-varying exponent pages retain their existing tile loads. This applies to
ordinary counted contractions, operand caches, bounded panels, shared-weight
batch flattening and fused quantized epilogues.

This is an Atlas scheduling transformation. The source tensor shapes determine
page lifetimes; no model names, observed exponent values or calibration samples
participate in selection. Merlin's graph and numerical contracts are unchanged.
The cache reuses the two existing restoration slots and adds no VMEM reservation.
Native allocation still checks the longer live ranges, initialized reads, buffer
dependencies, instruction bounds and the unchanged 128 KiB IMEM limit.

Each cached page is loaded before its first restoration. Per-batch pages are
refreshed inside the batch loop, before its M/N traversal. Rounded K accumulation,
exponent validation, normal-factor checks, scalar fallback and sticky failure
status are unchanged. Generated bindings publish outputs only after successful
device completion, including for fused epilogues.

Selection is recorded in execution and sizing records and included in tuning and
materialization identities. Rebinding and C export rederive page scopes from the
source and recorded selection, independently of the current environment. Old
records without the option retain their original program. Invalid settings and
modified scope records are rejected.

## Validation

Directed tests cover scalar, row and per-batch scales on either operand; partial
M/N/K tiles; multi-axis batches; shared and batched weights; inline and counted
panel restoration; and generated C with an atomic epilogue. Rust runs compare
all DRAM bytes with caching disabled, including distinct batch scales, scalar
fallback, last-page invalid scales and untouched inputs/guards. Small traced
cases check the exact number of eliminated DMA reads and unchanged stores.
Removing the prefetch must fail native initialized-read verification. Export
replay is checked with both old and enabled publications under a changed
compilation environment.

## Complete-layer results

| Identical fixture and arithmetic | Before cycles | After cycles | Dispatches |
| --- | ---: | ---: | ---: |
| TinyLlama B=1, 64 tokens | 76,325,780 | 74,571,452 | 56 → 56 |
| SmolVLA vision block B=1, 1,024 tokens | 566,685,538 | 520,987,059 | 113 → 113 |

The baseline includes [private layout packing](float-input-layout.md). The new
cache saves **1,754,328 cycles (2.30%)** on TinyLlama and **45,698,479 cycles
(8.06%)** on SmolVLA. The latter's attention-score contraction alone drops from
101,082,743 to 65,596,164 cycles. All **131,072** and **786,432** BF16 output words
match the independent original-FX Atlas-arithmetic references, respectively.
Weights, inputs, calibration, arithmetic and dispatch counts are unchanged.
SmolVLA is the independently calibrated vision block, not a full-policy result.

Executed traces confirm **3,129,344** and **67,545,088 bytes** fewer DRAM reads.
Each changed region has exactly the scale-page read reduction derived from its
source geometry, unchanged stores and unchanged MXU instructions. All other
images and executed counters remain identical. Final instruction counts are
7,931,181 and 77,844,979. Read/write totals are 66,662,400/13,081,408 bytes and
355,178,496/142,379,616 bytes. Same-work serial-launch roofline bounds are
54,902,252 and 379,019,098 cycles; these are not speedup predictions. Useful
MAC utilization is 3.71% and 1.66% of one MXU's peak.

Runs use Rust `be1b521015ff54ec75699c2d5d63e567ab0ae22a`, CPU, RTL numerics,
`register_rows`, 64 MiB I/O and a 128,000,000-instruction execution budget with
unchanged 128 KiB IMEM. Device cycles exclude host packing and are not silicon
measurements. Artifacts under `perf-studies/smolvla-dispatch` are
`tinyllama-exponent-page-cache`, `application-exponent-page-cache`,
`rust-tinyllama-exponent-page-cache` and `rust-exponent-page-cache`. Both runs
retain exact comparisons, roofline reports and per-region DMA-count receipts.

The focused suite passes 14 tests. Default-off regression coverage has 37 passes,
5 deliberately skipped Python-simulator tests and 2 passing subtests; one native
allocator crash at K=4097 passed on a focused replay. The two retained inputs
also produce byte-identical machine IR under release and diagnostic ASAN
builds. The diagnostic uses `allow_user_poisoning=0` for the mixed LLVM slab
allocator and is not full sanitizer qualification. The intermittent crash is
retained for investigation and is not claimed fixed by this feature.
