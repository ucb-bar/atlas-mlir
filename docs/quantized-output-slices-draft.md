# Unfinished BF16 output slicing experiment

This branch preserves work in progress from `perf/quantized-model-efficiency`
at `7c44a727`. It is not a validated optimization or a release candidate.

The experiment introduces `atlas.virtual_dma_store_bf16_slice` to write a
32-byte-aligned prefix of either physical BF16 register half. Its explicit
`tile_byte_offset` is 0 or 1024, and its size is at most 1024 bytes. A distinct
operation is necessary: an older compiler could ignore a new attribute on the
existing store operation and silently store the wrong half.

Lowering, source-derived tile/CFG contracts, allocation, memory effects and
virtual scheduling recognize the operation. The CFG checker checks the exact
source half even if an instruction and its tile record are changed together.
Full-tile transforms reject source-offset stores. The contraction generator has
an explicit `execution_layout(..., output_halves=True)` experiment; its default
remains false. It preserves the 2-KiB tile stride and leaves the middle gap
untouched.

Work still required:

1. Finish and review native source/contract tests. The earlier attribute-only
   experiment passed 76 tests and 900 subtests with three optional skips, but
   failed three new source-contract cases. The distinct-operation draft changes
   the schema field count; its final tests have not been run.
2. Execute both source halves in both pipelines against independent exact byte
   expectations, including guard memory, source overlap and malformed offsets.
3. Measure real contractions and late numerical refusal for small and near-full
   tiles. The earlier DMA-only probe shows that serial splitting can regress
   near-full tiles; choose a policy only from generated-program measurements.
4. Complete sizing records, publication verification and old-package replay
   before enabling automatic selection. The experimental layout is not yet
   propagated through all sizing/replay entry points.
5. Verify generated C output publication and every untouched gap. Keep the
   existing execution, image-size and compile-work limits.

Build logs, the superseded attribute patch, and partial test logs are retained
locally under `out/efficiency/resume-nov22/store-*`. No performance claim is made
for this draft.
