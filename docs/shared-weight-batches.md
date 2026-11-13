# Shared-weight panels across independent requests

The opt-in `panel-cache` policy can reuse a weight panel across an existing
batch of independent projections. It folds the physical tile-row traversal
from B × M to BM only when M is a multiple of 32, the RHS is one unbatched
matrix, and activation exponents are scalar or a contiguous row-page stream.
Per-batch scalar exponents, partial M tiles and batched attention RHS tensors
keep their existing traversal. The same 192-KiB operand reservation and all
native resource checks apply.

This is an Atlas physical packing and traversal decision. The source graph,
input/output shapes, weight bytes, K order, rounding, scale interpretation and
C ABI stay unchanged. Merlin's source batch is not rewritten, and this change
does not introduce a runtime request queue or concatenate attention sequences.
The flattened batch count is recorded in the execution layout and independently
rederived during source replay. A malformed record refuses compilation/export.

The existing single-status ABI conservatively refuses the whole batch after
an invalid device-side exponent; no caller output is published. It does not
provide per-request failure isolation or an automatic retry policy.

## Complete decoder comparison

Two distinct original 64-token input fixtures were embedded with unchanged
TinyLlama weights. Each request retains its own causal mask and original
positions 0–63. The reference executes the original B=1 FX graph independently
for each request and concatenates the resulting outputs only for comparison.

| Same two requests | Cycles | Dispatches |
| --- | ---: | ---: |
| Two separately executed B=1 decoders | 165,506,710 | 118 |
| Existing B=2 compilation, before shared panels | 242,170,792 | 69 |
| B=2 compilation with shared panels | 166,755,481 | 69 |

Shared panels save **75,415,311 cycles (31.14%)** against the same B=2 graph.
This version is still **0.75% slower than two serial B=1 runs**; it is not yet
an end-to-end throughput win over that baseline. The newer captured B=2 graph
retains private casts around its selected sigmoid/product SiLU region that
the earlier B=1 expansion already fuses. Those boundaries are a separate
optimization opportunity, not savings attributed to this change.

The subsequent [activation boundary folding](pointwise-boundary-folding.md)
removes those casts and reaches 140,285,224 cycles and 67 dispatches. That
combined result is 15.24% faster than the same two requests run separately.

All **262,144 BF16 output words** match both independent request references.
The first request also matches the previous B=1 reference. The second request's
separate execution measures 82,733,891 cycles; the first measures 82,772,819.
The batched output SHA256 is
`f9d6d6bc84cbe9c392a84c7e4f2e2680ccbb0f3c9bf6a6664babe711d230e332`.
Both complete B=1 regressions are unchanged and exact: TinyLlama 82,772,819
cycles/59 dispatches; SmolVLA vision block 881,588,507 cycles/151 dispatches.

These runs use the pinned Rust model `be1b521015ff54ec75699c2d5d63e567ab0ae22a`,
RTL numerics, register-row PACK layout, the optimized native pipeline,
64-MiB I/O, a 128,000,000-instruction execution budget and 128-KiB IMEM.
The independently captured B=2 source model SHA256 is
`5895e25074dbcd6a8a4926a836b4a80de72c2ada8aa0e680216c35417e189b75`.

Retained artifacts under `perf-studies/smolvla-dispatch` include
`tinyllama-batch2-reference-source`, `reference-tinyllama-batch2`,
`tinyllama-batch2-baseline`, `tinyllama-batch2-shared-panels` and their
`rust-` runs. The serial second-request execution is
`rust-tinyllama-second-serial`. Full B=1 regression artifacts are
`tinyllama-shared-panels`/`rust-tinyllama-shared-panels` and
`application-shared-panels`/`rust-shared-panels`.

Directed tests cover B=2/4/8, multidimensional batches, differing row scales,
K/N tails, declined M tails and batched RHS matrices, tampered execution
layouts, complete-DRAM equality, late invalid scales, native binding replay,
generated C publication and unchanged output on device rejection. B=4/8 are
kernel checks here, not complete-model measurements.
