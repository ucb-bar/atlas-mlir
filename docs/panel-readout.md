# Keep completed panel tiles on chip

Operand-panel matmuls previously wrote a completed BF16 tile to DRAM and loaded
it again to restore its exponents and apply an optional product epilogue. Small
panel bodies now pass that final MXU readout directly to the existing restoration
code and store the completed result once. Intermediate K panels still spill
their BF16 partial sums in the original order; accumulation rounding, exponent
admission, epilogue arithmetic and atomic host publication are unchanged.

This is Atlas scheduling and storage placement. It reuses Merlin's existing
semantic proofs and does not add a graph rewrite or model-specific matcher.
Both `paired-cache` and `panel-cache` use the optimization where admitted.

Inlining is bounded to four M tiles and 512 statically emitted K-tile products
per panel body. These are code-growth heuristics, not resource proofs. Larger
bodies retain the original counted restoration loop. Native allocation, address
and timing verification, the selected execution budget and the 128 KiB IMEM
limit remain mandatory. Unbounded inlining exceeded the existing compiler time
limit on larger panels; the bounded implementation preserves those cases.

The restoration emitter takes explicit input/result names and local prefixes.
It emits typed operations directly. Multi-guard bodies publish a v2 guard
receipt listing every actual factor loop; single-guard bodies retain v1.

## Complete-layer results

| Identical fixture and arithmetic | Before cycles | After cycles | Dispatches |
| --- | ---: | ---: | ---: |
| TinyLlama B=1, 64 tokens | 81,247,163 | 77,504,366 | 59 → 59 |
| SmolVLA vision block B=1, 1,024 tokens | 766,023,249 | 745,535,834 | 147 → 147 |

The matched baseline includes [scratch output priority](scratch-output-priority.md).
Savings are 3,742,797 cycles (4.61%) and 20,487,415 cycles (2.67%). Every one of
the 131,072 and 786,432 BF16 output words matches the independent original-FX
Atlas-arithmetic reference. Weights, inputs and numerical policies are unchanged.

Two independent 64-token TinyLlama requests also pass all 262,144 output words
at **137,340,799 cycles / 67 dispatches**. The preceding published batched result
was 140,285,224 / 67; that comparison includes both the scheduler change and
this readout change, rather than isolating readout alone.

All measurements use Rust `be1b521015ff54ec75699c2d5d63e567ab0ae22a`, CPU,
RTL numerics and `register_rows`, with 64 MiB I/O and a 128,000,000-instruction
execution budget. They are simulated device cycles, not host or silicon timings.
The [independently calibrated SmolVLA vision-block scope](dispatch-reduction.md)
and its distinction from framework-f32 arithmetic still apply.

## Validation

Directed Rust tests compare every DRAM byte against the existing wide-cache
schedule, including partial M/N/K tiles, nonuniform row scales, multiple K
panels, independent batches, both product-epilogue operand orders and late
invalid exponents. Actual executed DMA counts confirm that only intermediate
partials are reloaded. Guard receipts are checked against emitted labels.
Tests exercise both sides of the inline code-growth boundary and the existing
native capacity fallback. Generated C, canonical rebinding and atomic output
refusal are included.

The default restoration fragments remain byte-identical in 24 scalar/vector,
shape and scale configurations. The final bounded policy reproduces all 17
quantized source bodies and guard receipts used by the two complete B=1 layer
runs; the batched run was compiled with the final policy directly.

Retained applications under `perf-studies/smolvla-dispatch` are
`tinyllama-inline-readout-v2`, `application-inline-readout-v2` and
`tinyllama-batch2-inline-readout-v2`. Passing comparison receipts are in
`rust-tinyllama-inline-readout-v2`, `rust-inline-readout-v2` and
`rust-tinyllama-batch2-inline-readout-v2`. The release-worktree source replay is
`out/dispatch-audit/replay-panel-sources.json`.
