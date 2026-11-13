# Private activation boundary casts

The opt-in `pointwise_narrowing` pattern now covers the selected bounded
GELU and sigmoid policies, including an existing closed sigmoid/product SiLU
region. The new `pointwise_input_widening` pattern consumes the original BF16
input when its only consumer is a private exact widening into that region.
Enable both patterns to keep both boundaries in BF16.

Merlin's existing private-input and private-result proofs establish ownership,
dense identity maps and exact source casts. Atlas separately establishes that
the selected bounded arithmetic cannot produce NaN and that raw BF16 input
validation is equivalent to exact widening followed by the original encoder.
The native arithmetic, coefficients, LUTs, rounding and admitted domains stay
unchanged. No operator is moved across a contraction or reduction.

Raw transport refuses encoders that clamp, rescale, flush or canonicalize zero.
Escaping intermediates, changed cast semantics and unsupported boundaries keep
their existing implementations. The generated C checks the complete input
before device I/O, including the last element of a streamed tensor. Caller
output is published only after successful execution; actual input/output byte
widths are used for overlap checks.

## Complete-layer results

| Identical fixture and selected arithmetic | Before cycles | After cycles | Dispatches |
| --- | ---: | ---: | ---: |
| TinyLlama B=1, 64 tokens | 82,772,819 | 82,772,819 | 59 → 59 |
| TinyLlama B=2, two independent 64-token requests | 166,755,481 | 140,285,224 | 69 → 67 |
| SmolVLA vision block B=1, 1,024 tokens | 881,588,507 | 766,046,001 | 151 → 147 |

All 131,072 / 262,144 / 786,432 BF16 output words respectively match the
independent original-FX Atlas-arithmetic reference. Weights, inputs, masks,
positions and precision policies are unchanged. The B=1 TinyLlama capture's
older expanded SiLU already had its boundary casts folded, so it gains nothing
from this extension. The newer B=2 capture uses a closed sigmoid/product region.

The B=2 comparison retains [shared-weight panels](shared-weight-batches.md).
The same two requests run separately take 165,506,710 cycles and 118 dispatches;
the combined changes therefore save **15.24%** against that serial execution.
This is a throughput comparison, not a lower latency claim for each request.

SmolVLA removes 115,542,506 cycles (13.11%) around GELU: 36,050,166 from input
widening and 79,492,340 from output narrowing. GELU itself remains unchanged.
The [SmolVLA calibration scope and quality limitations](dispatch-reduction.md)
still apply: this is one independently calibrated real vision block, not a
full-policy quality evaluation.

Measurements use Rust `be1b521015ff54ec75699c2d5d63e567ab0ae22a`, RTL numerics,
`pack_layout=register_rows`, the optimized native pipeline and `panel-cache`.
The limits remain 64 MiB I/O, 128,000,000 executed instructions and 128 KiB IMEM.
These are simulation results, not silicon measurements.

## Validation and retained evidence

Directed C/Rust tests cover both output widths, multidimensional shapes,
partial tiles, exhaustive admitted GELU and SiLU input codes, sigmoid,
both SiLU source operand orders, signed zero, NaN/infinity/domain refusals,
escaping values, modified coefficients and disabled policies. A three-launch
streamed case checks each packed input chunk, its partial final chunk, every
output word, untouched device bytes and atomic refusal of a late invalid input.
Conversion-policy tests refuse clamping, flushing, rescaling and zero rewriting.
Native source replay and instruction/image limits remain mandatory.

Artifacts under `perf-studies/smolvla-dispatch` are
`tinyllama-pointwise-boundaries`, `tinyllama-batch2-pointwise-boundaries` and
`application-pointwise-boundaries`. Passing executions are
`rust-tinyllama-pointwise-boundaries`,
`rust-tinyllama-batch2-pointwise-boundaries-v2` / `-v3`, and
`rust-pointwise-boundaries` / `rust-pointwise-boundaries-traffic`.
Their comparison receipts bind source, execution and independent reference.

One preliminary batched run failed an immutable-memory check at an unchanged
projection before the new SiLU region. Its first 59 launches have identical
input/output memory hashes and all saved intermediate values match the passing
rerun. Two fresh runs of the same authenticated application passed, with and
without traffic capture. The initial failure is unexplained; this change does
not claim to fix it, and the failing memory check has not been relaxed.
