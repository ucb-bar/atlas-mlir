# Shape-selected MXU1 operand panels

Select `ATLAS_QUANTIZED_OPERAND_REUSE=panel-cache` to rank bounded activation
panels by weight-read and partial-result traffic. This extends `paired-cache`,
which retains its original two-row behavior. It does not change reduction
order, quantization, numerical policy, or the physical memory reservation.

The candidate uses 2, 4, 8, 16 or 32 matrix tile rows, dividing the padded M
extent. K panels have at most 64 tiles. All activation rows and one weight
column together occupy at most 192 KiB, leaving 48 KiB of the selected scratch
frame for restoration and ordinary allocation. The ranking accounts for
additional BF16 partial-result writes and reads when a wider M panel forces
smaller K panels. Candidates that cannot save traffic retain the existing
wide-row cache. Batch reuse is not part of this change.

The selected M/K geometry is recorded and rederived for native rebinding and
C export. Native allocation, initialization, timing, dynamic-instruction and
encoded-image checks remain mandatory. Sizing retries the old paired policy
once on an explicitly retained capacity refusal; unclassified compiler or
proof errors still propagate. In particular, a native pipeline failure is not
reinterpreted by matching an error string. The actual chosen policy and its
checked evidence identify the published program.

## Measured complete-layer result

The same held-out SmolVLA vision block, B=1, T=1024, width 768, improves from
**1,103,692,487 to 1,025,795,466 cycles**, saving **77,897,021 cycles (7.06%)**.
Dispatches remain 156. All 786,432 BF16 output words match the independent
original-FX Atlas-arithmetic reference. The only changed region cycle counts
are the six projections below.

| Projection shape M × N × K | Count | Old cycles each | Panel cycles each | M/K tile panel |
| --- | ---: | ---: | ---: | --- |
| 1024 × 768 × 768, Q/K/V/O | 4 | 19,401,532 | 11,440,720 | 4 / 38 |
| 1024 × 3072 × 768, MLP up | 1 | 75,653,692 | 43,810,192 | 4 / 38 |
| 1024 × 768 × 3072, MLP down | 1 | 42,792,453 | 28,582,180 | 8 / 21 |

Useful MXU1 utilization for the up projection rises from 3.12% to 5.39%, and
for down from 5.51% to 8.25%, counting logical MACs over cycles × 1024. These
are projection measurements, not whole-layer utilization or hardware results.
Divide by two when using both MXUs' combined peak as the denominator.

The unchanged TinyLlama B=1/T=64 decoder remains at **82,772,819 cycles and 59
dispatches**, with all 131,072 output words matching. Its small row count
already uses the same two-row geometry. Independent-request batching needs a
separate implementation and validation; concatenating sequence lengths would
not establish that result.

Both full layers use the pinned Rust model `be1b521015ff54ec75699c2d5d63e567ab0ae22a`,
RTL numerics and `pack_layout=register_rows`. SmolVLA uses the real calibrated
vision-block fixture described in [dispatch-reduction.md](dispatch-reduction.md).
Integration artifacts are `application-panel-geometry`, `rust-panel-geometry`,
`tinyllama-epilogue-panels` and `rust-tinyllama-epilogue-panels` under
`perf-studies/smolvla-dispatch`.

Tests cover tile tails, nonuniform row scales, complete-DRAM equality against
the previous emitter, late rejected exponents, unchanged surrounding memory,
unprofitable/unsupported geometry, tampered schedules, native rebinding,
generated C export, actual instruction-image capacity refusal, and visible
compiler failures. The compiler proof work limit was not raised; see
[initialization-proof-slicing.md](initialization-proof-slicing.md).
