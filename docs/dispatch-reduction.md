# Exact-float dispatch reduction

These changes extend Atlas PR #50 at `c1ba9cc02de190a9fe6ccab6dc7202d78518ab40`
and Merlin PR #63 at `5503777c8434a03c3a5b6c5751668f6ed8c2fed5`.
Both companion branches are named `perf/model-dispatch-reduction`. Merlin's
branch starts from `reed-nicolas/merlin:feat/tinyllama`; it does not modify that
fork branch.

## Changes

BF16 add/sub checks the qualified vector domain before checking exceptional
arithmetic. Passing the stronger domain check also proves validity, avoiding
redundant scalar work. Other inputs retain the original checked pair or scalar
routes. All input tiles must pass admission before output publication.

Merlin exposes `--execution-instruction-limit` through compile, verification,
and profiling. Atlas uses it for proved instruction bounds and exact-float
slice sizing. It is included in materialization, contraction admission, and
tuning identities. The transformation depends on operation semantics, tensor
sizes, and resource limits; it has no model-name or layer-number rules.

The selected RTL has **128 KiB IMEM: 32,768 four-byte instructions**. The
existing generated-stream verifier already enforces that capacity. A loop's
dynamic instruction count is separate. The default execution budget remains
1,000,000; the larger-dispatch layer compilation used:

```text
--execution-instruction-limit 128000000 --io-memory-bytes 67108864
```

Use the same execution budget when verifying the package. A simulator watchdog
must independently allow the admitted execution. See
[exact-float-device.md](exact-float-device.md) for RTL provenance, slice limits,
and the existing functional runner's option.

## Measurements

Single-layer TinyLlama source request:
`57ad325b7b56f9d50e8429654f02b04c31a930eb0895364d223dbbd9e3c4783a`.
The retained default package and the fresh larger-dispatch package have the
same source graph and 90 selected regions. Both include all operations in the
decoder layer.

| Planned NPU launches | Default package | Larger dispatches |
| --- | ---: | ---: |
| Exact-float regions | 503 | 16 |
| Other regions | 83 | 83 |
| Total | 586 | 99 |

This is an **83.1% reduction in compiled launch count**. Compilation and
independent source replay passed in 139 seconds; the largest emitted image
was 8,856 bytes. The full-layer Rust measurements below subsequently verified
output parity and device cycles. Keeping existing operand-cache and unary-unroll
options enabled matters: larger dispatches alone do not imply fewer device cycles.

The kernel measurements below use `npu_model_rs` commit `0186e37`, CPU compute,
RTL arithmetic, the default curve memory model, and idle-cycle skipping. Inputs
are uniform BF16 `1.0` and `0.5`; outputs match the independent host C reference
and input bytes remain unchanged. The baseline uses PR #50's original float
lowering, compiled with the same tools as the new lowering. Larger baseline
totals are repetitions of the measured 1,024-element baseline dispatch.

| BF16 add elements | Baseline cycles | New cycles | Baseline launches | New launches |
| --- | ---: | ---: | ---: | ---: |
| 1,024 | 145,389 | 67,399 | 1 | 1 |
| 16,384 | 2,326,224 | 1,119,879 | 16 | 1 |
| 131,072 | 18,609,792 | 8,953,607 | 128 | 1 |

These represent roughly 52–54% fewer device cycles on the qualified input path.
They exclude host dispatch and program upload time and are not a prediction
for every input distribution. The 131,072-element image is 4,452 bytes, with a
proved worst-case bound of 71,987,972 executed instructions under the selected
128,000,000 budget. Its Rust simulation took 0.247 seconds on this machine.
Fallback inputs can execute more instructions; their exact behavior and late
failure publication are covered by directed runtime tests.

## Validation

- Atlas: 33 focused instruction-bound, arithmetic, slicing, export, and replay
  tests passed, including rejection under a smaller execution or image budget.
- Generated C dispatch through the existing NPU functional runtime passed for
  all six exact-float operations, and separately for larger add/sub slices.
- Rust: four benchmark runs and four mixed-route/late-invalid add/sub runs
  passed exact output, rejection-before-publication, and immutable-input checks.
- Merlin: 34 code-generation and synthetic-target tests passed; touched files
  pass Ruff. Structure, documentation, artifact-layout, target-name, and
  assumed-constant checks passed.

Merlin's repository-wide regex check still reports three existing
`re.fullmatch` calls in `compile/baremetal_harness.py` (also present on the
unmodified PR head). This branch does not change that file or its allowlist.

## Full-layer Rust execution and private cast folding

The same fixed 64-token layer has now run through generated host C and the Rust
NPU simulator. All 131,072 BF16 output words match an independent replay of the
original FX graph using the pinned Atlas arithmetic. The source checkpoint and
quantized weight bytes are unchanged. These are device cycle measurements;
they exclude modeled host preparation, staging, uploads and polling.

| Configuration | Cycles | Dispatches |
| --- | ---: | ---: |
| Retained 1M-budget package (earlier cache/unroll settings) | 226,725,080 | 586 |
| 128M budget, compact unary kernels, ordered operands | 301,262,434 | 99 |
| 128M budget, unary unroll, binary lane groups, wide activation cache | 172,623,943 | 99 |
| Same options, plus pointwise and quantized narrowing folds | 152,606,339 | 95 |

The last comparison isolates removing four private casts: **20,017,604 fewer
cycles (11.60%)**. Three are new pointwise folds (two norm outputs and the MLP
division output); the fourth uses the existing quantized P.V narrowing fold.
The total improvement versus the retained package is 32.69%, but that comparison
combines dispatch policy, caches and arithmetic-lowering changes. The 301M row
illustrates why we do not attribute the combined improvement to dispatch size.

Rust simulation took about 5 seconds for the last row; running generated C,
callbacks, per-launch state files and subprocesses took about 6 seconds. Timings
are this machine's wall time, not hardware inference latency. Native compilation
and uncached source replay of the 86-region package took 147 seconds.

### Compiler/simulator ABI

Compiler PR #50 targets the Python model at
`47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`. Rust port `0186e37` follows `c119e54`,
which changed PACK/UNPACK lane ordering. The original Rust configuration produced
incorrect layer outputs despite self-consistent timings. An explicit compatibility
option in Rust commit `be1b521015ff54ec75699c2d5d63e567ab0ae22a` resolves that mismatch:

```json
{"numerics": "rtl", "pack_layout": "register_rows", "compute": "auto"}
```

The Rust default remains `tile_rows`. Select the layout matching the compiler's
model revision; neither setting is a claim of equivalence to another RTL revision.
The corrected output SHA256 is
`9de10cb80864257a8b300b533eaafb0685d2f4f040d39010d6e1b4f7c29f7bd9`.
The FX graph file SHA256 is
`394740ce5dfe2a14e715824ed9383e89ee71ac41edf24b9f8a2ee89b4dc8224b`.

### Enabling only the narrowing folds

```python
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
adapter = AdjacentDeviceAdapter(
    adapter, patterns=("pointwise_narrowing", "quantized_narrowing")
)
```

Use `ATLAS_QUANTIZED_OPERAND_REUSE=wide-cache`,
`ATLAS_EXACT_BINARY_LANE_GROUPS=1`, `exact_float_unroll=True`, and the execution
budget shown above to reproduce the selected optimization settings. The default
adapter is unchanged. Passing no `patterns` argument to `AdjacentDeviceAdapter`
retains all its supported composition patterns; an explicit subset permits
independent evaluation. Unknown or repeated pattern names are rejected.

The new `pointwise_narrowing` pattern accepts an adjacent private FP32 result
followed by the original dense `f32-to-bf16-rne-v1` cast. Supported selected
operators are bounded BF16 multiplication, addition, subtraction, division and
rsqrt. Their original precision policies, input admission, physical kernels,
rounding and padding remain unchanged. The non-NaN BF16 result would have been
widened exactly to FP32 and narrowed back to the same bits. The mixed-width C ABI
publishes those bits directly, using FP32 input spans and BF16 output spans in
all overlap checks. An escaping intermediate, different cast or unproved result
policy keeps the original smaller regions. No floating-point arithmetic is moved
to the host and no model name is used for recognition.

Four focused tests cover selection/refusal, every non-NaN BF16 encoding's exact
round trip, and generated C through Rust for multiplication/division with padding,
signed zero, late invalid input, immutable inputs, output sentinels and alias
rejection. The related composition/materialization suite passed 39 tests and
17 subtests, with three optional runtime gates skipped. Native source replay
and the complete layer reference comparison also passed.

## Make shared weight panels usable within the existing proof budget

The optional `paired-cache` schedule was already implemented, but full-width
MLP projections exhausted the source initialization verifier's work allowance.
The live-IR proof now discards dead scalar facts at each block entry using MLIR
liveness. Facts used later, including outer values and current block arguments,
remain available. At joins, initialized bytes still intersect across paths;
nonterminating scalar-state cycles and exhausted work still reject compilation.

The independent captured-CFG proof now stores one bit per aligned 32-byte DMA
beat instead of one tree node per initialized beat. Its exact coverage and
intersection rules are unchanged. This makes large output-panel snapshots cheap
enough for the unchanged 30-second tool timeout. With only the liveness change,
one gate-projection compile took 35.57 seconds and exceeded that timeout. With
the bitmap it took 2.26 seconds and emitted byte-identical machine IR. Neither
change increases the one-million-step proof allowance or any hardware limit.

Using the same narrowed layer and `ATLAS_QUANTIZED_OPERAND_REUSE=paired-cache`:

| | Wide activation cache | Shared weight panels |
| --- | ---: | ---: |
| Device cycles | 152,606,339 | 138,103,943 |
| Executed instructions | 31,408,505 | 31,530,988 |
| Dispatches | 95 | 95 |
| Largest instruction image | 24,832 bytes | 50,080 bytes |

This saves **14,502,396 cycles (9.50%)** despite slightly more instructions,
because it reduces weight traffic. The existing schedule's geometry restrictions
and fallback remain unchanged. Full-layer independent output parity passed again;
Rust simulation took 4.5 seconds, with 5.3 seconds through the generated C harness.
Compilation plus source replay took 157 seconds. Compared with the retained
586-dispatch package, the combined device-cycle reduction is 39.09%.

The added 512-iteration runtime-diamond regression fails the old verifier's work
budget and passes the new verifier; removing initialization on either branch still
fails. The memory/proof suite passed 32 tests and 119 subtests, and 12 native
contract/allocation tests passed. The selected full-width gate and down projections
both pass ordinary counted-image sizing and publication. No cache schedule or
arithmetic operation is reimplemented by these proof changes.

## Preliminary host-model comparison

A separate Rust build combines host-model commit `1b1242b` with the PACK ABI
compatibility fix. We recorded actual generated-C input writes, uploads, launches,
waits and output copies, then replayed them in one persistent Rust process.
For the 95-dispatch wide-cache/narrowing package, all 12,750,848 initialized
final device-memory bytes agree with the original C/standalone execution.

| Modeled host execution | Retained package | Wide cache plus narrowing |
| --- | ---: | ---: |
| Dispatches | 586 | 95 |
| NPU active cycles | 226,709,360 | 152,603,910 |
| Total elapsed cycles | 247,076,870 | 160,517,906 |
| NPU idle cycles | 20,367,510 | 7,913,996 |
| Upload cycles | 12,996,576 | 962,748 |
| Staging/copy cycles | 7,337,146 | 6,946,566 |
| Uploaded instruction bytes | 4,329,848 | 320,536 |

These are **uncalibrated host-model estimates**, with the model's default MMIO
and transfer costs, coherent memory, shared-link contention disabled, and host
arithmetic preparation set to zero. Input file writes charge destination traffic;
output copies charge both read and write traffic. There is no generated RV64/RVV
instruction execution. Consequently, this table is not a full end-to-end hardware
latency estimate. NPU active counts differ slightly from independent fresh-launch
runs because the host model uses its persistent restart/completion timeline.
The paired-cache layer has not been included in this host comparison.
