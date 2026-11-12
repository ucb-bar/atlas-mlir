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

This is an **83.1% reduction in compiled launch count**, not a whole-layer
cycle measurement. Compilation and independent source replay passed in 139
seconds; the largest emitted image was 8,856 bytes. No
whole-layer output-parity or new whole-layer cycle result is claimed here.

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
