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

## Vector preflight for qualified binary tiles

`ATLAS_EXACT_BINARY_VECTOR_PREFLIGHT=1` adds a sufficient-domain test before
BF16 add/sub's existing scalar input classification. VPU min/max reductions
across both absolute operands prove that every lane is nonzero and has magnitude
between `2**-32` and `2**32`. A successful proof selects the unchanged qualified
arithmetic. Zeros, subnormals, NaNs and other values outside that interval use the
existing classification and fallback. The whole-input validation barrier still
precedes every output store. The option defaults to off and participates in the
compilation/materialization identity.

| Paired-cache layer | Scalar preflight | Vector preflight |
| --- | ---: | ---: |
| Device cycles | 138,103,943 | 125,179,302 |
| Executed instructions | 31,530,988 | 18,251,147 |
| Dispatches | 95 | 95 |

This saves **12,924,641 cycles (9.36%)** on the fixed layer, and all 131,072
output words remain identical to the independent FX reference. Together with the
previous changes, cycles are 44.79% below the retained 226,725,080-cycle package.
The fresh native compilation/source replay took 161 seconds; Rust instruction
simulation took 4.2 seconds and the generated-C run took 5.0 seconds.

On uniform BF16 `1.0 + 0.5`, a 1,024-element kernel drops from 50,295 to 11,047
cycles; a 131,072-element kernel drops from 6,761,628 to 1,737,884. The latter
image is 6,552 bytes, with a proved worst-case instruction bound of 77,408,404.
Fallback tiles pay for the additional vector test; these are measured improvements
for the stated inputs, not universal speedups.

Validation exhaustively checks the stronger-domain predicate for all 65,536
BF16 encodings against the pinned arithmetic oracle. Directed Rust execution
covers add/sub, short padded tails, mixed fast/fallback tiles, signed zero,
subnormals, domain endpoints, infinities, overflow and late NaNs in either
operand, with immutable-input and rejection-before-publication checks. The
focused gate passed 14 tests and six subtests; the related arithmetic/application
regression gate passed 27 tests with two optional runtime checks skipped.

## Exact vector admission for BF16 widening

`ATLAS_EXACT_CAST_VECTOR_PREFLIGHT=1` replaces the per-lane scalar NaN scan in
`bf16_to_f32` with an absolute-maximum VPU reduction. Under the selected ordered-bit
min/max arithmetic, the reduced magnitude exceeds the infinity encoding exactly
when an input is NaN. The conversion still shifts the original bits; subnormals,
infinities and signed zeros retain their exact FP32 representations. Whole-input
validation still finishes before any output store. Other casts are unchanged.
The option defaults to off and is included in compilation identities.

With the preceding binary vector check and the same paired-cache/narrowing options,
the full layer drops from **125,179,302 to 119,843,302 cycles** (5,336,000 fewer,
4.26%). Executed instructions drop from 18,251,147 to 13,770,379; dispatches remain
95. Every output word matches the independent FX reference. Compilation and source
replay took 150 seconds, Rust simulation 3.9 seconds, and the generated-C run
4.7 seconds. The combined cycle reduction versus the retained package is 47.14%.

The arithmetic predicate is checked over all BF16 encodings. Rust executes every
non-NaN encoding, shuffled across tiles, plus padded tails and late positive and
negative NaNs with different payloads. Both scalar and vector versions match the
independent C reference, preserve inputs and refuse invalid output publication.
The focused gate passes five tests and ten subtests; 22 related regression tests
pass with two optional runtime gates skipped.

A separate all-pattern fusion experiment is exact but slower: 135,377,274 cycles
and 87 dispatches before the widening change. Some larger groups reintroduce
private casts or prevent the faster paired operand-cache schedule. It is not the
recommended configuration; reducing launch count alone is insufficient.

## Preserve narrowing when selecting a longer pointwise chain

`adjacent_pointwise_chain` now also recognizes three selected pointwise
operations followed by a private exact FP32-to-BF16 cast. It reuses the single
pointwise narrowing proof, preserves every intermediate admission/canonicalization
step, and publishes the final BF16 result directly. A cast with different
rounding, layout, input, or escaping FP32 result is refused. The generated mixed
input/output ABI retains its byte-range and alias checks.

Directed generated-C execution through Rust covers rsqrt and division chains,
compact column broadcasts, padded tails, signed zero, late invalid outer inputs,
internal denominator rejection, immutable input bytes and atomic output
publication. Three focused tests and two subtests pass; 26 composition/narrowing
regression tests and six subtests pass, with one optional model gate skipped.

For the full layer, select `adjacent_square_reduce`, `adjacent_reduction_scale`,
`adjacent_pointwise_chain`, `pointwise_narrowing`, and `quantized_narrowing` in the
adapter, with the preceding cache, unroll and vector-preflight options. This
compiles 78 regions and runs exactly in **87 dispatches and 120,664,430 cycles**.
It saves eight dispatches versus narrowing alone, with a **0.69% device-cycle
increase** from internal device boundary checks. Those checks remain necessary;
the tradeoff is explicit rather than treating every larger region as faster.
The uncached compile/source replay takes 153 seconds; Rust simulation takes
3.9 seconds. Quantized epilogues are excluded from this configuration because
their current schedule disables paired operand reuse.

## Use the available operand-panel capacity

The opt-in `paired-cache` schedule now keeps up to 64 K tiles per panel instead
of 32. Two activation rows plus one weight column occupy at most 192 KiB,
leaving 48 KiB of the selected 240 KiB frame for restoration and allocation.
The existing minimum reduction size remains 32 tiles; shape eligibility,
per-output K order, BF16 rounding, final restoration and native resource proofs
are unchanged. Fewer panel boundaries mean fewer partial-result stores/reloads.

| Projection shape (M, N, K) | 32-tile panels | 64-tile panels |
| --- | ---: | ---: |
| (64, 5632, 2048) | 16,285,613 cycles | 15,204,783 cycles |
| (64, 2048, 5632) | 15,302,709 cycles | 14,123,595 cycles |

The complete 87-dispatch layer drops from **120,664,430 to 116,439,296 cycles**,
saving 4,225,134 cycles (3.50%), with every output word still exact. Instructions
fall from 14,538,797 to 14,487,786. The largest image shrinks from 50,080 to
49,096 bytes. Compilation/source replay takes 155 seconds; Rust simulation takes
3.9 seconds and the generated-C run takes 4.6 seconds.

New Rust gates compare both panel sizes over K=1024/1025/2049, partial M/N tiles,
multiple row pairs and late rejected exponents. They verify whole-memory equality,
protected bytes, status, the ordinary instruction bound and the 128-KiB image
limit. The focused suite passes three tests and three subtests, with two legacy
Python-model gates skipped. The full-width kernel probes and complete independent
FX comparison also pass.

## Keep operand reuse when fusing a projection epilogue

Paired panels now admit the already supported dense product epilogue. The existing
restoration hook applies that product once, after the complete ordered reduction
and exponent restoration. No multiplication moves across K accumulation, and the
additional operand stays outside the writable output/status span. This removes
the earlier incompatibility between paired caching and epilogue fusion.

Adding `quantized_epilogue` to the preceding pattern set reduces the layer from
**116,439,296 cycles / 87 dispatches to 114,955,518 cycles / 85 dispatches**.
The two folded products follow the output projection and MLP up projection.
Every output word still matches the independent FX reference. Compilation/source
replay takes 159 seconds; Rust simulation takes 3.8 seconds, with 4.5 seconds for
the generated-C run.

Eight tests and two subtests pass, including Rust comparisons with the previous
wide-cache epilogue for BF16/FP32 outputs, both multiplication operand orders,
partial tiles, multiple row pairs, signed-zero operands, late rejected exponents,
immutable extra operands and whole-memory equality. Both routes retain ordinary
native instruction/image proofs. The complete generated-C layer also passes all
per-launch memory permissions and immutable-weight checks.

## Avoid scanning padding at composed boundaries

Device boundary checks now visit only the rows that can contain logical values.
Column-to-row transitions inspect the packed logical column, and column broadcasts
validate each source value once before copying its bits across the destination
row. Destinations are still initialized to admitted padding; logical tails,
negative-zero canonicalization, sticky rejection and atomic publication retain
their original behavior. No intermediate input guard is removed.

Directed small-column probes improve from 148,671 to 12,671 cycles for identity,
145,629 to 12,829 for column-to-row, and 139,485 to 15,511 for broadcast. On the
complete layer, the improvement is smaller: **114,955,518 to 114,427,634 cycles**,
with **85 dispatches** and all 131,072 output words exact. Instructions fall from
14,468,548 to 13,944,764. Compilation/source replay takes 161 seconds and Rust
simulation takes 3.8 seconds.

The directed Rust suite passes 33 tests and 11 subtests, with one legacy Python
model gate skipped. Thirteen composition regressions and two subtests also pass,
with one optional gate skipped. Coverage includes all three layouts, partial and
multiple pages, non-power-of-two row counts, signed zero, raw NaN payloads when
admitted, rejected logical values versus padding, immutable bytes, and sticky
status across composed transitions.

## Keep an expanded SiLU inside its native loop

The opt-in `expanded_silu` pattern recognizes a closed BF16 input → exact FP32
widen → sign flip → selected EXP → add bound literal one → selected division →
exact BF16 narrow graph. It uses the existing five-operation native SiLU body:
zero subtraction, EXP LUT, add, reciprocal LUT and multiply. The selected division
already uses that reciprocal/product sequence. No IEEE source division is
reassociated and no additional precision policy is enabled.

All original operation recognizers, dense maps, source-bound literal bytes and
private intermediate boundaries are checked. The external guard is the exact
intersection of the original EXP and numerator domains. Existing exhaustive
EXP-domain proof covers the removed intermediate guards. Sign flip and zero
subtraction differ at signed zero, where both EXP results are one; the final
product preserves the original input sign. The recognizer declines changed
policies, constants, edges, maps, rounding or escaping intermediates. It uses no
model names, layer numbers or fixed dimensions.

With this pattern added, the unchanged layer runs in **100,447,374 cycles and
80 dispatches**, versus **114,427,634 and 85**: **13,980,260 cycles saved (12.22%)**.
The activation path itself falls to 1,206,701 cycles. All 131,072 output words still
match the independent FX reference. Compilation/source replay takes 155 seconds;
Rust simulation takes 3.6 seconds, and the generated-C run takes 4.3 seconds.

Four focused tests and 19 subtests pass. They exhaust all 65,536 BF16 encodings
for admission equivalence, compare every admitted code against the original
selected expansion, and run all 17,412 admitted values through generated C and
Rust with partial tiles, immutable inputs and late invalid-value rejection.
Thirteen related tests and 20 subtests pass; two optional legacy-model gates are
skipped. Source replay and ordinary native instruction/image proofs remain active.
