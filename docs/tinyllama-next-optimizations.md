# TinyLlama follow-up optimizations

Branch `perf/tinyllama-next-optimizations` starts at Nicolas's Atlas TinyLlama
head `7e00f41d9794198c35dec01de565471d556b8be6`. Validation uses the Merlin PR #63
head `5503777c8434a03c3a5b6c5751668f6ed8c2fed5` in a separate, detached worktree;
it does not require writing to Nicolas's fork.

## Binary validation and classification

BF16 add/sub keep their exact validation, qualified-vector, pair-reduction and
exceptional-scalar arithmetic routes. Validation and classification each process
four adjacent lanes per counted iteration, sharing scalar constants and address
calculation within the straight-line group. The constants do not escape that
block. A single-tile leaf retains the validated immutable operands in private
VMEM for computation, removing their second DMA load. Multi-tile programs keep
loading each computation tile separately. Validation still completes before any
result is written; a rejected input preserves the original output.

Select `ATLAS_EXACT_BINARY_LANE_GROUPS=1` to enable both changes; default binary
emission is unchanged. Unary emission and its explicit
`exact_float_unroll` option are unchanged. Physical image and instruction bounds
remain mandatory; no limits have been raised.

A freshly built compiler and the unchanged NPU cycle model at
`47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` measured a 1,024-element BF16 add
(1.0 + 0.5, RTL numerics, randomized state and poisoned VMEM):

| | TinyLlama base | Lane groups and retained inputs |
| --- | ---: | ---: |
| Cycles | 145,490 | 113,555 |
| Executed instructions | 132,554 | 103,883 |
| Program bytes | 4,256 | 5,584 |

The measured kernel improvement is 21.95%; this is not a whole-layer speedup.
Both programs matched the independent host reference and preserved protected
DRAM. The larger image is the explicit tradeoff. Local reproduction and logs
are not included in the source commits.

Twenty focused tests passed, including generated C dispatch on the functional
model, independent directed/random arithmetic comparisons, all unary encoding
regressions, group boundaries, multi-tile execution, shared operands, NaNs and
late-failure atomic publication. Run with the pinned model enabled:

```
ATLAS_RUN_NPU_MODEL=1 python -m pytest -q \
  test/test_exact_float_device.py test/test_exact_float_application.py \
  test/test_exact_float_binary_groups.py test/test_exact_float_unary_tiles.py
```

The same binary implementation also matched the retained independent outputs for
all 400 BF16-add states in the recorded TinyLlama layer. A cycle replay using an
actual layer input gave the same 145,490 to 113,555-cycle improvement above.

## Bounded activation prefix reuse

Select `ATLAS_QUANTIZED_OPERAND_REUSE=prefix-cache` to reuse the first 64 K32 tiles
of a wide row across output columns, streaming the remaining tiles through the
existing ordered loop. The cache remains capped at 64 KiB. Existing ordered and
full-row-cache virtual programs were byte-identical in 52 compared configurations.
See [the transport contract](quantized-operand-reuse.md#bounded-prefixes-for-wider-reductions).

For an independent signed-input fixture with M=1, N=65, K=5632:

| | Ordered | Prefix cache |
| --- | ---: | ---: |
| Cycles | 844,963 | 751,701 |
| Bytes loaded | 1,093,632 | 962,560 |
| Bytes stored | 8,192 | 8,192 |
| Program bytes | 3,408 | 11,224 |

Both complete output memories matched. This fixture improves by 11.04%; it is
not the full TinyLlama down projection. Three new tests passed (including cycle
execution across four geometries and native export), plus 29 regression tests
and four subtests. The separate optional qualified-native wide functional test
was not selected; the new cycle tests execute the unchanged model directly.

The full-size down projection was subsequently replayed with the recorded layer's
actual operands and scales: M=64, N=2048, K=5632. Both images were freshly built
from this branch and executed on the same pinned cycle model above.

| | Ordered | Prefix cache |
| --- | ---: | ---: |
| Cycles | 35,804,760 | 29,631,705 |
| DRAM bytes loaded + stored | 46,925,824 | 38,668,288 |
| DRAM service cycles | 31,771,037 | 26,180,285 |
| Program bytes | 3,476 | 11,040 |

This saves **6,173,055 cycles (17.24%)** and **8,257,536 DRAM bytes** in that
projection. Every retained reference read matched, all protected bytes were
checked independently, and the complete final memory hashes were identical.
The initial memory SHA256 was
`f67cfb760a68838810c3456ac22eaa59ebe44281d7f4bcc2ea04ec0cbdcbf29c`.
The local replay driver and complete roofline results are
not included in the source commits.
DRAM service still dominates; the next reuse work should reduce the remaining
weight/activation traffic before trying to increase MXU arithmetic throughput.
This is a projection measurement, not a newly validated whole-layer total.

## Publish quantized BF16 through an exact narrowing cast

The existing opt-in `AdjacentDeviceAdapter` now recognizes a quantized
contraction with an FP32 result consumed only by an adjacent exact
`f32_to_bf16` cast. Successful Atlas restoration already produces finite BF16
values, widened exactly for FP32 publication. Publishing the original BF16
bits directly removes that widening and subsequent narrowing without changing
the NPU program, ordered K arithmetic, scale admission or sticky rejection status.

```python
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter

# Wrap the existing AtlasApplicationAdapter passed to application compilation.
adapter = AdjacentDeviceAdapter(adapter)
```

Recognition requires the original cast policy and dense identity indexing,
matching shapes, unchanged code/exponent inputs, and a private intermediate.
An escaping FP32 result, another rounding policy, or a host consumer retains
the separate operations. Only a successfully sized counted contraction is
offered; overlarge images retain the original alternatives. Native tool errors
continue to propagate. Output publication remains atomic after device admission.
The fused group is not opted into the application's materialization cache.

The recorded TinyLlama layer's `op112r0` P.V and `op114r0_s0` narrowing pair
selects and compiles through this adapter. For its `[1,32,64,64]` result, the
fold removes eight cast launches at the current default chunk size and avoids
publishing a 512 KiB FP32 intermediate. The final dense BF16 output is 256 KiB.
The generated virtual device program is identical to the original contraction.
No additional cycle speedup is claimed for this fold until a complete execution
comparison is measured. Its local compilation receipt is
not included in the source commits.

`test_quantized_narrowing.py` covers real adapter selection, escaping results,
changed policies/ports, sizing refusal and tool failures, unchanged device
programs, every finite BF16 encoding against Merlin's independent narrowing
implementation, native publication/rebinding, generated C dispatch, aliases,
late invalid inputs, and NPU execution with padded/batched outputs and a late
exponent rejection. Run it with the existing epilogue regression gates:

```
ATLAS_RUN_NPU_MODEL=1 python -m pytest -q \
  test/test_quantized_narrowing.py test/test_quantized_epilogue.py \
  test/test_quantized_epilogue_lowering.py
```

All 13 tests and nine subtests passed. The broader fallback, DRAM-boundary,
epilogue and materialization suite passed 31 tests and 21 subtests, with one
optional test skipped. All implementation changes are in Atlas; Merlin's
existing region selector consumes the target-proved alternative unchanged.

## Wider activation reuse and shared weight panels

Two further opt-in schedules extend the quantized MXU1 path. `wide-cache`
retains up to 192 activation tiles, covering the full K=5632 row. `paired-cache`
shares weight panels across two M32 tile rows while also reusing their activation
panels across output columns. Its partial sums cross K-panel boundaries as
ordered BF16 values in the existing output buffer, then undergo the original
restoration and admission checks. See the [schedule contracts](quantized-operand-reuse.md).

Fresh cycle runs on the same full down projection (M=64, N=2048, K=5632) and
unchanged model revision gave:

| Schedule | Cycles | DRAM bytes loaded + stored | Program bytes |
| --- | ---: | ---: | ---: |
| Ordered | 35,804,760 | 46,925,824 | 3,476 |
| Previous 64-tile prefix | 29,631,705 | 38,668,288 | 11,040 |
| Wide activation cache | 18,822,137 | 24,217,600 | 24,832 |
| Paired activation/weight panels | 15,302,709 | 15,828,992 | 50,080 |

Wide caching saves another **10,809,568 cycles (36.48%)** over the prefix.
Paired weight reuse saves another **3,519,428 cycles (18.70%)** over wide caching,
including the cost of writing and reloading partial sums. Together these two
features save **14,328,996 cycles (48.36%)** over the previous prefix schedule,
or **57.26%** versus ordered execution. These are down-projection results, not
a remeasured whole-layer total.

Every retained host reference read matched in both full runs. Independent
protected-memory checks passed, and their final complete-memory SHA256 equals
the ordered and prefix runs:
`c6e39e07404c9af7c1984b551226550833c0c7e0c640c2727c392daf7c7ebe17`.
Replaying the integrated policies emitted byte-identical benchmark images.
The physical instruction bounds were 529,983 (wide) and 561,260 (paired), below
the unchanged one-million limit; image sizes remain below 128 KiB.

The paired schedule's DRAM service cost remains 10,716,974 cycles. Additional
VMEM movement and partial-sum handling mean its cycle savings are smaller than
the raw bandwidth savings alone would suggest. Its larger image is another
explicit tradeoff. The schedule declines small K, odd M-tile counts, multiple
batches and quantized multiply epilogues, using the wide-cache policy instead.

Enable both opportunities for supported shapes before compilation:

```bash
export ATLAS_QUANTIZED_OPERAND_REUSE=paired-cache
```

For comparison, select `wide-cache`, `prefix-cache` or `ordered`. The default
remains `ordered`; no hardware limits, operand codes or exponent semantics
change. All code is in Atlas; the pinned Merlin fork checkout remains untouched.

Three wide-cache tests and four paired-cache tests cover native publication,
recorded-policy replay, fallback, tails, batches/row pairs, nonuniform scales,
late rejection, exact traffic and complete-memory equality on the NPU model.
Native publication passed both mandatory and optimized pipelines, and the
optimized paired program also passed NPU execution against the ordered reference.
The broader regression run passed 45 tests and 30 subtests, with four optional
tests skipped and one qualified-native test deselected. Local evidence, including the wide and paired benchmark results, the
image-equivalence log and the test logs, is not included in the source commits.

## Remaining work

PR #44 already implements A/W caching, accumulator retention and allocation
fallbacks in the older model-compilation path. Those changes are not duplicated
here. The new schedules apply those reuse principles to the distinct ordered
quantized MXU1 path. Further work includes reducing partial-sum traffic, supporting
additional panel geometries, and eliminating more intermediate buffers. A fresh
whole-layer profile should determine their priority.
