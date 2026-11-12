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

This changes binary emission by default. Unary emission and its explicit
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
are in `out/next-audit/`.

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
`out/next-audit/bench_down.py` and `out/next-audit/down-benchmark.json`.
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
`out/next-audit/pv-narrowing-compiled/package.json`.

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

## Remaining work

PR #44 already implements A/W caching, accumulator retention and allocation
fallbacks in the older model-compilation path. Those changes are not duplicated
here. The outstanding work is broader panel reuse beyond the resident prefix,
weight reuse combined with activation reuse, and removal of further intermediate
buffers beyond the exact narrowing pair above. Those features are not implemented
by these commits and need their own correctness and performance gates.
