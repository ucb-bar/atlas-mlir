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

## Remaining work

PR #44 already implements A/W caching, accumulator retention and allocation
fallbacks in the older model-compilation path. Those changes are not duplicated
here. The outstanding work is broader panel reuse beyond the resident prefix,
weight reuse combined with activation reuse, and shared graph/transport changes
for eliminating intermediate materialization. Those features are not implemented
by these commits and need their own correctness and performance gates.
