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

## Remaining work

PR #44 already implements A/W caching, accumulator retention and allocation
fallbacks in the older model-compilation path. Those changes are not duplicated
here. The outstanding work is bounded wide-K reuse on the quantized MXU1 path,
weight reuse combined with activation reuse, and shared graph/transport changes
for eliminating intermediate materialization. Those features are not implemented
by the binary-kernel commit and need their own correctness and performance gates.
