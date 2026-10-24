# Bounded model tile runner

`tools/run_model_tile.py` accepts a source-bound contraction, independently
checks conversion/packing and MXU0 arithmetic, assembles a conservative fixed
schedule with the current model assembler, and executes its decoded instruction
objects with explicit `rtl` numerics. It writes raw instruction words as
reviewable artifacts; the model does not decode or execute `program.bin`.
Results are model evidence only.

The named recipe `atlas_mxu0_fp8_unit_bf16acc_v1` converts source operands to
BF16 nearest-even, then applies the VPU FP8Pack unit-scale rule (E8M0 code 127).
It flushes underflow, uses the VPU saturation behavior, and differs from MXU
FP8-pop conversion and generic Torch FP8 casts. MXU0 rounds every MAC to BF16,
resets the first K32 product and continues later products. This is an explicit
target derivative; native outputs remain an unchanged diagnostic reference. No
model-quality equivalence is asserted.

The opt-in `--recipe tensor-pow2` selects the experimental
`atlas_mxu0_fp8_tensor_pow2_bf16acc_v1` derivative. Each complete captured BF16
operand tensor gets the smallest power-of-two scale whose maximum normalized
magnitude is at most 448; all-zero tensors use exponent zero. Selection uses
exact rational comparisons, admits operand exponents -126 through 127, and is
independent of workload identity. The independent scalar encoder and pinned VPU
FP8Pack converter must agree byte-for-byte. Scales remain constant across every
K contribution and output tile. After the BF16 MXU0 pop, actual VLI.ALL and
VMUL.BF16 instructions multiply both output halves by the product of the operand
scales; those instructions and their delays are included in model cycles. The
product scale must be normal finite BF16, and output overflow is refused. Source
error and underflow remain diagnostic; this recipe is not adopted model
arithmetic or calibrated block scaling. Omitting the option preserves the
original unit-scale instruction image.

The rescale oracle preserves signed zero, matching VPU multiplication. Receipts
bind exact oracle source bytes; changed helpers require fresh qualification.

Use the supplied Python 3.14 model environment with `PYTHONDONTWRITEBYTECODE=1`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. Supply
an isolated model root at `9e6313bddecb8ba936206eb347f5cd4e5335282c`. The runner
rejects another revision, tracked model-source changes, and imports outside that
root. All path components must be physical paths without symlinks. Select a
fresh physical output directory.

```sh
python tools/run_model_tile.py --model-root /physical/pinned/npu-model \
  --input /physical/capture/operands.npz --manifest /physical/capture/manifest.json \
  --out /physical/new-artifact-directory
```

For initial loading/layout bringup, replace `--input` and `--manifest` with
`--smoke`. This constructs a nonsymmetric K64 fixture, explicitly labeled
synthetic. Default runs use two independent reset-initialized model instances
with distinct deterministic storage seeds. Completion requires ECALL, no
outstanding engines or DMA, exact independent output bytes, unchanged
inputs/guards, and observed DMA bytes equal to requested transfers. The runner
observes these conditions before any stop/flush helper; it never invokes that
helper. Smoke also executes omitted-transpose, final-K-reset and wrong-oracle
negative controls; only the expected output-mismatch exception after successful
completion/guard/traffic checks satisfies these controls. `--negative-controls`
requests them for another K>=64 fixture.

The NPZ must contain finite float32 arrays `A[32,K]`, `B[K,32]`, and
`native_C[32,32]`; K is a positive multiple of 32 up to 4096. BF16 sources use
exact float32 views of their values. Optional uint16 arrays `A_bf16`, `B_bf16`,
`native_C_bf16` authenticate the corresponding exact views. Before NumPy loads a
member, the ZIP directory must contain exactly the required flat `.npy`
filenames and only these optional filenames, with no duplicates or path members,
and at most 32 MiB total uncompressed content. No object arrays, unsupported
dimensions or additional keys are accepted. The capture remains responsible for
the native operation boundary and complete reduction range.

The JSON manifest requires `schema_version: 1`, `npz_sha256`, `source_dtype`
(`float32` or `bfloat16`), and nonempty `source` and `site` objects containing
capture identities and semantic/range metadata. It is retained verbatim.
Workload names never choose production behavior.

`receipt.json` binds input, runner, oracle, imported model sources, memory curve
and output hashes. It reports model cycles from the first fetch edge through
ECALL, issue/stall/engine-busy observations, actual DMA A-channel bytes, guards,
and native-error diagnostics. `cycles-*.jsonl` retains issue/halt and DMA
handshake events; `perfetto-*.json` contains the model trace. Requested transfer
bytes and granted DMA VMEM traffic are separate from engine traffic; the runner
does not claim measured MREG traffic, host execution, transfer packing cost or
device timing. Raw source, encoded operands, initial DRAM, independent expected
output, actual output and instruction image are retained. Failure leaves its
receipt and available artifacts in place and returns nonzero.

Portable oracle/admission controls run without importing the model:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s test -p test_tile_arithmetic.py -v
```

The exact-rational oracle is independent of the execution model. Its finite
domain rejects nonfinite source and BF16 overflow; unit-scale packing avoids a
new calibration or scale-compensation policy. Bounded
decoded-value/product/transition caches preserve exact integer keys and rational
arithmetic; cached and uncached recurrences have separate directed and seeded
controls. Source NPZ/manifest, imported model files, selected pin/diff, memory
curve, runner and oracle hashes must agree before/after qualification. Broader
arithmetic, overlapping schedules, integrated RTL, runtime restart semantics and
source-quality admission require separate evidence.

`tools/atlas_program.py` provides contraction assembly and input admission to
`tools/compile_workload_inputs.py` and `tools/run_workload_contraction.py`.
These are provider APIs; use the workload producer below for complete physical
contractions, or the [ResNet graph route](resnet-full-host.md) for host
integration.

The optional NumPy module `tools/atlas_vector_arithmetic.py` exports
`ordered_matmul(uint8[M,K], uint8[K,N]) -> uint16[M,N]`, `round_bf16(float64
array) -> uint16 array`, and `rescale_bf16_array(uint16 array, exponent) ->
uint16 array`; callers explicitly choose and bind that module. The rational
implementation remains the default unless a caller explicitly selects the vector
provider. Both arithmetic implementations visit K in order and round every
nonzero FP8 product update directly to BF16 nearest-even. The vector
implementation uses exact binary64 intermediates and integer bit rounding, with
no float32 intermediate: finite FP8 products lie on a 2^-18 grid and have
magnitude below 2^18, while zero-initialized BF16 accumulation cannot cross the
2^26 barrier. This bounds the exact sum to fewer than 46 significant binary
digits, inside binary64's 53. Rescaling keeps the same exponent admission, exact
overflow refusal, subnormal rounding and signed-zero behavior. The module
docstring states the proof; `test/test_vector_arithmetic.py` compares exhaustive
finite FP8 pair products, all finite BF16 decodes, ordered chains, rounding
boundaries, rescaling and physical output layout against the rational
implementation. This optional arithmetic helper changes no device instruction,
output reference, tolerance, producer source pin or native-quality gate. A
caller using it must authenticate its source separately; it does not silently
change the existing capsule/replay oracle contract.

`tools/run_workload_contraction.py` is a separate development producer for
finite FP32 physical operands with M/N/K positive multiples of 32, K<=8192, at
most 1,048,576 output cells, at most 64 MiB guarded DRAM, and at most 8192
instruction words. The existing input archive admission still limits
uncompressed source members to 32 MiB. Callers provide explicit tail padding and
a source-bound A/B/native_C archive/manifest; the tool executes the complete
physical contraction and reports native error diagnostically without a quality
gate. It selects resident B if that operand fits one 256 KiB bank, otherwise
resident A if it fits, otherwise streams both, using the unchanged
instruction-schedule generator. Its receipt records bank 1 resident bounds,
disjoint bank 0 scratch ranges, the single DMA low32 window under high base 1,
actual whole-output equality, traffic, guards and drained ECALL. Five imported
producer helpers are path checked, snapshotted and bound before/after execution.
This producer has its own identity and does not change the original K<=4096
admission or qualify an existing capsule/replay receipt. A separate
current-program RTL gate and host runtime qualification are still required after
its instruction-model result.
