# TinyLlama layer through generalized Merlin passes

This diagnostic path compiles a complete batch-one prefill layer to a native
host library plus Atlas programs. It uses the generalized Merlin branch, rather
than the older `atlas-model-compilation` model stack. No compiler-authoring or
optimization agent loop is involved.

The capture contains RMSNorm, Q/K/V projections, rotary position encoding,
grouped-query attention, causal masking, softmax, output projection, both
residual additions, and the gated SiLU MLP. It returns the layer output and the
current K/V tensors. All nine contractions execute on Atlas; other operations
run in the original f32 graph lowered through Merlin's existing MLIR-to-LLVM
pipeline. This is synchronous host/device partitioning. It is not an all-device
layer, a complete language model, historical KV-cache management, or decoding.

The fixture uses reproducible synthetic weights with the full TinyLlama geometry:
hidden size 2048, intermediate size 5632, 32 query heads, four KV heads, head
width 64, and sequence length 32. Model names never select production lowering.
The same compiler also handles the reduced 128/256-width fixture and a
120/270-width fixture with sequence length seven and 30-wide heads.

## What changed

Merlin's closed tensor reader now understands named `linalg.batch_matmul` with
checked batch geometry, indexing maps, scalar multiply/add recurrence, and a
positive-zero seed. The same checks qualify both regular and batched matmul for
a generic callback outlining pass. Fresh output buffers, preserved other
results, strided input copies, host lowering, the C-interface bridge, and the
MLIR C runtime stay in Merlin. There are no Atlas resource facts in these passes.

Atlas selects one reusable 32-by-K-by-32 program per padded K. The existing
shared dense scheduler checks the actual ordered tile calls. The OOT consumes
that trace with a counted K loop, carrying the BF16 accumulator through
readout/restore. It never splits K into independent partial sums. M/N tiles and
batches are dispatched independently by the test transport; positive-zero tails
and boundary conversion are explicit numerical-policy choices.

Bounded DMA analysis, address verification and code-size checking reuse Atlas
commit `3779a1ff4ca7529121778b02b78cbda51f9e1d39` (cherry-picked as `9e6b8cd`).
Only this separate worktree was built or edited. The reference model is pinned
to `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`, and the compiler checks selected RTL
identity `0079c0541111197741a231c002e3843fa6f545b2`.

## Numerical and execution contract

Compilation requires the explicit policy
`f32-e4m3-rne-saturate-flush-ordered-bf16-padded-v1`:

- Contraction operands must be finite f32. Conversion rounds to E4M3FN, saturates
  finite overflow to magnitude 448, and replaces encoded subnormals and either
  zero sign with positive zero.
- Each device multiply/add rounds its accumulator to BF16 in ascending K order.
  Output BF16 is widened exactly to f32 before continuing the host graph.
- K is padded to a multiple of 32 and limited to 8192 in this diagnostic path.
  M/N padding is discarded only after the complete device output is checked.
- Non-contraction host operations retain f32 semantics. The callback transport
  is synchronous and invocation-exclusive, with no CPU matmul fallback.

This policy is an approximation to the original f32 model. Synthetic-weight
correctness against the selected arithmetic does not establish pretrained model
quality, a suitable quantization policy, or hardware/RTL execution correctness.
FP4/FP6 codecs in Merlin are unchanged; Atlas does not execute those formats here.

## Reproduce

Set the paths to your separate checkouts and environments. The capture Python
needs Torch, NumPy and torch-mlir. The compiler Python needs Merlin's xDSL/solver
dependencies. The model Python needs Torch, NumPy and the pinned npu_model.
The tested LLVM checkout is `a47bddccec30255619bb8c37fa59700e661d4e66`.

```sh
export MERLIN_ROOT=/path/to/merlin-atlas-target
export ATLAS_ROOT=/path/to/atlas-mlir-generalized-tinyllama
export MODEL_ROOT=/path/to/npu-model-atlas-target
export LLVM_ROOT=/path/to/llvm-project/build
export CAPTURE_PY=/path/to/capture/python
export COMPILER_PY=/path/to/compiler/python
export MODEL_PY=/path/to/model/python
cd "$ATLAS_ROOT"
cmake -G Ninja -S . -B out/build/layer/compiler \
  -DMLIR_DIR="$LLVM_ROOT/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_ROOT/lib/cmake/llvm" -DCMAKE_BUILD_TYPE=Release
cmake --build out/build/layer/compiler -j 6
"$CAPTURE_PY" test/models/tinyllama_layer.py --full \
  --destination out/build/layer/capture
export MERLIN_COMPILER_PYTHON="$CAPTURE_PY"
export MERLIN_MLIR_INSTALL="$LLVM_ROOT"
for pipeline in source-order optimized; do
  PYTHONPATH="$MERLIN_ROOT/src:$ATLAS_ROOT/python" "$COMPILER_PY" \
    -m atlas_mlir_native.shared_model out/build/layer/capture/layer.mlir \
    --entry main --precision f32-e4m3-rne-saturate-flush-ordered-bf16-padded-v1 \
    --destination "out/build/layer/$pipeline" \
    --atlas-bin-dir out/build/layer/compiler/bin \
    --llc "$LLVM_ROOT/bin/llc" --cc /usr/bin/gcc --pipeline "$pipeline"
  PYTHONPATH="$MERLIN_ROOT/src:$ATLAS_ROOT/python:$ATLAS_ROOT:$MODEL_ROOT" \
    "$MODEL_PY" test/run_shared_layer.py --workers 6 \
    "out/build/layer/$pipeline" out/build/layer/capture \
    "out/build/layer/run-$pipeline"
done
```

Use fresh output directories. Omit `--full` for the reduced fixture; use
`--tail --sequence 7` for the uneven-width fixture. `--workers` only parallelizes
independent simulator checks; it does not change generated code or cycle totals.

Each compilation retains the original source, outlined host MLIR, LLVM IR,
linked library, shared schedule, virtual/machine IR, instruction words and timing
checks. The manifest binds the tool and artifact hashes. The harness verifies
the model revision and tracked-source cleanliness, decodes and re-encodes the
actual binary, executes every output tile, checks input/guard preservation, and
compares every BF16 output bit with an independent Torch-based ordered reference.
It then checks every host partition against Torch at the recorded device
boundaries (rtol/atol 1e-5), including all three returned tensors. A separate
free-running Torch graph using the same contraction policy reports end-to-end
drift; it is not treated as bit-exact source equivalence. Expected results are never compiler
inputs. Final reports and output arrays are written to each run directory.

Cycle reports sum completed device launches from npu_model. They exclude host
computation, quantization, packing, physical transport and launch overhead.
Static instruction-word counts describe resident programs, not dynamic retired
instructions. These numbers are not end-to-end latency or a comparison against
a newly generated OOT backend.

## Quantization-boundary finding

The full-width test exposed a real numerical sensitivity, not a device-word
mismatch. At flattened RMSNorm element 30664 (sequence row 14), Torch gives
`-0.7187499403953552` and the compiled host gives `-0.7187501788139343`.
Both satisfy the host partition tolerance, but E4M3 round-to-nearest selects
codes `0xb3` and `0xb4`. This affects the gate and up projections and gives a
maximum final-output difference of approximately 54 against the free-running
Torch reference with synthetic weights. K/V outputs agree. Both compiler
scheduling modes produce the same effect.

The harness preserves this discrepancy as `free_reference_output_max_abs_errors`.
Its correctness gate is compositional: each actual device tile must match the
independent ordered arithmetic exactly; every host region must match Torch at
common device boundaries within the recorded f32 tolerance. Replaying recorded
device outputs is only a host-region test, never a replacement for simulator
execution or a compiler input. Reports say `partition_checks_passed`, not
whole-model equivalence. Stronger end-to-end accuracy needs a separately chosen
precision/calibration policy or a specified, consistent host reduction contract.

## Validation recorded on this branch

Merlin commit `22a0c1dda7cc0138966088e0f239315d9522872f` passed 255 focused tests.
The Atlas build passed all 156 CTest entries in the private test environment;
optional legacy-provider and hardware fixtures retain their explicit skips.
The additional counted-loop/rounding/replay suite passed 16 tests. Merlin's
repository-wide wiring gate has 17 failures, reproduced identically in a clean
`a80ae190` worktree; the other pre-commit checks passed. No ledger was relaxed.

| Synthetic layer | Tiles per mode | Source-order device cycles | Optimized device cycles |
|---|---:|---:|---:|
| 128/256 widths, sequence 32 | 40 | 328,984 | 318,216 |
| 120/270 widths, sequence 7 | 42 | 353,206 | 341,602 |
| 2048/5632 widths, sequence 32 | 656 | 77,091,152 | 74,111,184 |

Across these six runs, all 1,476 device tiles match the independent ordered
arithmetic bit-for-bit, input/guard bytes remain intact, and every host partition
passes. The three returned tensors are bit-identical between scheduling modes
for every fixture. All host-replay final-output differences are zero. The small
and tail fixtures also match the free-running Torch reference exactly; the
full-width free-running discrepancy remains approximately 54 as described above.
The full-width optimized cycle sum is 3.87% lower, with the previously stated
exclusion of host and transfer costs.

Local artifacts are retained under `out/build/generalized-layer`, including
`review.json`, compilation manifests, per-call operand/output snapshots and the
six final run reports. This is an initial integration baseline. Production
transport, stronger end-to-end numerical qualification, pretrained weights,
full-size stacked layers, pretrained embedding/logits and KV-cache decoding remain
separate work. The [additional model validation](generalized-model-validation.md)
now exercises integer-token ABI and embedding/logits on the scaled Llama PoC,
plus reduced SmolVLA and complete ResNet-50 at 32×32.
