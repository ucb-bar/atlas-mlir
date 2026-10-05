# Bounded native shape-seed execution on selected AtlasCore

This observation uses Merlin `3a10b83dfd9da119efb8e22e504ca5ebef76b02e`,
Atlas native support wheel `0.0.6`, and Atlas RTL
`0079c0541111197741a231c002e3843fa6f545b2`. A fresh native target
snapshot built in 6.686 seconds. The installed `merlin-targetgen` CLI compiled
all eight public E3 seed shapes in strict-native mode. Compilation received
only typed requests, fixed ABIs, and the selected target profile. The runtime
inputs and expected outputs were constructed later by the evaluator.

| Shape `(M,K,N)` | Emitted programs | Native compile and replay | Exact selected-core panels |
| --- | ---: | ---: | ---: |
| `(32,32,32)` | 1 | pass | 2/2 |
| `(64,32,96)` | 6 | pass | 2/2 |
| `(32,64,32)` | 1 | pass | 2/2 |
| `(64,64,64)` | 4 | pass | 2/2 |
| `(31,32,33)` | 1 | pass | 2/2 |
| `(33,31,65)` | 6 | pass | 2/2 |
| `(1,65,7)` | 1 | pass | 2/2 |
| `(7,1,33)` | 1 | pass | 2/2 |

The 16/16 panels launched 42 physical programs. Each launch halted, preserved
its input and guard regions, and contributed to a byte-for-byte BF16 output
comparison. The observed launch cycles ranged from 495 to 1,083. The panels
use E4M3 operands drawn from zero and ±1, so their expected integer sums are
exact BF16 values. This is a deliberately bounded arithmetic check, including
multi-tile and tail shapes, not a qualification of general FP8 arithmetic.

A separate `(31,32,33)` compile with `candidate_nodes=12` forced the combined
graph to hit `resource_limit`; search feedback emitted two independently
selected programs and replayed their manifests. It was a compiler mechanism
test and was not included in the 16 executed panels.

The selected standalone model and state SHA-256 values were
`196380fca0cb3416a188538b342f8940802ebcf4d42e8d05dfe351b00b2c46fc`
and `db2d8ae3c8a4ce6a417b0c691be1d446f1c0be95efe5607a251a6946a80de9e3`.
The ModeLIR Atlas driver SHA-256 was
`d1897fe1b49adf68cc3d8dc46794f22f1b98418eef6cfea9231aee15cb9c79c2`.
The evaluator-only source is [`qualify_native_shapes.py`](../test/qualify_native_shapes.py);
the compiler input generator and replay script are
[`generate_native_shape_seeds.py`](../test/generate_native_shape_seeds.py) and
[`compile_native_shape_seeds.py`](../test/compile_native_shape_seeds.py).

The same eight shapes and forced fallback compiled in a fresh wheel-only Python
environment without ACT compiler packages. Its 119 checked program and plan
files were byte-identical to the source-pin artifacts. This checks packaging
and deterministic output for this suite; it does not establish a complete
ACT-unavailable process/import trace.

To reproduce with locally selected paths and fresh output directories:

```sh
merlin-targetgen native-build --engine merlin_native --support atlas_tensor \
  --cargo-target-dir "$ARTIFACT_ROOT/cargo" \
  --source-revision 3a10b83dfd9da119efb8e22e504ca5ebef76b02e \
  --out "$ARTIFACT_ROOT/snapshot"
python3 test/compile_native_shape_seeds.py \
  --snapshot "$ARTIFACT_ROOT/snapshot" \
  --target-source "$ATLAS_RTL_ROOT" --out "$ARTIFACT_ROOT/compiled"
python3 test/qualify_native_shapes.py \
  --arc-model "$ARC_MODEL" --arc-state "$ARC_STATE" \
  --modelir "$MODELLIR_ROOT" \
  --expected-model-sha256 196380fca0cb3416a188538b342f8940802ebcf4d42e8d05dfe351b00b2c46fc \
  --expected-state-sha256 db2d8ae3c8a4ce6a417b0c691be1d446f1c0be95efe5607a251a6946a80de9e3 \
  --expected-modelir-driver-sha256 d1897fe1b49adf68cc3d8dc46794f22f1b98418eef6cfea9231aee15cb9c79c2 \
  --compiled-root "$ARTIFACT_ROOT/compiled" \
  --out "$ARTIFACT_ROOT/execution-receipt.json"
```

This is standalone AtlasCore evidence with diagnostic static delays. It does
not establish integrated `EE290SimConfig` timing, a host/Zephyr launch path,
full model invocation, or D/F/M/N release gates.
