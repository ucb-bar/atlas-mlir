# Merlin-backed pip package

This replaces the earlier standalone tile-driver plan. User commands and loader
examples are in the [README](../README.md).

## Ownership and pins

One `atlas-mlir` distribution owns `atlas_mlir`, `atlas_mlir_native`, and
`atlas_m2m_adapter`, with exactly one Atlas compiler plugin and one adapter plugin.
The default build calls the pinned Merlin capture worker, stages model2MLIR's
materialized bundle, compiles an application, and replays verification. Atlas
owns region emission and runtime ABI files. Application orchestration stays in
Merlin. The legacy direct importer remains only in `tile_demo.py`.

| Component | Selected revision / version |
| --- | --- |
| Atlas integration base | on-chip deployment `b707519f6448ea74000e438650584dd071e7ac60` |
| Merlin | `650cc3400858515bd2722db2603bf35ee6a05f2f` |
| model2MLIR | `ebb1c600e0d7f3205c801d8f00acb960015b32f2` |
| atlas-torchao source | `86b525b0ff93fd94d66875d30a4ad5c5b055b443` |
| npu_model | `b97b2f5a97e432141c4ef8504bec396651d77a8f` |
| Capture Python / torch / xDSL | 3.12 / `2.11.0+cpu` / `0.68.0` |
| Numerical model Python / torch | 3.14 / `2.11.0+cpu` |
| Wheel LLVM/MLIR | conda-forge 23.1.2 |

The consolidation branch was not published when this integration was made.
The selected deployment branches are the handoff's fallback. Git dependencies
are pinned, not installed from the unrelated `merlin` distribution on PyPI.
`atlasc doctor` checks source identities and duplicate plugin registrations.
Quantized capture also checks exporter and reference submodule revisions.

## Packaging and provenance

`toolchain.py` resolves the installed target root and binaries for every provider
module. The wheel carries the complete runtime C/H/JSON/S/LD set, inventory, and
Python tools. An explicit source override never silently borrows another
installation's binaries. `native-compile --target-source` receives the wheel's
`atlas_mlir/share` by default.

The wheel build record includes the Atlas revision, dirty state, LLVM version,
compiler identity, and SHA-256 of binaries, bundled libraries and support files.
Reports include this record and observed Python dependency versions. Hashes are
relative to the package/build output; diagnostic logs can include absolute paths.
Merlin's deterministic receipt option is enabled. Capture source identities may
still differ between directories because upstream IR includes its weights path;
this is not a promise that every report is byte-identical after relocation.

To make an sdist from a clean checkout and build its wheel with the selected LLVM:

```sh
git diff --exit-code HEAD
git rev-parse HEAD > SOURCE_REVISION
CMAKE_ARGS="-DMLIR_DIR=$CONDA_PREFIX/lib/cmake/mlir -DLLVM_DIR=$CONDA_PREFIX/lib/cmake/llvm" \
  python -m build
```

Activate the C/C++ compiler environment belonging to LLVM first. `SOURCE_REVISION`
is a generated release stamp, included in the sdist and ignored by Git. Remove
it before subsequent development builds so they report the current checkout.

## Qualification boundaries

CI builds the sdist/wheel, supplies `ATLAS_NPU_MODEL_ROOT` to CTest, exercises the
installed wheel outside the source tree, compiles and verifies a floating matmul
and a raw quantized linear, runs adapter mutation/requantization rejection tests,
and runs emitted-program numerical tests with the updated CPU model.

This does not reuse CUDA-variant arithmetic receipts. Compiler replay is not
numerical execution, FPGA qualification, or whole pretrained-model validation.
The older kit's LLVM `a47bddcc` is not provisioned by this wheel. Cross-toolchain
equivalence requires a separate corpus comparison and is not claimed here.
