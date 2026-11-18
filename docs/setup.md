# Setup and compiler development

For the wheel quickstart, see the [README](../README.md). This guide contains
source-build instructions, capture dependencies and optional test setup.
Run the commands from the repository root.

## Capture and application compilation

See the [README](../README.md) for the Merlin-backed build, raw export loader,
explicit precision policy, and Python 3.12 CPU capture setup. `atlasc build` does
not use torch-mlir. The Python API has the same loader contract:

```python
from pathlib import Path
from atlas_mlir import build
result = build(Path("model.py"), out=Path("atlas-out/model"),
               dtype="fp32", precision_policy=Path("precision.py"),
               application_memory_bytes=536870912)
```

`--capture-python` selects a separate interpreter with the same pinned Atlas,
Merlin and model2MLIR installations. `--snapshot` reuses a native snapshot;
otherwise Cargo builds one in the output directory. `--recipe`, `--board`,
`--application-residency` and `--application-region-loop` forward selections to
Merlin. No compiler failure falls back to the standalone tile path.

The historical torch-mlir path is explicitly `atlasc tile-demo`; its optional
`[tile-demo]` dependencies belong in a separate environment because they use
PyTorch 2.10.0 instead of the application compiler's 2.11.0+cpu.

## Build from source

Use MLIR and LLVM from one installation. The wheel CI uses conda-forge
LLVM/MLIR 23.1.2; [ci/environment.yml](../ci/environment.yml) specifies its build
environment. The original source build also used LLVM/MLIR 23.0.0git,
`a47bddccec30255619bb8c37fa59700e661d4e66`.

Activate the C/C++ compiler environment that supplied LLVM before building
(for conda, activate the environment in `ci/environment.yml`). A mismatched
compiler may link successfully but crash on MLIR input; wheel installation
now refuses that build. With MLIR installed, `CMAKE_ARGS="-DMLIR_DIR=/path/to/lib/cmake/mlir"
python -m pip install .` builds and installs everything in one command.
For compiler development, use the separate editable Python/CMake workflow:

```sh
python -m pip install -e '.[test]'
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=/path/to/model-venv/bin/python \
  -DATLAS_NPU_MODEL_ROOT=/path/to/npu_model
cmake --build build --parallel 2
PYTHONPATH=/path/to/pinned-merlin-checkout ctest --test-dir build --output-on-failure
```

CTest uses Python 3.14 with the pinned npu_model `b97b2f5a` checkout, CPU
PyTorch 2.11.0+cpu, pytest, tabulate, and the installed Atlas/Merlin dependencies.
Use an editable model install: its arithmetic corpus is absent from its wheel.
Several provider tests import Merlin source fixtures; put the pinned Merlin
checkout root on `PYTHONPATH` for CTest (the production wheel does not need it).
`-DATLAS_ENABLE_TESTS=OFF` builds the compiler without the model.

CTest runs the portable tests with no hardware. Tests that run the RTL need
`ATLAS_RTL_ROOT`, `ATLAS_MODEL_ROOT`, `ATLAS_ASSEMBLER_ROOT`, `ATLAS_ARC_MODEL`,
`ATLAS_ARC_STATE`, `ATLAS_MODELIR_ROOT`, and `ATLAS_LLVM_BIN`; without them they
skip. To run the Python tests against another build, set `ATLAS_OOT_BIN_DIR` to
its `bin` directory.

`ATLAS_ASSEMBLER_ROOT` selects the directory containing `atlas_asm.py`.
The shared test helper runs canonical npu_model assembly in a Python 3.14
subprocess; CTest keeps its own Python interpreter and test dependencies.
Initialize Atlas's `npu-model` submodule and install `uv`, then prepare its
locked dependencies outside the source tree:

```sh
export ATLAS_ASSEMBLER_ROOT=/path/to/atlas-npu/baremetal
export NPU_MODEL_DIR=/path/to/atlas-npu/npu-model
export UV_PROJECT_ENVIRONMENT="$PWD/out/assembler-venv"
uv sync --locked --no-install-project --project "$NPU_MODEL_DIR" --python 3.14
ctest --test-dir build --output-on-failure
```

`NPU_MODEL_DIR` defaults to Atlas's sibling `npu-model` directory and must
match its gitlink. A selected but missing tool or dependency fails the test.
These assembler revisions are separate from the recorded RTL/model evidence.

## Low-level MLIR tools

```sh
mkdir -p out/quick
build/bin/atlas-opt --lower-atlas-virtual-to-machine --insert-atlas-dma-waits \
  --insert-atlas-delays test/examples/virtual_fp8_two_layer_mlp.mlir > out/quick/mlp.atlas.mlir
build/bin/atlas-emit out/quick/mlp.atlas.mlir                  # hex words
build/bin/atlas-emit --program-json out/quick/mlp.atlas.mlir   # words with decoded fields
build/bin/atlas-opt --convert-atlas-to-llvm-calls out/quick/mlp.atlas.mlir |
  build/bin/atlas-opt --finalize-atlas-llvm-calls -              # LLVM dialect
```

For full end-to-end runs, see the [examples](../examples/handoff/README.md): LLVM
and ELF stages for small tiles, a captured attention + MLP block from Linalg
to a device program and host ELF, and tiled matmuls launched from a host.

## Package CI

Every push, pull request and merge queue run builds an sdist, builds its wheel,
runs the portable compiler tests, then installs the wheel on separate runners
for base-package and PyTorch smoke tests. Artifacts are downloadable after the
build. Configure the **Package** workflow's checks as required in branch
protection to enforce the gate; the workflow itself cannot change repository
branch-protection settings. No PyPI credentials are needed.

