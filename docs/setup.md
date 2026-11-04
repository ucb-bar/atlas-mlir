# Setup and compiler development

For the wheel quickstart, see the [README](../README.md). This guide contains
source-build instructions, capture dependencies and optional test setup.
Run the commands from the repository root.

## Capture dependencies

The capture extra pins PyTorch 2.10.0 and torch-mlir 20261001; CI covers
Python 3.10/3.12. For CPU-only PyTorch (or if your index lacks torch-mlir):

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-mlir==20261001 -f https://github.com/llvm/torch-mlir/releases/expanded_assets/v20261001
```

## Advanced driver usage

`atlasc build model.pt2` accepts `torch.export.save` files. Use `-o DIR` to choose
the output directory and `--force` to replace an earlier build. A RISC-V compiler
on PATH adds a bare-metal host ELF; `--host-cc clang` selects Clang with LLD, and
`--no-host-elf` skips the link. `atlasc doctor` checks the base package;
`atlasc doctor --capture` also checks PyTorch capture imports.

The Python API is `from atlas_mlir import compile; result = compile(model, inputs,
"atlas-out/model")`. Already captured graphs use `atlasc build --linalg FILE
--arguments MANIFEST --weights WEIGHTS [--inputs INPUTS]`.

The default `fp8` policy uses diagnostic FP8/BF16 arithmetic. Use `--policy
policy.json` to select an explicit numerical policy. See the
[binding guide](native-merlin-binding.md) for supported operations and shapes.

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
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

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

