# Atlas MLIR

Atlas MLIR is an out-of-tree MLIR dialect and compiler for the Atlas NPU. It is
written by hand as a reference to compare against Merlin-generated target
dialects, and it targets one pinned RTL revision: `ucb-bar/atlas-npu`
`0079c0541111197741a231c002e3843fa6f545b2`.

It has two Atlas levels:

- **Virtual SSA**: BF16/FP8 tile values, state tokens, control flow, atomic DMA,
  and MXU handles before physical placement.
- **Machine IR**: one typed operation per Atlas instruction with its encoded
  fields. All 99 decoder modes of the RTL can be written and encoded.

From machine IR, `atlas-emit` writes the 32-bit instruction words directly, or
the program can go through LLVM (`mlir-translate`, `llc`) as one inline-assembly
block in a RISC-V object. A RISC-V CPU cannot run those words; a host loads them
into Atlas and starts it ([host launch](docs/host-launch.md)).

The `atlas-mlir` pip package provides the `atlasc` command, the native
`atlas-opt` and `atlas-emit` tools, and the Atlas plugin for Merlin. Wheels
include the compiler libraries and host runtime sources; LLVM is only needed
when building the package from source.

## Install and compile

Download the Linux x86-64 wheel from the **Package** GitHub Actions artifact,
unzip the artifact, and run these two commands in a Python environment:

```sh
python -m pip install ./atlas_mlir-*.whl
atlasc demo
```

The bundled two-layer MLP needs no PyTorch, external LLVM installation, or
hardware. `atlas-out/demo` contains the device binary, decoded instructions,
virtual and machine MLIR, host headers, memory layout and `report.json`.
This compiles a program; executing it requires an Atlas model or device.

For your own PyTorch model, install capture support and compile a loader:

```sh
python -m pip install './atlas_mlir-0.1.0-py3-none-linux_x86_64.whl[capture]'
atlasc build model.py
```

A loader is an ordinary Python module defining `get_model_and_inputs()`:

```python
import torch


def get_model_and_inputs():
    model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(),
                                torch.nn.Linear(32, 32)).eval()
    return model, (torch.ones(32, 32),)
```

`atlasc build model.pt2` also accepts `torch.export.save` files. Output defaults
to `atlas-out/<model-name>`; use `-o DIR` to choose it, and `--force` to replace
an earlier `atlasc` build. A RISC-V compiler on PATH adds a bare-metal host ELF;
`--host-cc clang` selects Clang with LLD, and `--no-host-elf` skips the link.
`atlasc doctor --capture` checks the installed compiler and capture imports.
`atlasc doctor` checks only the base package.

The capture extra pins PyTorch 2.10.0 and torch-mlir 20261001; CI covers
Python 3.10/3.12. For CPU-only PyTorch (or if your index lacks torch-mlir):

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-mlir==20261001 -f https://github.com/llvm/torch-mlir/releases/expanded_assets/v20261001
```

From Python: `from atlas_mlir import compile; result = compile(model, inputs,
"atlas-out/model")`. Already captured graphs use `atlasc build --linalg FILE
--arguments MANIFEST --weights WEIGHTS [--inputs INPUTS]`.
The default `fp8` policy is diagnostic FP8/BF16 arithmetic, not f32 equivalence.
Use `--policy policy.json` for an explicit numerical policy. See the supported
operations and shape restrictions in the [binding guide](docs/native-merlin-binding.md).

Every push, pull request and merge queue run builds an sdist, builds its wheel,
runs the portable compiler tests, then installs the wheel on separate runners
for base-package and PyTorch smoke tests. Artifacts are downloadable after the
build. Configure the **Package** workflow's checks as required in branch
protection to enforce the gate; the workflow itself cannot change repository
branch-protection settings. No PyPI credentials are needed.

## Build from source

Use MLIR and LLVM from one installation. The wheel CI uses conda-forge
LLVM/MLIR 23.1.2; [ci/environment.yml](ci/environment.yml) specifies its build
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

For full end-to-end runs, see the [examples](examples/handoff/README.md): LLVM
and ELF stages for small tiles, a captured attention + MLP block from Linalg
to a device program and host ELF, and tiled matmuls launched from a host.

## Read next

| Doc | Covers |
| --- | --- |
| [Dialect reference](docs/dialect-reference.md) | Operations, verifiers, passes, tools, adding a pass. |
| [Compiler pipeline](docs/compiler-pipeline.md) | Virtual lowering, interpreter, physical program, DMA, LLVM handoff. |
| [Register spilling](docs/register-spilling.md) | FP8/BF16 pressure recovery, VMEM slots, checks and current limits. |
| [Native Merlin binding](docs/native-merlin-binding.md) | Compiling graphs through Merlin. |
| [Host launch](docs/host-launch.md) | Host runtime, memory map, mailbox, board builds. |
| [Hardware status](docs/hardware-status.md) | Which instructions have run on the RTL, and known RTL/model differences. |

## Limits

This is not a general model compiler. It handles static 32×32 tiles and the
graphs listed in the binding doc, with FP8/BF16 precision policies that change
results relative to f32. Programs have run on a standalone Atlas core built
from the pinned RTL, not on an integrated Shuttle + Atlas SoC or an FPGA.
