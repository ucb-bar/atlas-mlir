# Atlas MLIR

Atlas MLIR compiles supported PyTorch and Linalg workloads into programs for the
Atlas NPU. The pip package includes `atlasc`, the native `atlas-opt` and
`atlas-emit` tools, and the Atlas plugin for Merlin.

## Install and run `atlasc`

On `feat/qwen-kernels`, `atlasc` is the command-line entry point for compiling
supported PyTorch and captured Linalg workloads. Use Python 3.12 on Linux
x86-64 and the wheel supplied with this branch, or download and unzip its
**Package** GitHub Actions artifact:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install ./atlas_mlir-*.whl
atlasc doctor
atlasc demo
```

The bundled MLP demo needs no PyTorch or local LLVM installation. It writes the
device binary, MLIR, host headers and a report to `atlas-out/demo`. Running the
compiled program requires an Atlas model or device.

## Compile a PyTorch reference with `atlasc`

In the same environment, install the tested CPU capture dependencies:

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-mlir==20261001 -f https://github.com/llvm/torch-mlir/releases/expanded_assets/v20261001
atlasc doctor --capture
```

Save your reference as `model.py`, defining `get_model_and_inputs()`. For example:

```python
import torch


def get_model_and_inputs():
    model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(),
                                torch.nn.Linear(32, 32)).eval()
    return model, (torch.ones(32, 32),)
```

Compile it into an Atlas device program and host headers:

```sh
atlasc build model.py -o atlas-out/model --no-host-elf
```

`atlasc` captures the reference, applies the FP8/BF16 policy, schedules and
allocates Atlas instructions, and emits `atlas-out/model/compiled/program.bin`.
The output also includes captured and lowered MLIR, packed constants, host
headers and `report.json` with input addresses and program information.
Running the program on `npu_model` or hardware is a separate step.

| Command | Use |
| --- | --- |
| `atlasc doctor` | Check the installed compiler and support files. |
| `atlasc doctor --capture` | Also check the PyTorch capture dependencies. |
| `atlasc demo` | Compile the bundled MLP without installing PyTorch. |
| `atlasc build model.py --no-host-elf` | Compile a PyTorch loader. |
| `atlasc build model.pt2 --no-host-elf` | Compile a saved `torch.export` program. |
| `atlasc build --help` | Show policies, captured-Linalg inputs and other options. |

Outputs default to `atlas-out/<model-name>`; use `-o DIR` for another location.
Choose a fresh directory or use `--force` to replace an earlier `atlasc` build.
Omit `--no-host-elf` to also build a host ELF when a supported RISC-V compiler
is installed. See [setup](docs/setup.md) for source builds and the Python API.

Compilation currently supports static 32×32 tile graphs with FP8/BF16 precision
policies that change results relative to float32. See the
[binding guide](docs/native-merlin-binding.md) for supported graphs and policies.

## Development and documentation

For source builds, dependency versions, advanced flags and optional hardware
tests, see [setup and development](docs/setup.md). CI builds and tests wheels
on pushes and pull requests and makes the artifacts available for download.

## Qwen kernel experiments

Keep a checkout of `feat/qwen-kernels` alongside the installed wheel for the
Qwen scripts. With `uv`, Git and Cargo available, run from the checkout:

```sh
python examples/qwen/run.py --setup
python examples/qwen/run.py
```

This campaign uses the installed Atlas tools with Qwen-specific Merlin graph
policies and numerical model checks. Its setup creates separate capture and
simulation environments. Use it for the attention and SwiGLU benchmarks;
`atlasc build` supports the standalone importer's tile graphs and does not
compile the full Qwen model. See the [Qwen setup guide](examples/qwen/README.md).

The [pretrained model bringup](docs/qwen-model-bringup.md) exports the real
Qwen session, extracts validated PyTorch regions, and benchmarks model-derived
kernel launches. Start with its [reproduction commands](examples/qwen/full_model/README.md).

See the [measured results and compiler recommendations](docs/qwen-results.md)
and the [reproduction instructions](examples/qwen/README.md). The workloads
are plain PyTorch subgraphs of DeepSeek-R1-Distill-Qwen-1.5B.

## Read next

| Guide | Covers |
| --- | --- |
| [Setup and development](docs/setup.md) | Source builds, dependencies and tests. |
| [Compiler pipeline](docs/compiler-pipeline.md) | Virtual SSA, machine IR, scheduling and emission. |
| [Dialect reference](docs/dialect-reference.md) | Operations, verifiers and passes. |
| [Merlin binding](docs/native-merlin-binding.md) | Supported graphs and native compilation. |
| [Host launch](docs/host-launch.md) | Runtime, host ELF and device launch. |
| [Examples](examples/handoff/README.md) | End-to-end compiler flows. |
| [Hardware status](docs/hardware-status.md) | RTL revision, validation and known limits. |
