# Atlas MLIR

Atlas MLIR compiles supported PyTorch and Linalg workloads into programs for the
Atlas NPU. The pip package includes `atlasc`, the native `atlas-opt` and
`atlas-emit` tools, and the Atlas plugin for Merlin.

## Quickstart

Download and unzip the Linux x86-64 wheel artifact from the **Package** GitHub
Actions workflow, then run:

```sh
python -m pip install ./atlas_mlir-*.whl
atlasc demo
```

The bundled MLP demo needs no PyTorch or local LLVM installation. It writes the
device binary, MLIR, host headers and a report to `atlas-out/demo`. Running the
compiled program requires an Atlas model or device.

## Compile a PyTorch model

Install capture support and compile a loader:

```sh
python -m pip install './atlas_mlir-0.1.0-py3-none-linux_x86_64.whl[capture]'
atlasc build model.py
```

The loader defines `get_model_and_inputs()`:

```python
import torch


def get_model_and_inputs():
    model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(),
                                torch.nn.Linear(32, 32)).eval()
    return model, (torch.ones(32, 32),)
```

Exported `.pt2` files are also accepted. Outputs go to `atlas-out/<model-name>`;
use `-o DIR` to choose a directory. `atlasc doctor --capture` checks the setup.

Compilation currently supports static 32×32 tile graphs with FP8/BF16 precision
policies that change results relative to float32. See the
[binding guide](docs/native-merlin-binding.md) for supported graphs and policies.

## Development and documentation

For source builds, dependency versions, advanced flags and optional hardware
tests, see [setup and development](docs/setup.md). CI builds and tests wheels
on pushes and pull requests and makes the artifacts available for download.

| Guide | Covers |
| --- | --- |
| [Setup and development](docs/setup.md) | Source builds, dependencies and tests. |
| [Compiler pipeline](docs/compiler-pipeline.md) | Virtual SSA, machine IR, scheduling and emission. |
| [Dialect reference](docs/dialect-reference.md) | Operations, verifiers and passes. |
| [Merlin binding](docs/native-merlin-binding.md) | Supported graphs and native compilation. |
| [Host launch](docs/host-launch.md) | Runtime, host ELF and device launch. |
| [Examples](examples/handoff/README.md) | End-to-end compiler flows. |
| [Hardware status](docs/hardware-status.md) | RTL revision, validation and known limits. |
