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

The repository also ships a Python package (`atlas-mlir-native-support`) that
plugs Atlas into Merlin's native compiler, and a small C runtime for the host
side.

## Build

Use MLIR and LLVM from one installation (tested with LLVM/MLIR 23.0.0git,
`a47bddccec30255619bb8c37fa59700e661d4e66`).

```sh
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
pip install -e .        # optional: Merlin native binding
```

CTest runs the portable tests with no hardware. Tests that run the RTL need
`ATLAS_RTL_ROOT`, `ATLAS_MODEL_ROOT`, `ATLAS_ASSEMBLER_ROOT`, `ATLAS_ARC_MODEL`,
`ATLAS_ARC_STATE`, `ATLAS_MODELIR_ROOT`, and `ATLAS_LLVM_BIN`; without them they
skip. To run the Python tests against another build, set `ATLAS_OOT_BIN_DIR` to
its `bin` directory.

## Quick start

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
| [Guarded PACK proposal](docs/guarded-pack.md) | Opt-in all-scale conversion with explicit finite-source contracts and conservative fallback. |
| [Native Merlin binding](docs/native-merlin-binding.md) | Compiling graphs through Merlin. |
| [Host launch](docs/host-launch.md) | Host runtime, memory map, mailbox, board builds. |
| [Hardware status](docs/hardware-status.md) | Which instructions have run on the RTL, and known RTL/model differences. |

## Limits

This is not a general model compiler. It handles static 32×32 tiles and the
graphs listed in the binding doc, with FP8/BF16 precision policies that change
results relative to f32. Programs have run on a standalone Atlas core built
from the pinned RTL, not on an integrated Shuttle + Atlas SoC or an FPGA.
