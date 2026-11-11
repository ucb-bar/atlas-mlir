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

## Initial Merlin shared dense path

The `feat/atlas-target` companion worktrees add a small diagnostic path from
explicit ordered FP8/BF16 Linalg through Merlin's existing checked dense loops
to Atlas virtual SSA and the existing compiler. It admits full 32×32 MXU0 tiles,
including continued K reductions and multiple output tiles. Atlas owns the
physical tile layout, numerical policy, instruction expansion and scheduling.
The source must select `mxu0-e4m3-ordered-bf16-finite-v1` and
`finite-normal-e4m3-or-positive-zero`; `pack_inputs` checks actual input codes.
This path performs no implicit conversion from f32, padding or reassociation.

Use these compatible baselines:

| Component | Revision |
| --- | --- |
| Merlin before the local changes | `d56133b168bbee3d73cb8b4304d3622d4191fc6a` |
| Atlas MLIR before the local changes | `c9f50c3310a68d9eb8fa318e68492896a5a45737` |
| `npu_model` for execution | `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5` |
| RTL identity selected by Atlas MLIR | `0079c0541111197741a231c002e3843fa6f545b2` |

The model pin is deliberate: `c119e54` changed PACK row layout, and the older
`rtl-match` tip uses different DMA encoding. The selected model matches both
the instruction encoding and the PACK layout expected by this compiler. This
is model execution evidence; this change has not been executed on RTL.

From this Atlas worktree, using the Merlin environment (tested with xDSL 0.73.0):

```sh
export MERLIN_ROOT=/home/jeremy/Projects/merlin-atlas-target
export MERLIN_PYTHON=/home/jeremy/Projects/merlin/.venv/bin/python
export ATLAS_BIN="$PWD/out/build/atlas-ready/compiler/bin"
export MODEL_ROOT=/home/jeremy/Projects/npu-model-atlas-target
export MODEL_PYTHON=/home/jeremy/Projects/npu_model/.venv/bin/python
export PYTHONPATH="$MERLIN_ROOT/src:$PWD/python:$PWD"

"$MERLIN_PYTHON" -m atlas_mlir_native.shared_dense \
  test/examples/native_mxu0_ordered_linalg.mlir --entry mxu0_tile \
  --destination "$MERLIN_ROOT/out/build/atlas-single" \
  --atlas-bin-dir "$ATLAS_BIN" --pipeline optimized

"$MERLIN_PYTHON" -m pytest -q test/test_shared_dense.py
"$MERLIN_PYTHON" tools/check_shared_dense.py \
  --output-dir "$MERLIN_ROOT/out/build/atlas-regression" \
  --atlas-bin-dir "$ATLAS_BIN" --model-root "$MODEL_ROOT" \
  --model-python "$MODEL_PYTHON"
```

Choose fresh output directories. Compilation writes the source, actual checked
logical schedule, virtual/machine IR, final bytes and a manifest. The regression
driver checks the model revision, compiles each program once, independently
decodes its final bytes, and executes different runtime inputs. It compares
every output byte and checks that input and guard bytes are preserved. Its
finite-range FMA oracle uses exact integer arithmetic, independently of both
the compiler and the model. This is a deterministic test driver, with no
compiler generation or optimization agent loop.

Measured model cycles include the complete diagnostic program and transfers:

| M × K × N | Source-order words / cycles | Optimized words / cycles |
| --- | --- | --- |
| 32 × 32 × 32 | 37 / 3,271 | 42 / 3,183 |
| 32 × 64 × 32 | 53 / 4,921 | 66 / 4,712 |
| 64 × 32 × 64 | 125 / 11,362 | 146 / 10,122 |
| 64 × 64 × 32 | 90 / 8,163 | 108 / 7,344 |

The 32×32 shared result is byte-identical to the direct Atlas virtual matmul
reference in both modes. The driver also tests direct Atlas matmul→ReLU and
matmul→ReLU→PACK→matmul with an observable intermediate output. These reference
chains establish the baseline; they do not yet implement shared floating fusion.

The main-based shared graph path now also lowers standalone BF16 ReLU,
matmul→ReLU and matmul→ReLU→PACK→matmul, preserving intermediate outputs.
`tools/check_shared_graph.py` exercises these at 32×32 and 64×64 in both modes.
Its PACK witnesses establish the declared seven-code intermediate domain;
arbitrary BF16 truncation remains unqualified. The graph adapter retains all
source domain premises in its manifest. Add/subtract/multiply source recognition
exists in Merlin, but this initial Atlas graph adapter does not lower those forms.

Rebuild this branch's tools before using either driver: `atlas-emit` now exports
`--issue-timing-json`. Every shared compilation binds its actual machine words to
Merlin `schedule.time_annotation` / `schedule.overlap` decisions and checks the
existing timing provider's pair issue distances. The OOT verifier still checks
physical reservations, DMA completion and control flow. No second latency model
is introduced; block-relative fixed-latency annotations are not end-to-end cycles.

```sh
"$MERLIN_PYTHON" tools/check_shared_graph.py \
  --output-dir out/build/shared-graph-preflight \
  --atlas-bin-dir "$ATLAS_BIN" --model-root "$MODEL_ROOT" \
  --model-python "$MODEL_PYTHON"
```

The 32×32 two-layer chain uses 111 words / 6,792 cycles in source order and
120 words / 5,090 cycles optimized. The 64×64 chain uses 456 / 27,131 and
491 / 20,186 respectively. Both retain hidden and final outputs. These are model
measurements of the shared diagnostic adapter with existing Atlas physical
lowering, not generated-backend performance. The direct 32×32 Atlas comparator
matches those counts.

The companion Merlin `examples/atlas/shared-passes.md` documents a standalone
40-member compiler library, the reproducible preflight and existing generation
admission gates. It runs without capture, reference compiler or grader imports.
Tails, arbitrary graph-domain qualification, reductions and final RTL qualification
remain outside the initial scope. FP4/FP6 grouped storage and scalar codecs in
Merlin do not grant those formats to Atlas hardware. No generation loop is run.

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
