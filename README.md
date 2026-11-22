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

The TinyLlama examples use Merlin's `feat/tinyllama` branch
(https://github.com/ucb-bar/merlin/pull/63), which uses the model2MLIR
quantization adapter and the Atlas native provider together. See the
whole-model run instructions in that pull request
for the matching repositories, adapter, bundle regeneration, and commands.

## Build

Use MLIR and LLVM from one installation (tested with LLVM/MLIR 23.0.0git,
`a47bddccec30255619bb8c37fa59700e661d4e66`). The build also needs CMake, Ninja
(or `-G "Unix Makefiles"` in place of `-G Ninja`), and a C/C++ compiler.

```sh
python -m pip install -e '.[test]' -r tools/requirements-virtual-evaluator.txt
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DATLAS_NPU_MODEL_ROOT=/path/to/npu-model
cmake --build build --parallel 2
mkdir -p out/work
export TMPDIR="$PWD/out/work"
ctest --test-dir build --output-on-failure
```

`atlas-opt`, `atlas-emit`, and CTest need the build compiler's C++ runtime.
Set `LD_LIBRARY_PATH=/path/to/compiler/lib:$LD_LIBRARY_PATH` when it differs
from the system runtime; check with `build/bin/atlas-opt --version`.

`ATLAS_NPU_MODEL_ROOT` must name a usable `npu_model` checkout (tested at
`b97b2f5` with Python 3.14). Configure checks its assembler and RTL numerics
imports using `Python3_EXECUTABLE`; CTest adds that path to `PYTHONPATH`.
Install its dependencies before configuring. Optional execution tests still
require their explicit environment settings.

CTest runs the portable tests with no hardware. Tests that run the RTL need
`ATLAS_RTL_ROOT`, `ATLAS_MODEL_ROOT`, `ATLAS_ASSEMBLER_ROOT`, `ATLAS_ARC_MODEL`,
`ATLAS_ARC_STATE`, `ATLAS_MODELIR_ROOT`, and `ATLAS_LLVM_BIN`; without them they
skip. To run the Python tests against another build, set `ATLAS_OOT_BIN_DIR` to
its `bin` directory.

The full native/evaluator suite also needs the selected Merlin source and an
unpacked `npu_model` source tree with its Python dependencies. In particular,
the arithmetic checks read `tests/rtl/arithmetic.json`; a model wheel alone
does not include that reference fixture. Select these before running CTest:

```sh
export MERLIN_NATIVE_ROOT=/path/to/merlin
export NPU_MODEL_ROOT=/path/to/npu_model
export PYTHONPATH="$MERLIN_NATIVE_ROOT/src:$MERLIN_NATIVE_ROOT:$NPU_MODEL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

These are test dependencies; ordinary compilation does not consume the model
or its reference results. Hardware and model-execution tests remain opt-in.

The full native C export tests also need Clang with RISC-V target support, both
on `PATH` and in `ATLAS_LLVM_BIN`. The native independence test uses
`merlin-targetgen` beside the selected Python interpreter or on `PATH`.

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

For Baremetal-IDE, start with the [single-image layer examples](examples/handoff/resident_layers/README.md).
They produce C/H, device binaries, constants and buffer metadata. The platform
application supplies the driver, startup and linker script, then builds the
host ELF. The attention export contains program/data declarations; the matrix
products also provide callable C wrappers.

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

## Tensor applications and C integration

The [C integration guide](docs/baremetal-c-integration.md) covers the generated
C/H pair, parameter capture, platform callbacks and Baremetal-IDE build boundary.
The [instruction memory guide](docs/instruction-memory.md) describes final size
checks, long branches and measured reduction unrolling. These paths preserve the
selected numerical policies; software/model checks do not qualify an FPGA build.

The [already-quantized intake guide](docs/already-quantized-intake.md) records byte-preserving code transport and the paired BF16 readout factor domain.

The [exact-float input layout guide](docs/float-input-layout.md) describes folding
private slices and broadcasts into checked input packing without changing arithmetic.

The [exponent page cache guide](docs/exponent-page-cache.md) describes reusing
invariant quantization scales in VMEM across output tiles.

The [batched operand panel guide](docs/batched-operand-panels.md) covers independent
batch reuse, bounded DMA address generation and measured complete-layer results.

## Small on-chip deployment and bench handoff

Use the configured compiler environment and an authenticated small export:

```sh
atlasc deploy EXPORT --jtag -o OUT
python3 OUT/kit/bench/run_kit.py OUT/kit --dry-run
python3 OUT/kit/bench/check_rung.py kit --bundle OUT/kit --dry-run
```

`OUT` must be fresh. Deployment uses `atlasc build`, an independent nonzero
reference, and bound EXACT plus an executed output-changing mutation before
linking and packaging. Both file-only checks must pass. The qualified scope is
small Linear plus ReLU exports at K32 and K64: host code, data, zero-fill, heap,
and stack must fit 64 KiB, and device storage must fit 786,432 bytes. There is no
DRAM loading, paging, or larger-model slicing. See the
[deployment guide](docs/deployment.md) for compiler prerequisites and limits.

Packaging and dry checks grant no hardware authorization or silicon evidence.
`atlasc run OUT` is the live run: fresh preflight and one rung on the chip
(`--negative` selects the expected-fail rung). `atlasc run OUT --dry-run`
repeats the file-only checks without opening transport. The bench ladder below uses the kit's `run.sh`, not
`atlasc run`. The bench handoff uses the packaged runner,
operator procedure at `OUT/kit/README.md`, and
[deployment contract](docs/deployment.md). Every step below needs the bench
owner's explicit word for that step, fresh reviewer GO, exclusive ownership, a
fresh UTC stamp, and a fresh authenticated preflight. Preserve supply settings.
Run the ladder in this order:

0. Frozen Kit A action control with its own runner, original clock metadata,
   and `--baud 384615`. Proceed only on `KIT PASS` with runner 0 and its
   checker `PASS` with rc 0. Do not rebuild or relabel the frozen control.
1. K32 positive. Then stop for a decision.
2. K32 negative.
3. K64 positive.
4. K64 negative.

Stop at the first unexpected result, including a preflight refusal or a
non-empty process check. Do not retry. Preserve all output and receipts.
Before step 0, a dry connect (OpenOCD with the kit's `bench/atlas-soc.cfg`,
`halt`, `reg pc`, and `shutdown`, no load or resume) must halt the hart and
print a PC. One known recovery is allowed, with the bench owner's word: if it
prints `unable to halt hart 0`, unplug the USB cable, power-cycle the board,
replug the USB cable, and rerun the dry connect. Continue only if it then halts
and prints a PC.

Before any hardware step, stage each selected bundle and verify its inventory:

```sh
(cd OUT/kit && sha256sum -c SHA256SUMS)
```

Before each step, run the process check as its own command. It must print only
`pgrep rc=1`; any match means stop and establish ownership:

```sh
pgrep -fa '[r]un_full|[c]hain22|[o]penocd'; echo "pgrep rc=$?"
```

For the control, `KITA_ROOT` is the frozen Kit A staging root (it holds
`bench/run_kit.py` and `out/kitA-action`), `KITA_CHECKER` is its desk checker
`check_rung.py`, `PRE` is a fresh directory for terminal captures, and
`NOV_OPENOCD` is the approved executable's absolute path:

```sh
cd "$KITA_ROOT"
python3 bench/run_kit.py out/kitA-action --openocd "$NOV_OPENOCD" \
  --logs "runs/action-100mhz-$RUN_STAMP" --baud 384615 \
  > "$PRE/kitA-action-terminal.txt" 2>&1; echo "rc=$?" >> "$PRE/kitA-action-terminal.txt"
python3 "$KITA_CHECKER" action "$PRE/kitA-action-terminal.txt" \
  --logs "$KITA_ROOT/runs/action-100mhz-$RUN_STAMP" --run-stamp "$RUN_STAMP" \
  --kit-root "$KITA_ROOT" --openocd-path "$NOV_OPENOCD"; echo "checker rc=$?"
```

The capture must read `KIT PASS UART and dumped bytes match frozen reference`
then `rc=0`, and the checker must print `PASS` first and `checker rc=0`.

The packaged runner's preflight refuses when any other live command line
contains `openocd`, including a shell that wrote the path in the same command.
Write the exports for one new-kit step into an env file, as a separate command,
in a directory whose path does not contain that word, with a fresh stamp for
every step:

```sh
cd "$BENCH"
cat > env.sh <<EOF_ENV
export NOV_OPENOCD=/path/to/approved/openocd
export BUNDLE=$BENCH/k32
export RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EOF_ENV
```

Then, from `$BENCH`, preflight, run the process check alone, launch, and
check, each as its own command:

```sh
. ./env.sh; (cd "$BUNDLE" && sha256sum -c --quiet SHA256SUMS) \
  && python3 "$BUNDLE/bench/run_kit.py" "$BUNDLE" --dry-run \
  && python3 "$BUNDLE/bench/check_rung.py" kit --bundle "$BUNDLE" --dry-run
pgrep -fa '[r]un_full|[c]hain22|[o]penocd'; echo "pgrep rc=$?"
. ./env.sh; sh "$BUNDLE/run.sh" "$BUNDLE" "$RUN_STAMP"; echo "wrapper rc=$?"
. ./env.sh; python3 "$BUNDLE/bench/check_rung.py" kit \
  "$BUNDLE/runs/kit-$RUN_STAMP-terminal.txt" --bundle "$BUNDLE" \
  --logs "$BUNDLE/runs/kit-$RUN_STAMP" --run-stamp "$RUN_STAMP" \
  --openocd-path "$NOV_OPENOCD"; echo "checker rc=$?"
```

The runner captures directly to fresh paths; do not use `tee`, and do not run
`tail -f` or `less` on an OpenOCD log during a step. Run the checker on the same
host with the same absolute `$BUNDLE`: it compares the recorded commands
exactly. The wrapper stops a runner that outlives its own bounded waits and
records `KIT TRANSPORT_ERROR` with rc 2. A positive requires
wrapper rc 0 and checker rc 0. For a negative step, add `--elf negative/chip.elf`
to the run_kit.py preflight, `--negative` to both checker commands, and
`negative` after the stamp in the `run.sh` command. It requires wrapper rc 1
with a `KIT FAIL UART hash mismatch` line naming each changed output, checker
rc 0, completed status zero, and every output byte matching its bound changed
prediction. A timeout, transport failure, missing completion, or incomplete
capture never qualifies. Deployment failures also stop: preserve
`deploy-status.json` and logs, diagnose the failing stage, and use a fresh
output directory for a corrected deployment. Never edit reference bytes or
receipts.
