# pip package: PyTorch model to Atlas program in one install and one command

Branch: `feat/pip-package`, from `handwritten-implementation`.

## Goal

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, see below
pip install atlas_mlir-0.1.0-py3-none-linux_x86_64.whl
atlasc build models/mlp.py        # -> program.bin, host ELF, report
```

Model users never install LLVM, MLIR, Rust or a C++ compiler. Compiler developers keep
the CMake build in the root README.

## Why

Following the docs today takes about 8 hand-run commands from a model to a host ELF, more
than 20 environment variables, three Python environments, the conda C++ compiler (the
system one makes `atlas-opt` segfault), a 13 GB clang 23 source build, and hand-written
files that must agree with each other (ABI slots, `layout.json`, `host.c`, repeated
`native-compile`/`native-verify` flags). On this machine that is about 20 GB of tools.

## Three ideas

1. **One install.** A wheel carries prebuilt `atlas-opt` and `atlas-emit` with the shared
   libraries they load, the driver, the `tools/` scripts it calls, and the host runtime C.
   pip installs the remaining Python dependencies. No environment variables.
2. **One command.** `atlasc build` owns the glue: capture, policy, compile, DRAM layout,
   host files, output directory. Nothing is hand-written to match something else.
3. **Wrap, don't rewrite.** The driver calls the existing `tools/` scripts unchanged, so
   their manifests, hash checks and refusals stay in force.

## Fit with Chipyard

Anything substantial is checked against Chipyard's conda environment
(`conda-reqs/chipyard-base.yaml`, `chipyard-extended.yaml`, `riscv-tools.yaml` on
ucb-bar/chipyard `main`) before it is added.

| Need | Chipyard has | Decision |
| --- | --- | --- |
| Python | `python>=3.9,<3.11` | Support Python 3.10. |
| RISC-V C compiler for the host ELF | `riscv-tools` (`riscv64-unknown-elf-gcc`) | Use it. The only non-pip dependency, and optional: without it atlasc stops at `program.bin` and the host headers. |
| numpy | yes | Use it. |
| cmake, ninja, GCC 15 | yes | Wheel builds only. GCC 15 is also the compiler conda-forge's MLIR 23.1.2 was built with. |
| torch, torch-mlir | no | Needed to read a PyTorch model; torch-mlir is what model2mlir's `fx_importer` backend already runs. |
| xdsl 0.65.0, safetensors | no | Already required by the Linalg importer (`tools/requirements-captured-mlp.txt`). |

Rejected: `ziglang` (a 96 MB pip-installable clang) for the host ELF, because riscv-tools
covers it.

## Entry points

| Input | Command |
| --- | --- |
| Loader `.py` defining `get_model_and_inputs()` (the existing convention, e.g. `test/models/directed_mlp32.py`) | `atlasc build model.py` |
| `.pt2` from `torch.export.save` | `atlasc build model.pt2` |
| Python | `atlas_mlir.compile(model, (x,), "out/mlp")` |
| Already captured Linalg + argument manifest + weights | `atlasc build --linalg f.mlir --arguments a.json --weights w.safetensors [--inputs x.safetensors]` |
| Toolchain check | `atlasc doctor` |

A plain weights-only `.pt` cannot work: it has no graph.

## What `atlasc build` does

1. Load the model and example inputs; `torch.export`, then torch-mlir's FX importer with
   parameters kept as function arguments, lowered to `linalg-on-tensors`. No
   decomposition table, so `aten.linear` becomes the zero-filled matmul plus one bias-add
   generic the importer admits. Writes `linalg.mlir`, `arguments.json`,
   `weights.safetensors` (the same three files model2mlir writes) and the example inputs.
2. Policy: a named preset (`fp8`: every contraction on MXU0, pack scale 127) or a policy
   JSON path, written to `policy.json`.
3. `tools/compile_atlas_linalg_tile.py`: virtual SSA, `atlas-opt --atlas-compile`,
   `atlas-emit`, `program.bin`.
4. Derive the DRAM layout from the import manifest, then `atlas_boot_pack.py`,
   `atlas_captured_host_data.py`, `atlas_host_embed.py`.
5. Pack the example inputs with the importer's own packers into `atlas_run_inputs.h`, and
   link `runtime/atlas_run.c` (load, stage constants and inputs, launch once, keep the
   outputs in `atlas_run_outputs`, report `atlas_run_status`) with riscv-tools GCC.
6. Write `report.json`.

The output directory (default `atlas-out/<name>`) must be empty, as the tools require;
`--force` replaces it only if an earlier atlasc build made it.

## Default compile route

The standalone Linalg importer, because everything it needs installs from pip. The
Merlin-native route needs Merlin (not on PyPI; the `merlin` name there is an unrelated
LLNL project), model2mlir (not on PyPI) and Rust for `native-build`. It becomes
`atlasc build --engine native` once Merlin ships a wheel and the e-graph bridge is built
in CI.

One importer change was needed: it now also admits torch-mlir's ReLU,
`select(cmpf ugt(x, +0), x, +0)`, which maps -0 to +0 exactly as Atlas ReLU does. The
torch-mlir capture of `directed_mlp32` imports to the same virtual program as the
model2mlir capture (`test/examples/captured_mlp32_torch_mlir_linalg.mlir`).

## Wheel build

```sh
E=~/miniconda3/envs/atlas-mlir-build   # conda-forge mlir 23.1.2 and its GCC 15
PATH=$E/bin:$PATH SKBUILD_CMAKE_ARGS="-DMLIR_DIR=$E/lib/cmake/mlir;-DLLVM_DIR=$E/lib/cmake/llvm;-DCMAKE_C_COMPILER=$E/bin/x86_64-conda-linux-gnu-cc;-DCMAKE_CXX_COMPILER=$E/bin/x86_64-conda-linux-gnu-c++" \
  python -m pip wheel . --no-deps -w dist
```

- scikit-build-core configures CMake with `ATLAS_ENABLE_TESTS=OFF` and installs only the
  `atlas-wheel` component: the two tools under `atlas_mlir/bin` (`RPATH=$ORIGIN/../lib`),
  every non-system library they load copied once under its soname and stripped into
  `atlas_mlir/lib`, and the support files under `atlas_mlir/share` in the source layout.
- An editable install (`pip install -e .`) builds nothing; the driver then uses
  `build/bin` (or `ATLAS_OOT_BIN_DIR`) and the source tree.
- Tag `py3-none-linux_x86_64`: the tools do not use the Python ABI. PyPI needs a
  manylinux build (CI in a manylinux_2_28 image); GitHub Releases do not.

Measured: 76 MB download, 220 MB installed (`libLLVM.so` 168 MB, ICU data 32 MB,
`atlas-opt` 9 MB). On top come CPU torch (~180 MB download) and torch-mlir (76 MB).
Installing torch without the CPU index pulls the CUDA build and about 4 GB of NVIDIA
libraries; Atlas does not need them.

## Status

Done and checked on this machine:

- Wheel built; installed into a fresh Python 3.10 venv with no conda on `PATH`; `atlasc
  doctor` and `atlasc build` work from outside the repository, and every non-system
  library resolves inside the wheel.
- Loader `.py`, `.pt2` and Python API builds of `directed_mlp32` give the same 101-word
  `program.bin`; with the fixture policy it is byte-identical to compiling the model2mlir
  capture.
- Host ELF from riscv-tools-style GCC (`riscv64-unknown-elf-gcc` 13.2): RV64, entry
  `0x80000000`. Built and inspected, not run: a plain Spike run cannot execute Atlas MMIO.
- Tests: `test/test_atlasc.py` (driver), two new cases in
  `test/test_captured_mlp_import.py` (torch-mlir ReLU form and its mutations).

Next:

1. Expected outputs and a precision report. `tools/atlas_virtual_evaluator.py` needs
   npu_model (Python 3.14) for MXU numerics, so this belongs with `atlasc run`, not the
   wheel.
2. Capture beyond the MLP: torch-mlir's forms for softmax and attention have not been
   compared with what the importer admits.
3. CI wheel build and a GitHub Release.
4. `--engine native` through Merlin; `atlasc run` on npu_model and the RTL core.
5. Root README quick start (it still names the old `atlas-mlir-native-support`
   distribution).

## Limits carried over

Static 32x32 tiles and the operations the importer accepts (zero-filled matmuls, weight
transposes, bias add, ReLU, exp, splat scale, row max/sum with broadcasts, row
subtract/divide, up to 16 outputs). The precision policy changes results relative to f32.
The host ELF is diagnostic: no bitstream is bound and DRAM is assumed coherent.

## Open decisions

- Where to publish: GitHub Releases on ucb-bar (private, `pip install --find-links`) or PyPI.
- Whether `torch` stays a hard dependency or moves to an extra, so a Chipyard environment
  that only compiles captured Linalg does not pull it in.
- Merlin's package name, before the native route can depend on it.
