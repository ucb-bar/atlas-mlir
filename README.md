# Atlas MLIR handwritten implementation

Atlas MLIR is an out-of-tree MLIR dialect and lowering path for the selected
[Atlas RTL revision](docs/source-discrepancies.md). It provides a handwritten
reference for comparing target dialect generation in Merlin. It does not import
or invoke ACT.

The package has two Atlas stages: virtual SSA values before physical placement,
and typed machine operations with selected instruction fields. It can lower a
checked machine stream through the LLVM MLIR dialect to RISC-V object words
using unmodified LLVM. The resulting function is a reset-entry program body,
not a C-callable Atlas function.

## What is implemented

| Area | Current scope |
| --- | --- |
| Machine dialect | 26 parameterized operation classes cover the 99 selected RTL decoder rows; verifiers and `atlas-emit` check fields and source-level word encodings. |
| Virtual dialect | Bounded BF16/FP8 SSA tiles, state, CFG edges, DMA handles, and MXU resource handles; virtual-to-machine lowering assigns the currently supported physical resources. |
| LLVM handoff | Per-instruction LLVM call markers are checked and finalized into one ordered inline-assembly word block. |
| Execution evidence | Bounded instruction programs ran on a selected-source-linked standalone `AtlasCore` ARC model. No integrated SoC or general host runtime result is claimed. |

The [mode census](docs/selected-variant-census.md) reports **99/99 represented
and word-emitted decoder modes, 69/99 with bounded standalone-core semantic
tests, and 0/99 admitted for a full software target**. These counts concern
decoder modes, not every field combination, numerical input, or temporal
interaction. All 99 retain full-qualification blockers. The census and
[observation index](docs/README.md) carry the evidence and remaining work.

## Compiler stages

`Atlas virtual SSA` → `Atlas machine IR` → `LLVM call markers` →
`LLVM inline-assembly block` → `RISC-V object / ELF`

- `--verify-atlas-virtual-stream` checks virtual SSA, state, and CFG edges.
  `--lower-atlas-virtual-to-machine` performs the bounded placement and emits
  typed machine operations.
- `--verify-atlas-generated-schedule` checks the generated stream's chosen
  delay and DMA-wait policy. `--verify-atlas-machine-stream` checks the physical
  state chain, encoding, and control targets.
- `--convert-atlas-to-llvm-calls` retains operation fields as LLVM-dialect call
  markers. `--finalize-atlas-llvm-calls` rechecks them and emits one ordered
  inline-assembly block. Those markers are compiler IR, not runtime calls.
- `mlir-translate` and `llc` produce ordinary RISC-V artifacts containing the
  selected Atlas instruction words. A generic RISC-V CPU cannot execute the
  custom tensor instructions.

The physical `--insert-atlas-delays` and `--schedule-atlas-stream` passes use a
ported timing model. They are separate from virtual-to-machine lowering and
do not establish an RTL-qualified schedule. The
[dialect reference](docs/dialect-reference.md) lists every operation, pass,
tool, allocation limit, and pass extension point. The
[functional-stream format](docs/functional-stream-contract.md) is the separate
instruction-level input for a functional model.

## Build and test

Use MLIR and LLVM tools from the same installation. The recorded local build
used LLVM/MLIR 23.0.0git; see [source selection](#source-selection).

```sh
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
build/bin/atlas-opt test/examples/mxu.mlir
build/bin/atlas-emit test/examples/mxu.mlir
```

CTest runs the portable tests without hardware. Tests that require the selected
RTL, assembler, or standalone core need their corresponding `ATLAS_RTL_ROOT`,
`ATLAS_MODEL_ROOT`, `ATLAS_ASSEMBLER_ROOT`, `ATLAS_ARC_MODEL`,
`ATLAS_ARC_STATE`, `ATLAS_MODELIR_ROOT`, and `ATLAS_LLVM_BIN` paths. A skip is
not execution evidence. For a direct Python test run with another build tree,
set `ATLAS_OOT_BIN_DIR` to its `bin` directory. The source-bound inventory
check and its required source pins are documented in the
[mode census](docs/selected-variant-census.md).

## Examples

| Example | Starting point | What it shows |
| --- | --- | --- |
| [MLP tile](examples/handoff/mlp_tile/01-atlas-machine.mlir) | Atlas machine IR | Fixed 32×32 instruction stream through LLVM, assembly, and ELF. |
| [Attention tile](examples/handoff/attention_tile/01-atlas-machine.mlir) | Atlas machine IR | Fixed attention-like tile stream through the same stages. |
| [Virtual MLP tile](examples/handoff/virtual_mlp/00-atlas-virtual-ssa.mlir) | Atlas virtual SSA | Placement and lowering before LLVM emission. |
| [Captured MLP diagnostic](examples/handoff/captured_mlp/00-linalg.mlir) | Parsed Linalg IR | A bounded capture-to-ELF route with an explicit FP8/BF16 policy. |

The [handoff guide](examples/handoff/README.md) gives the exact regeneration
commands and names every generated Atlas, LLVM, assembly, object, ELF, map, and
manifest file. Generated outputs belong under `out/` or another invocation
output directory. The examples are fixed, bounded fixtures; they do not
establish general MLP or attention model compilation.

## Source selection

| Role | Revision |
| --- | --- |
| Selected RTL | `ucb-bar/atlas-npu` `0079c0541111197741a231c002e3843fa6f545b2` |
| Examined model | local `npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` |
| Recorded LLVM/MLIR build | `a47bddccec30255619bb8c37fa59700e661d4e66` (23.0.0git) |

The RTL and inspected model disagree on several encodings, arithmetic paths,
layouts, and control rules. The [source comparison](docs/source-discrepancies.md)
records those differences. For this selected RTL, the
[column-reduction compatibility decision](docs/selected-rtl-column-reduction-contract.md)
uses the executing 64×16 physical layout while retaining the 32×32
architectural intent as a discrepancy.

## Limits and documentation

This is a handwritten comparison implementation, not a clean-room oracle or a
Merlin-generated compiler.
Selected-core results are bounded diagnostics. Full numerical, timing,
physical-effect, legality, and integrated execution evidence is still needed
before treating the selected configuration as a complete target dialect.
There is no general instruction selector, whole-model compiler, callable Atlas
ABI, or full D/F/M/N qualification in this repository.

Start with the [documentation index](docs/README.md) for the dialect,
pass-development guidance, examples, source decisions, execution observations,
and launch limitations.
