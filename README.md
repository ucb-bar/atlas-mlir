# Atlas MLIR hand-authored reference candidate

This is an out-of-tree ODS/C++ **machine-stage** dialect for one selected Atlas
RTL revision. It is a reviewable reference candidate for comparing Merlin's
generated dialect against an implementation written directly from the selected
RTL and the `npu_model` sources. It was authored with Codex assistance and is
not a clean-room or certified reference. It is not a Merlin compiler, a qualified
executable target dialect, or an Atlas hardware certificate.

The dialect has a typed `!atlas.state` token and 26 parameterized operation
classes covering the **99 selected RTL BitPat rows**: tensor load/store,
DMA load/store/config/wait, both MXUs, VPU arithmetic/reduction/pack/immediate,
XLU transpose, scalar ALU/load/store, CSR, branch/jump/delay, and termination.
It deliberately does not create one MLIR class per channel, unit, or mode.
Verifiers check known physical register, pair, slot, channel, CSR-address,
immediate, and mode limits. Machine operations declare conservative physical
state read/write effects. `atlas-emit` requires a linear state chain and
checks that branches/jumps have a non-redirecting delay-slot instruction.

`atlas-opt` uses MLIR's parser/printer and verifiers. `atlas-emit` emits one
eight-digit hexadecimal 32-bit word per instruction, after checking the whole
flat stream. It uses the RTL decoder's six-bit VR field layout and the selected
RTL's DMA config encoding. The output is an instruction-word listing, not a
binary plus constants or a launchable execution plan.

## Source selection

| Role | Revision |
| --- | --- |
| Selected RTL | `ucb-bar/atlas-npu` `0079c0541111197741a231c002e3843fa6f545b2` |
| Model source examined | local `npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` |
| MLIR/LLVM used for local build | LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`, 23.0.0git |
| Packaging precedents | `ucb-bar/gemmini-mlir` `baseline` `7e7a883` and `stable/agent_spec_v1_mlir_oot` `9e18b840` |

The selected source files are `Instructions.scala`, `IDecode.scala`,
`ScalarDecoder.scala`, and `ScalarCore.scala` under
`src/main/scala/atlas/scalar/`; the model files are
`npu_model/configs/isa_definition.py` and `npu_model/isa.py`. The precise model
revision is a later local branch than the original Merlin v2 plan's inspected
`6c86010` and must be qualified separately. See
[source discrepancies](docs/source-discrepancies.md).

## Build and test

Use an MLIR and LLVM installation from the same build. The following commands
were run with LLVM/MLIR 23.0.0git:

```sh
cmake -S . -B build -G Ninja -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
build/bin/atlas-opt test/examples/mxu.mlir
build/bin/atlas-emit test/examples/mxu.mlir
python -m unittest discover -s test -v
ctest --test-dir build --output-on-failure
ATLAS_RTL_ROOT=/path/to/atlas-npu ATLAS_MODEL_ROOT=/path/to/npu_model-atlas \
  python -m unittest discover -s test -v
```

The source-linked test checks all 99 emitted words against the selected
RTL BitPats and records known model/RTL encoding disagreements. The portable
test also checks exact operand positions, parser/printer round trips, 33
negative verifier cases, and rejection of an invalid state/delay-slot stream. A BitPat
match checks fixed encoding bits; it does not establish hardware legality or
semantic correctness.

## Scope of this candidate

| Obligation | Current evidence |
| --- | --- |
| Custom RTL pattern representation | 50/50 selected custom BitPat rows have a typed parameterized MLIR operation route |
| Scalar/control/CSR representation | 49/49 selected BitPat rows have a typed parameterized MLIR operation route |
| Source-level word emission | 99/99 selected BitPat rows crosschecked at fixed-bit level; valid fields and integrated decode remain unqualified |
| Exact numerical semantics | Not qualified; MXU/VPU and scale behavior need discriminating hardware checks |
| Temporal validity and DMA completion | Not qualified; the token conservatively orders issue only |
| Branch/control behavior | Source-level delay-slot structure checked; integrated branch and halt behavior not qualified |
| Program binary, ABI, execution | Not implemented; no hardware execution claim |

This package should become a golden *comparison reference* only after independent
semantic and hardware tests pass. Merlin's generated dialect and this hand
implementation should consume the same frozen selected target contract; this
repository must not become a hidden second authority for numerical semantics.

No `.merlin` certification or release manifest is included because the
required backend and execution evidence do not yet exist. No remote push has
been made.
