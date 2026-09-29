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
RTL's DMA config encoding. The registered `--convert-atlas-to-llvm` pass
lowers a flat validated stream to `llvm.func @atlas_program` with one
side-effecting `llvm.inline_asm` block containing the selected words in order.
One block keeps direct branch targets and the next delay-slot instruction
adjacent through LLVM lowering. The pass rejects JALR and direct targets
outside the block; it does not resolve dynamic control flow.

This is physical instruction lowering for the selected Atlas RISC-V target.
The inline assembly words are **not executable on a generic host CPU** and an
ELF object is **not a qualified Atlas end-to-end runtime**. The void function
has no Atlas launch ABI, input/output plan, register-save policy, memory
initialization, or completion/drain protocol. Two test programs have diagnostic
execution evidence on a CIRCT ARC model of `AtlasCore` rebuilt from a fresh
elaboration of the selected Atlas source copy.
The tests load the extracted instruction words through ModeLIR's existing
TileLink driver; they do not call the void function using a C ABI.
In particular, fixed scalar-register instructions may alter the return-address
or other ABI registers if called as an ordinary RISC-V function. Atlas's PC
uses an instruction index and shifts encoded branch byte displacements by one;
ordinary RISC-V object linking or execution does not certify equivalent
branch behavior. No ACT compiler package or executable is used by this repo.

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
cmake -S . -B build -G Ninja -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/path/to/test-python
cmake --build build --parallel 2
build/bin/atlas-opt test/examples/mxu.mlir
build/bin/atlas-emit test/examples/mxu.mlir
build/bin/atlas-opt --convert-atlas-to-llvm test/examples/mxu.mlir > build/mxu-llvm.mlir
mlir-translate --mlir-to-llvmir build/mxu-llvm.mlir > build/mxu.ll
llc -mtriple=riscv32-unknown-elf -filetype=obj build/mxu.ll -o build/mxu.o
llvm-readelf -h build/mxu.o
llvm-objdump -d build/mxu.o
python -m unittest discover -s test -v
ctest --test-dir build --output-on-failure
ATLAS_RTL_ROOT=/path/to/atlas-npu ATLAS_MODEL_ROOT=/path/to/npu_model-atlas \
ATLAS_LLVM_BIN=/path/to/llvm-install/bin \
ATLAS_ASSEMBLER_ROOT=/path/to/atlas-npu/baremetal \
  python -m unittest discover -s test -v
```

To run the optional core-model checks, also set `ATLAS_ARC_MODEL` to the
selected `.so`, `ATLAS_ARC_STATE` to its arcilator state JSON,
`ATLAS_MODELIR_ROOT` to the ModeLIR checkout, and `ATLAS_RTL_ROOT` to the
selected RTL checkout. The current diagnostic run passed 22/22 Python test
methods with these paths supplied: the typed branch program executed one
delay slot, while a changed branch target produced a different checked state;
the typed DMA loopback performed four reads and four writes, matched 32/32
output words, and preserved all 32 input words and three guard words. Both
programs' object words matched the emitter and independent selected assembler.
For the current standalone-core diagnostic, all 69 shared Atlas Scala files
in the selected checkout and the Chipyard generator copy matched byte for
byte. Fresh elaboration produced FIRRTL SHA-256
`fefa711dba44498317573ee5af2cfd82edb674cdfff1505318ea93e896fc123d`,
equal to the FIRRTL used to extract the 78-module `AtlasCore` closure. A
fresh ARC build from that closure produced state JSON SHA-256
`db2d8ae3c8a4ce6a417b0c691be1d446f1c0be95efe5607a251a6946a80de9e3`.
All 22 Python tests passed again against the rebuilt shared library. This
links the diagnostic standalone core to the selected Atlas source bytes; it
does not qualify the full integrated SoC, all Chipyard dependencies, or
physical RTL simulation.
The selected Atlas commit records submodule revisions `9a0cc09c` for
`sp26-fp-units` (which supplies `E4M3FMA`) and `ab0cf6f5` for `fpex`.
Fresh checkouts at those pins matched all 45 tracked `sp26-fp-units` files
and all 26 `fpex` files present in the Chipyard copy byte for byte; the three
absent `fpex` files are tests. Numerical claims remain scoped to the exact
rebuilt model and tested inputs. Other Chipyard dependencies and full SoC
behavior have not been qualified.
The optional ModeLIR driver imports NumPy; configure CTest with the Python
environment that contains it when enabling the core-model checks.

The source-linked test checks all 99 emitted words against the selected
RTL BitPats and records known model/RTL encoding disagreements. The portable
test also checks exact operand positions, parser/printer round trips, 33
negative verifier cases, and rejection of an invalid state/delay-slot stream. A BitPat
match checks fixed encoding bits; it does not establish hardware legality or
semantic correctness.

The LLVM conversion tests check a single ordered side-effecting assembly
block, 98/99 selected pattern variants with statically bounded control flow,
explicit rejection of JALR, escaping targets, unsupported nested operations,
and broken state/delay-slot chains. A smoke test translates the MXU example to
LLVM IR and assembles an ELF32 RISC-V object whose disassembly contains the
four selected words. This checks lowering and object emission only.

The seeded encoding check uses seed `0xA71A5`, 12 passes over the 99 selected
patterns, and 1,188 positive words (1,135 distinct pattern/word pairs in the
local Python 3.12 run). It varies register fields, slots, channels, and
immediates within the verifier's declared domain. This is source-level
encoding coverage, not 1,188 hardware-executed instruction cases.

## Scope of this candidate

| Obligation | Current evidence |
| --- | --- |
| Custom RTL pattern representation | 50/50 selected custom BitPat rows have a typed parameterized MLIR operation route |
| Scalar/control/CSR representation | 49/49 selected BitPat rows have a typed parameterized MLIR operation route |
| Source-level word emission | 99/99 selected BitPat rows crosschecked at fixed-bit level; valid fields and integrated decode remain unqualified |
| Exact numerical semantics | Not qualified; MXU/VPU and scale behavior need discriminating hardware checks |
| Temporal validity and DMA completion | Not qualified; the token conservatively orders issue only |
| Branch/control behavior | Typed branch program and changed-target mutation ran on the rebuilt selected-source-linked standalone `AtlasCore` ARC model; integrated SoC behavior remains unqualified |
| LLVM dialect/object lowering | Registered pass and ELF32 RISC-V smoke test for statically bounded flat streams; JALR rejected |
| DMA movement | Typed loopback ran on the rebuilt selected-source-linked standalone core model with 32/32 output words and input/guard preservation; general DMA timing and integrated behavior remain unqualified |
| Program binary, ABI, execution | Extracted object words run through an external diagnostic driver; no Atlas launch ABI, constants package, or qualified hardware execution claim |

This package should become a golden *comparison reference* only after independent
semantic and hardware tests pass. Merlin's generated dialect and this hand
implementation should consume the same frozen selected target contract; this
repository must not become a hidden second authority for numerical semantics.

No `.merlin` certification or release manifest is included because the
required backend and execution evidence do not yet exist. No remote push has
been made.
