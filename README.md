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
has no callable Atlas ABI, register-save policy, dynamic arguments, or
qualified completion/drain protocol. A separate bounded
[reset-entry capsule](docs/atlas-launch-abi-gap.md) now declares fixed DRAM
input/output regions and checked IMEM words for one standalone program; it
does not make the LLVM function C-callable. Multiple test programs have diagnostic
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
[source discrepancies](docs/source-discrepancies.md). The source-bound
[selected variant census](docs/selected-variant-census.md) tracks all 99
decoder modes and separate required, admitted, represented, emitted, bounded
semantic, standalone execution, and blocker counts.

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

CTest binds the Python tests to the `atlas-opt` and `atlas-emit` binaries in
its own CMake build directory. For a direct Python invocation against another
build tree, set `ATLAS_OOT_BIN_DIR` to that tree's `bin` directory.

To run the optional core-model checks, also set `ATLAS_ARC_MODEL` to the
selected `.so`, `ATLAS_ARC_STATE` to its arcilator state JSON,
`ATLAS_MODELIR_ROOT` to the ModeLIR checkout, and `ATLAS_RTL_ROOT` to the
selected RTL checkout. The current diagnostic run passed 66/66 Python test
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
All 26 Python tests passed again against the rebuilt shared library. This
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
four selected words. A separate hand-authored 35-word MXU0 stream also
matched the selected assembler and LLVM object words, then executed on the
selected-source-linked standalone core for four sparse/control input cases.
A 42-word variant stored both BF16 register halves; a weight at logical output
column 20 appeared in the second half at checked halfword index 4.
The hand-authored 36-word VRELU stream matched the independent selected
assembler and the RISC-V object bytes, then executed two dense BF16 panels
through both register halves on the same selected-source-linked standalone
core. All 1,024 cells per panel matched a small independent finite-value
reference; an otherwise identical VMOV-mode program produced distinguishable
negative-value output. See [the bounded VPU observation](docs/vpu-relu-observation.md).
An independent exact-rational oracle additionally checked 43 vectors
of length 2–32 at one MXU0 output cell: 40 seeded vectors across the finite
normal E4M3 domain plus three directed cases to include every normal input
encoding at least once. All 43 selected-source-linked
standalone-core results matched ordered per-product BF16 round-to-nearest-even
under this bound. The oracle also checks the positive tie and negative
half-ULP cases. This is a bounded MXU0 observation, not general numerical
qualification; subnormals, NaNs, overflow, initial accumulators, MXU1, and
the other output cells in that single-tile numerical corpus still need
independent checks.
The separate [two-K-tile MXU0 check](docs/mxu0-k-continuation-observation.md)
compares reset and continuation on a 57-word typed stream. Four
selected-source-linked standalone-core runs checked every output cell in both
BF16 register halves against the independent ordered-BF16 oracle, including
a reset mutation that yields the second tile alone. The tested finite FP8
set, timing and launch scope remain bounded as described in that note.
The separate [MXU1 anchor-tree check](docs/mxu1-anchor-observation.md) uses
a 42-word typed stream and an independent exact-rational round-once reference.
Three MXU1 standalone-core runs checked both BF16 register halves across
1,024 output cells each. A tie case produced `0x3f81` on MXU1 and `0x3f80`
on the separately executed MXU0 path, demonstrating a numerical-policy
distinction on that input. Broader MXU1 ranges and state remain open.
The [two-K-tile MXU1 continuation check](docs/mxu1-continuation-observation.md)
compares a rounded prior tile plus the second tile against reset and a
single 64-product rounding. Six bounded standalone-core runs checked all
1,024 output cells in both halves, input preservation, and a guard. General
anchor alignment and temporal qualification remain open.
The separate [XLU transpose check](docs/xlu-transpose-observation.md) uses a
29-word typed stream and an independent raw-byte index reference. Three
selected-source-linked standalone-core executions checked all 1,024 bytes
for an all-encoding panel, a nonsymmetric finite panel, and a same-register
source/destination variant. Each preserved the DRAM input and guard; the
different-register runs also preserved the source matrix register. These
cases do not establish general XLU scheduling or completion bounds.
The separate [VPU BF16 addition check](docs/vpu-add-observation.md) uses a
49-word typed stream and an exact-rational reference on a restricted finite
domain. Two full 1,024-element BF16 pair panels matched the selected
standalone core. A directed sum distinguishes the RTL's final bit chop from
BF16 round-to-nearest-even. Wider numerical and temporal behavior remains
unqualified.
The [VPU BF16 multiply check](docs/vpu-mul-observation.md) uses a 49-word
typed stream and an exact-rational BF16 nearest-even reference. Two complete
1,024-element panels matched the selected standalone core, including
halfway rounding in both directions and signed products. Exceptional
values and broader temporal behavior remain unqualified.
The [VPU BF16 subtraction check](docs/vpu-sub-observation.md) uses a 49-word
typed stream and an exact-rational reference on an FP32-exact finite-normal
subset. Two complete 1,024-element panels matched the selected standalone
core, including a case that distinguishes the RTL's final BF16 bit chop from
nearest-even rounding. Other arithmetic and temporal behavior remains
unqualified.
The [VPU BF16 pairwise maximum check](docs/vpu-max-observation.md) uses a
49-word typed stream and a separate raw-bit ordering calculation. Two
complete 1,024-element panels matched the selected standalone core, including
signed-zero and NaN encoding cases that a generic floating-point maximum
cannot explain. General timing and integrated execution remain unqualified.
The [VPU BF16 square check](docs/vpu-square-observation.md) uses a 36-word
typed stream and an independent raw-bit reference for exact powers of two and
special encodings. Two complete 1,024-element panels matched the selected
standalone core, including its NaN-to-positive-zero behavior. The selected
funct7 `0x46` differs from the inspected model class's `0x4e`; full-domain
arithmetic and timing remain unqualified.
The [VPU BF16 cube check](docs/vpu-cube-observation.md) uses the same physical
pair shape and a separate exact-power raw-bit oracle. Two complete
1,024-element panels matched the selected standalone core, including signed
NaN-to-signed-zero behavior. The selected funct7 `0x47` differs from the
inspected model class's `0x4f`; normal non-power fractions and general timing
remain unqualified.
The [VPU BF16 pairwise-minimum check](docs/vpu-min-observation.md) compares
all 1,024 raw BF16 words in each of two panels against an independent ordering
oracle. Signed zeros, NaN encodings, and infinities distinguish the selected
RTL's bit ordering from conventional numerical minimum. Both input pairs and
guard memory were preserved on the standalone core; in-place use and general
timing remain unqualified.
The separate [LLVM boot-entry check](docs/llvm-boot-entry-observation.md)
loaded all 37 words of one LLVM-produced `atlas_program` function into the
selected standalone core and executed it to ECALL halt. It checked the ELF
symbol, executable section, and absence of text relocations; the complete
output matched the independent VSQUARE reference. This is a fixed-address
diagnostic entry, not a callable Atlas ABI or an ELF runtime loader.
The [DMA scalar-pointer capture check](docs/dma-pointer-capture-observation.md)
uses a 16-word typed stream and a before/after-launch mutation. On the
selected standalone core, changing the address register immediately after
the load launch preserved the original DRAM source; changing it before launch
selected the other source. This bounds one scalar-lifetime observation, not
general DMA timing or overlap.
The [VLI.ALL raw-bit check](docs/vli-all-observation.md) uses a 36-word typed
stream. Its selected assembler, OOT emitter, and LLVM-object words agreed.
Four selected standalone-core runs filled both BF16 register halves with the
raw immediate, including negative zero and a NaN encoding; all 1,024 cells
per run matched an independent bit-repetition reference. Unrelated loaded
input memory and a guard were preserved. General availability bounds remain
unqualified.
The [VLI.ROW/COL/ONE raw-bit check](docs/vli-selective-observation.md) uses a
second 36-word typed stream. Six selected-core runs at two raw immediates
matched all 2,048 output bytes per run, including ROW in both pair halves and
preservation of the unselected half for COL/ONE. All six selected-assembler,
OOT emitter, and LLVM-object streams agreed; full-domain and temporal behavior
remain unqualified.
The separate [VPU BF16 row-sum check](docs/vpu-row-sum-observation.md) uses a
36-word typed stream and an exact-rational FP32 tree reference. Two full
32-row panels matched the selected standalone core, including a row that
distinguishes adjacent-tree reduction from serial addition and rows that
check final BF16 rounding. Other reductions and exceptional values remain
unqualified.
The [VPU BF16 row-minimum check](docs/vpu-row-min-observation.md) uses a
36-word typed stream and an independent exact finite-normal reference.
Two complete 32-row panels matched both broadcast result halves on the
selected standalone core, including rows whose minimum resides in the
second source register. Input and guard memory were preserved; general
temporal qualification remains pending.
The [VPU BF16 row-maximum check](docs/vpu-row-max-observation.md) uses a
36-word typed stream and an independent raw-bit ordering reference. Two
complete 32-row panels matched both broadcast result halves on the selected
standalone core. Directed signed-zero, NaN, infinity, and second-half winners
distinguish the selected RTL behavior; general temporal qualification remains
pending.
The [VPU BF16 column-minimum check](docs/vpu-col-min-observation.md) found a
layout disagreement: the selected standalone core reduces 64 physical rows
of 16 lanes across the BF16 register pair and broadcasts those 16 minima to
both halves, while the architectural/model view expects separate minima for
32 logical columns. Two panels made the high and low halves win in turn and
matched the selected RTL's 64-by-16 result. The intended 32-by-32 behavior
remains blocked pending a reviewed compatibility decision.

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
| Exact numerical semantics | Bounded MXU0 ordered-BF16 and MXU1 anchor-tree cases executed separately on selected-source-linked standalone core, including a tie that distinguishes the units; full MXU/VPU and scale semantics remain unqualified |
| Temporal validity and DMA completion | Not qualified; the token conservatively orders issue only |
| Branch/control behavior | Typed branch program and changed-target mutation ran on the rebuilt selected-source-linked standalone `AtlasCore` ARC model; integrated SoC behavior remains unqualified |
| LLVM dialect/object lowering | Registered pass and ELF32 RISC-V smoke test for statically bounded flat streams; JALR rejected |
| LLVM-produced function boot entry | One full 37-word ELF `.text` function executed on selected standalone core with fixed DRAM preload and ECALL halt; PC trace excluded LLVM RET. No general call ABI or ELF loader |
| DMA movement | Typed loopback ran on the rebuilt selected-source-linked standalone core model with 32/32 output words and input/guard preservation; general DMA timing and integrated behavior remain unqualified |
| DMA scalar pointer capture | Typed 16-word stream matched selected assembler and LLVM object bytes; selected standalone core chose A when the address register changed to B after issue and B when changed before issue, with complete 128-byte output/input/guard checks. General queueing, timing and integrated behavior remain open |
| MXU0 arithmetic | Hand-authored typed 35-word program lowered through LLVM to object bytes matching the selected assembler; four sparse/control cases and 43 finite-normal vectors executed with first-cell checks. A 42-word typed variant checked the second BF16 register half. A 57-word two-K-tile stream checked reset versus continuation across all 1,024 output cells on dense and mixed inputs. General arithmetic and scheduling remain open |
| MXU1 arithmetic | Hand-authored typed 42-word reset and 57-word two-K-tile continuation programs matched selected assembler and LLVM object bytes; nine bounded MXU1 runs checked all 1,024 output cells per run, with three continuation panels also reset-mutated. A paired MXU0 tie comparison distinguished the units. General anchor precision, seeded accumulators, and scheduling remain open |
| VPU BF16 ReLU | Hand-authored typed 36-word program matched selected assembler and LLVM object bytes; two dense 32-by-32 finite panels executed on the selected-source-linked standalone core with both register halves checked. Other VPU modes and exceptional values remain open |
| VPU BF16 addition | Hand-authored typed 49-word program matched selected assembler and LLVM object bytes; two bounded 1,024-element panels executed with both output halves, all inputs, and guard checked. FP32-inexact sums and exceptional values remain open |
| VPU BF16 multiply | Hand-authored typed 49-word program matched selected assembler and LLVM object bytes; two finite-normal 1,024-element panels checked BF16 nearest-even ties, signs, both output halves, inputs and guard on selected standalone core. Exceptional values and general timing remain open |
| VPU BF16 subtraction | Hand-authored typed 49-word program matched selected assembler and LLVM object bytes; two FP32-exact finite-normal 1,024-element panels checked subtraction order, signs, final BF16 bit chop, both output halves, inputs and guard on selected standalone core. FP32-inexact differences, exceptional values, and general timing remain open |
| VPU BF16 pairwise maximum | Hand-authored typed 49-word program matched selected assembler and LLVM object bytes; two sampled raw-encoding 1,024-element panels checked signed zeros, NaN bit ordering, both output halves, inputs and guard on selected standalone core. General timing and in-place use remain open |
| VPU BF16 pairwise minimum | Hand-authored typed 49-word program matched selected assembler and LLVM object bytes; two sampled raw-encoding 1,024-element panels checked signed zeros, NaN bit ordering, both output halves, inputs and guard on selected standalone core. General timing and in-place use remain open |
| VPU BF16 square | Hand-authored typed 36-word program matched selected assembler and LLVM object bytes; two 1,024-element panels checked exact powers, signed NaNs/zeros, subnormals, infinities, underflow and overflow across both register halves on selected standalone core. Nonzero normal fractions and general timing remain open |
| VPU BF16 cube | Hand-authored typed 36-word program matched selected assembler and LLVM object bytes; two 1,024-element panels checked signed exact powers, NaNs/zeros, subnormals, infinities, underflow and overflow across both register halves on selected standalone core. Nonzero normal fractions and general timing remain open |
| VPU BF16 row sum | Hand-authored typed 36-word program matched selected assembler and LLVM object bytes; two 32-row panels checked both broadcast result halves, a tree-order witness, final BF16 rounding, input and guard on selected standalone core. Other reduction modes and exceptional values remain open |
| VPU BF16 row minimum | Hand-authored typed 36-word program matched selected assembler and LLVM object bytes; two 32-row finite-normal panels checked both broadcast result halves, input and guard on selected standalone core. Exceptional values and general timing remain open |
| VPU BF16 row maximum | Hand-authored typed 36-word program matched selected assembler and LLVM object bytes; two 32-row raw-bit panels checked both broadcast result halves, signed zeros, NaNs, infinities, input and guard on selected standalone core. Other encodings and general timing remain open |
| VPU BF16 column minimum | Hand-authored typed 36-word program matched selected assembler and LLVM object bytes; two finite-normal panels checked the selected core's 64-by-16 physical-column result and demonstrated disagreement with the 32-by-32 architectural/model result. The layout discrepancy, full-domain semantics, and timing remain unresolved |
| VLI.ALL raw fill | Hand-authored typed 36-word program matched selected assembler and LLVM object bytes; four raw immediates checked all 1,024 BF16 cells across both register halves on selected standalone core, with unrelated input and guard preservation. General timing remains open |
| VLI.ROW/COL/ONE raw fill | Hand-authored typed 36-word program matched selected assembler and LLVM object bytes for six mode/immediate combinations; selected standalone core matched all 2,048 output bytes per run and preserved input/guard. Full-domain and general timing remain open |
| XLU transpose | Hand-authored typed 29-word program matched selected assembler and LLVM object bytes; three 32-by-32 byte panels executed on the selected-source-linked standalone core, including all byte encodings and in-place transpose. General timing and cross-family overlap remain open |
| Program binary, ABI, execution | Extracted object words run through an external diagnostic driver; no Atlas launch ABI, constants package, or qualified hardware execution claim |
| Bounded reset-entry capsule | ACT-independent `atlas-boot-pack` binds checked Atlas source words to one complete ELF `.text`, writes `program.bin` and an explicit fixed-DRAM manifest, and executes the packaged bytes on selected standalone AtlasCore; no callable ABI or integrated SoC claim |

No row in the **99 selected RTL BitPat-row inventory** lacks a typed operation
route or source-level emitter: the count of unrepresented BitPat rows is 0/99.
This is not gate D. A frozen executable software configuration has not been
qualified here, and BitPat coverage does not establish every legality,
semantic, physical-effect, timing, or execution variant. In particular,
FP8 scale/pack paths, full-domain VLI qualification, many unary/reduction and
scalar/control variants, and cross-family temporal effects still lack the
independent execution and reference evidence required to close D.

This package should become a golden *comparison reference* only after independent
semantic and hardware tests pass. Merlin's generated dialect and this hand
implementation should consume the same frozen selected target contract; this
repository must not become a hidden second authority for numerical semantics.

No `.merlin` certification or release manifest is included because the
required backend and execution evidence do not yet exist. No remote push has
been made.
