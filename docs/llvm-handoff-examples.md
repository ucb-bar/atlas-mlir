# Atlas MLIR to LLVM MLIR handoff examples

These two **hand-authored machine-stage diagnostics** show the last MLIR stage
produced by this out-of-tree package. They are not generated from PyTorch,
Linalg, or Merlin's native selector. They do not qualify whole-model MLP or
attention compilation.

| Example | Intended instruction chain | Atlas words | Numbered stage bundle |
| --- | --- | ---: | --- |
| [MLP tile](../test/examples/handoff_mlp_tile.mlir) | MXU0, BF16 ReLU, E8M0 pack, VMEM relayout, MXU1 | 102 | [Atlas → LLVM → assembly → ELF](../examples/handoff/mlp_tile/01-atlas-machine.mlir) |
| [Attention tile](../test/examples/handoff_attention_tile.mlir) | QKᵀ, row normalization, E8M0 pack, VMEM relayout, PV | 116 | [Atlas → LLVM → assembly → ELF](../examples/handoff/attention_tile/01-atlas-machine.mlir) |

The [examples index](../examples/handoff/README.md) links every numbered
stage, both linked ELF files, disassemblies, and operation-to-word maps.

Both inputs use fixed 32×32 FP8 tiles and already oriented weights. The MLP
has no bias; the attention sequence has no mask or causal state. `SELI 127`
selects unit E8M0 scale. The BF16 scale constant is raw `0x3e83`. MXU0 and
MXU1 have different arithmetic. The handwritten delays are diagnostic spacing
based on public instruction examples, **not** general hardware availability
bounds. The selected RTL revision is
`0079c0541111197741a231c002e3843fa6f545b2`.
After PACK, a 32-iteration scalar VMEM loop copies each 16-lane row half into
the corresponding logical FP8 row. It uses scalar `lw`/`sw`, a backward `blt`,
and an explicit nonredirecting branch delay-slot instruction. This is a
fixed-shape implementation of the selected physical layout, not a general
layout conversion pass.

Representative Atlas machine MLIR from the MLP fixture:

```mlir
%s27 = "atlas.mxu_matmul"(%s26) {unit = 0 : i32, src = 0 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false} : (!atlas.state) -> !atlas.state
%s28 = "atlas.delay"(%s27) {cycles = 95 : i32} : (!atlas.state) -> !atlas.state
%s29 = "atlas.mxu_pop"(%s28) {format = "bf16", unit = 0 : i32, dst = 8 : i32, slot = 0 : i32, scale_reg = 0 : i32} : (!atlas.state) -> !atlas.state
```

The full file shows the ReLU, pack, scalar relayout loop, second MXU, DMA
stores, and termination. The state token fixes instruction order, including
the loop's branch and its one executed delay slot.

## Structured LLVM dialect and final encoding

`--convert-atlas-to-llvm-calls` produces one LLVM-dialect call per checked
Atlas instruction. For example:

```mlir
llvm.call @atlas_emit_mxu_matmul() {atlas.fields = {acc_slot = 0 : i32, accumulate = false, src = 0 : i32, unit = 0 : i32, weight_slot = 0 : i32}, atlas.source_op = "atlas.mxu_matmul", atlas.word = 335544439 : i32, atlas.word_index = 26 : i32} : () -> ()
```

The checked-in stage 02 files show the exact calls and values. These calls
are compiler markers, not runtime functions. The LLVM dialect can carry
per-instruction fields and annotations without changing LLVM. Before LLVM IR
translation, `--finalize-atlas-llvm-calls` reconstructs the Atlas stream,
rechecks each field/word pairing and control target, and replaces the calls
with the final word block. An inconsistent field, word, or index is rejected.

```sh
build/bin/atlas-opt --convert-atlas-to-llvm-calls \
  test/examples/handoff_mlp_tile.mlir > out/mlp.structured-llvm.mlir
build/bin/atlas-opt --finalize-atlas-llvm-calls \
  out/mlp.structured-llvm.mlir > out/mlp.encoded-llvm.mlir
mlir-translate --mlir-to-llvmir out/mlp.encoded-llvm.mlir > out/mlp.ll
```

The final stage has one function and one ordered, side-effecting inline assembly
operation. The MLP snapshot starts as follows; the full checked-in files retain
all 102 or 116 words:

```mlir
module {
  llvm.func @atlas_program() {
    llvm.inline_asm has_side_effects ".word 0x00100e13\0A.word 0x00000293\0A...", "~{memory}" : () -> ()
    llvm.return
  }
}
```

The real LLVM MLIR uses the precise word string shown in each snapshot. The
pass emits one block so LLVM keeps Atlas branch targets and delay-slot words
adjacent. It is a void reset-entry program body, not a C-callable Atlas ABI.
The `.word` stream is not executable on a generic host RISC-V CPU.

To reproduce a complete bundle with the Atlas source, LLVM MLIR, LLVM IR,
RISC-V `.word` assembly, relocatable object, linked ELF32 executable,
disassembly, extracted `.text`, operation map, and hashes:

```sh
python tools/export_llvm_handoff.py \
  --atlas-bin-dir build/bin \
  --llvm-bin-dir /path/to/matching/llvm-install/bin \
  --linker /path/to/ld.lld \
  --output-dir out/llvm-handoff-fresh
```

The output directory must be empty. The exporter checks that each structured
call corresponds to one emitted word, then checks that exactly one final
side-effecting inline-assembly operation contains all source words, compares
the leading object words against `atlas-emit`, and checks that linking did not
change `.text`. The linked ELF has entry `atlas_program` at zero. It records
source and tool identity in `manifest.json`. `test/test_handoff_examples.py`
compares every checked-in stage with fresh output from the pinned toolchain.
Rebuild `atlas-opt` from this repo before reproducing the bundle. The snapshot
was generated with unmodified LLVM/MLIR 23.0.0git; the available `ld.lld`
18.1.3 only linked the LLVM-produced object. Linking does not qualify an
Atlas launch ABI, and generic RISC-V disassembly is not a semantic decode of
Atlas custom operations.

## What the selected-core smoke runs establish

On the source-linked standalone `AtlasCore`, three uniform FP8-one MLP input
tiles produced 1,024 BF16 outputs of `0x4480` (1024). Zero Q/K and FP8-one V
produced 1,024 BF16 attention outputs of `0x3f80` (1). Both streams halted,
preserved their three inputs and a DRAM guard, and performed 96 DMA reads and
64 writes. The MLP run took 2,966 selected-core cycles; attention took 3,428.
These are two directed smoke inputs, not a timing qualification or general
model result. `test/test_handoff_examples.py` reproduces them when the selected
standalone-core environment is supplied. It additionally extracts `.text`
from each checked-in linked ELF and reruns one directed panel using those
exact LLVM-produced words; both halt and produce the expected 1,024 BF16
cells.

The fixed-shape loop was added after an earlier version's sparse tests exposed
the selected PACK layout: a logical value at row 0, column 20 appeared at row
16, column 4 before relayout. With the loop, an MLP input containing just that
value and two identity weights returned exactly one BF16 one at row 0, column
20. A second directed MLP panel used two identity weights and four positive
FP8 codes to check all 1,024 output coordinates against their expected BF16
values. An attention input with elevated logits at (row 0, key 20) and
(row 17, key 3), plus identity V, returned `0x3d20` at those two coordinates;
the other 1,022 codes were `0x3d00`. The input tensors and a DRAM guard
remained unchanged in these selected-core runs. These cases check the
composition route and layout position. They do not establish full-domain
arithmetic equivalence to framework MLP or softmax, masks, tails, arbitrary
shapes, a qualified schedule, or an integrated SoC launch.

## Delay-analysis handoff

The structured LLVM snapshot retains operation names, physical fields, and
instruction order. The final encoded LLVM snapshot is useful for object
generation and exact byte comparison but has lost those roles. A delay
scheduler can consume the **Atlas machine MLIR** or the structured LLVM-call
stage with a source-bound timing/effects contract. Changes to an LLVM-call
stream must retain field/word consistency and update word indexes and the
source map; the current finalizer rechecks the former but does not regenerate
the latter. Scheduling and availability qualification are not implemented here.

The exporter emits `<example>.word-map.json` alongside LLVM MLIR. Its
`atlas.machine_word_map.v1` rows bind every zero-based Atlas word index and
byte offset to the parsed machine operation, encoding, and typed attribute
text. A delay row includes its explicit stall count. A branch row includes
its direct word-index target; the following row identifies its delay-slot
owner. `manifest.json` hashes the map, source, LLVM MLIR, and object. The
exporter checks map words against both `atlas-emit` and the LLVM object prefix.
The map is generated from parsed MLIR operations inside `atlas-emit`; the
Python exporter does not recover operation identity from source text.

The current dialect gives every issued operation conservative physical-state
read/write effects. The sidecar reports `availability: "unknown"` for every
row, including an explicit `atlas.delay`: a stall count alone does not prove
an MXU result or DMA transfer has completed. Jeremy's scheduler can use the
index map to attach qualified resource, scalar lifetime, and completion facts
to the right word, then return an edited Atlas MLIR stream for verification
and re-export. Unit-specific effects and timing bounds require source-backed
contracts and tests before they can replace these unknowns.

For a consumer that only accepts LLVM MLIR, stage 02 supplies one structured
call per instruction; the source-to-word-index sidecar links it to the final
binary. Editing individual `.word` instructions inside the final
inline-assembly string would bypass the earlier dialect checks. Schedule edits
should use Atlas MLIR or the structured LLVM-call stage and be revalidated.
The exact adapter to another project remains to be defined against that
project's parser and pass interface.
