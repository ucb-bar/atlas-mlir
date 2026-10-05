# Physical Atlas program boundary for a functional model

`atlas-emit --program-json <atlas-machine.mlir>` exports a checked physical
instruction stream as `atlas.physical_program.v1`. This is the handoff to a
separate instruction-level functional model. It is not an interpreter for
virtual SSA, a complete instruction semantics specification, or a timing
certificate. The model executes the `word_u32` values; the operation names and
typed fields help diagnose the instruction that produced each word.

For example:

```sh
build/bin/atlas-emit --program-json \
  test/examples/branch_delay.mlir > out/branch.physical-program.json
build/bin/atlas-emit test/examples/branch_delay.mlir > out/branch.words.txt
```

The exporter first verifies the flat physical `!atlas.state` chain, schedule,
encoding, delay slots, and in-block direct targets. It refuses virtual Atlas
SSA and produces no JSON after a failed check. `--map-json` remains the
source-oriented per-word sidecar; its v1 schema is unchanged.

## Consumer contract

The root records `selected_rtl_revision`, `entry_word_index = 0`,
`pc_unit = instruction_word`, `word_endianness = little`, and `word_count`.
Each instruction row has a zero-based `word_index`, exact `word_u32` and
eight-digit `word_hex`, physical Atlas operation name, typed `fields`, and
`control`. Integer fields are JSON integers, booleans are booleans, and enum
names are strings. An attribute type the exporter cannot preserve is an
error. A consumer must check the selected revision and compare every
`word_u32` with the actual loaded binary/ELF text before using the metadata.

Control records distinguish:

| `control.kind` | Supplied information |
| --- | --- |
| `sequential` | Encoded instruction, normally followed by the next word. |
| `conditional_direct` | Target and fallthrough word indexes, plus the intervening delay-slot word. |
| `jump_direct` | Direct target and delay-slot word indexes. |
| `jump_register` | Base scalar register, signed **word** offset, and delay-slot word; the runtime target is not guessed by the exporter. |
| `trap` | Selected ECALL/EBREAK encoding; the model decides its architectural halt/trap behavior. |

The instruction after a branch/jump also has
`delay_slot_for_word_index`. A `delay` operation has
`explicit_delay_cycles`. Direct branch/JAL target indexes use the selected
core's instruction-index PC convention: `branch_pc + signed_byte_offset / 2`.
JALR uses a register value plus its signed word offset. These are **Atlas**
conventions; treating the indexes as ordinary RISC-V byte PCs would be wrong.
The focused branch, loop, and JALR selected-core tests in this repository
check their bounded behavior. They do not establish every dynamic JALR target.

The model can use a conventional `fetch(pc) -> decode/execute -> update pc`
loop. A branch records a pending target, executes its delay-slot word, then
commits the redirect or fallthrough. Tensor, scalar, CSR, memory, DMA, and
termination state must follow the selected ISA/RTL contract. The exporter
sets `timing_scope` to explicit delay only: it does not assert availability
latencies, DMA completion, or the numerical meaning of any operation. Unknown
semantics must remain an explicit unsupported result in a qualified model.

The [captured MLP source](../examples/handoff/captured_mlp/00-linalg.mlir)
can generate a physical program and a separate compiler manifest under
`out/captured-mlp/public-compiler/`, using the command in the
[compiler notes](captured-mlp-compiler.md). The manifest records input,
constant, output, and precision-policy bindings. The exported
program has no sample inputs or golden outputs. Its 143 Atlas words match
the leading object and linked ELF text words; LLVM's trailing return is not
an Atlas instruction. A functional-model invocation needs its own runtime
input and memory preload under that ABI.

The currently examined `npu_model` constructs Python instruction objects and
does not directly ingest this word stream. It is useful as a source of
candidate operation semantics, but known MXU arithmetic and control-flow
differences from the selected RTL must be reconciled before using its outputs
as an independent oracle. No `npu_model` execution is claimed by this export.
