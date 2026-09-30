# Atlas machine dialect reference

This hand-authored OOT dialect targets selected RTL
`0079c0541111197741a231c002e3843fa6f545b2`. Its source of truth is
[`AtlasOps.td`](../include/Atlas/AtlasOps.td), with verifiers/effects in
[`AtlasOps.cpp`](../lib/AtlasOps.cpp) and word encoding/stream checks in
[`AtlasEncoding.cpp`](../lib/AtlasEncoding.cpp). The
[99-mode census](selected-variant-census.md) binds parameterized operation
modes to decoder rows. Encoding coverage does not establish full arithmetic,
timing, or integrated SoC qualification.

## Common contract

`!atlas.state` is an ordering token, not a tensor. `atlas.start` produces the
initial state and emits no word. Each other operation consumes one state and
returns the next. The encoder requires one flat linear chain, at least one
machine operation, and one nonredirecting instruction after each branch or
jump for the selected RTL's delay slot. Machine operations currently declare
conservative physical-state read/write effects. Physical registers and slots
are attributes, so MLIR SSA does not yet express their dataflow.

The machine register file has 64 registers of 1,024 bytes. BF16 tiles occupy
even-based pairs in `0..62`; FP8 tiles use one register in `0..63`. Weight and
accumulator slots `0..1` belong to a specific MXU `0` or `1`. Scalar registers
are `0..31`. The parser/printer uses MLIR generic operation syntax. This is
machine IR; semantic tensor operations, dynamic shapes, host calls, and a
callable ABI are absent.

## Every current operation

All machine rows take and return `!atlas.state`. The fields shown are typed
attributes. Their verifier checks the listed forms, with mode-specific
bounded execution evidence in the census and source-discrepancy notes.

| Operation | Attributes and verified forms | Physical meaning and limit |
| --- | --- | --- |
| `atlas.start` | none | Begin stream; no encoded word. |
| `atlas.vload` | `dst, base, offset, format="raw"`; tensor destination, scalar base, signed 12-bit offset in 32-byte units | Load one 1,024-byte VMEM tile into a tensor register. |
| `atlas.vstore` | `src, base, offset, format="raw"`; same register and offset bounds | Store one 1,024-byte tensor register to VMEM. |
| `atlas.dma` | `direction=load/store, channel=0..7, reg, dram, size`; latter three are scalar register numbers | Launch asynchronous VMEM/DRAM transfer; completion is separate. |
| `atlas.dma_wait` | `channel=0..7` | Wait for that DMA channel. |
| `atlas.dma_config` | `channel=0..7, base_reg=0..31` | Set channel base from scalar register; encoding follows selected RTL. |
| `atlas.mxu_push` | `kind=weight_fp8/acc_fp8/acc_bf16, unit=0/1, src, slot=0/1`; BF16 source even pair | Push tile to one unit's local weight or accumulator slot. |
| `atlas.mxu_matmul` | `unit=0/1, src, weight_slot=0/1, acc_slot=0/1, accumulate=bool` | Launch reset or continuing contraction. MXU0/MXU1 have distinct arithmetic; issue is not completion. |
| `atlas.mxu_pop` | `format=fp8/bf16, unit=0/1, dst, slot=0/1, scale_reg=0..31`; BF16 destination even pair and scale register zero | Read local accumulator to tensor registers. General FP8 scale policy is open. |
| `atlas.vpu_binary` | `kind=add/sub/mul/min/max, dst, lhs, rhs`; all even BF16 pair bases | BF16 pair operation. Add/sub and min/max have selected-RTL rounding/order quirks. |
| `atlas.vpu_unary` | `kind=mov/recip/exp/exp2/square/cube/relu/sin/cos/tanh/log2/sqrt, dst, src`; even BF16 pairs | Two-register unary mode; several mode semantics remain unqualified. |
| `atlas.vpu_pack` | `direction=bf16_to_fp8/fp8_to_bf16, dst, src, scale_reg=0..31`; BF16 side even pair | E8M0 conversion. PACK joins consecutive physical 16-lane BF16 rows into 32-byte FP8 rows, not logical row-major output. |
| `atlas.vpu_reduce` | `kind=col_sum/col_min/col_max/row_sum/row_min/row_max, dst, src`; even BF16 pairs | Column modes scan 64 physical rows × 16 lanes and broadcast 16 results on selected RTL; row modes span two 16-lane halves. |
| `atlas.vli` | `mode=all/row/col/one, dst, immediate=0..65535`; all/row use even pair | Write raw 16-bit immediate under physical broadcast mask, not numeric BF16 conversion. |
| `atlas.xlu_transpose` | `dst, src` single tensor registers | Transpose a 32×32 FP8 byte tile. |
| `atlas.alu_reg` | `kind=add/sub/sll/slt/sltu/xor/srl/sra/or/and, dst, lhs, rhs`; scalar registers | Selected RV32 register ALU modes. |
| `atlas.alu_imm` | `kind=addi/slti/sltiu/xori/ori/andi/slli/srli/srai, dst, src, immediate`; signed 12-bit nonshift or shift `0..31` | Selected RV32 immediate ALU modes. |
| `atlas.branch` | `kind=beq/bne/blt/bge/bltu/bgeu, lhs, rhs, offset_bytes`; even signed displacement `[-4096,4094]` | Conditional redirect; RTL applies `offset_bytes/2` to its instruction-word PC and executes one delay slot. |
| `atlas.jump` | `kind=jal/jalr, dst, base, offset`; JAL requires base zero and even byte offset `[-1048576,1048574]`; JALR has signed word offset `[-2048,2047]` | Link and redirect after one delay slot. JALR uses scalar word-index target; LLVM lowering needs a proven in-block constant. |
| `atlas.delay` | `cycles=0..4095` | Frontend counter stall. Count alone does not prove MXU/VPU/DMA availability. |
| `atlas.upper` | `kind=lui/auipc, dst, immediate=0..1048575` | Upper immediate or selected instruction-index PC addition. |
| `atlas.csr` | `kind=rrw/rrs/rrc/rrwi/rrsi/rrci, dst, source, address`; only `0xC00..0xC03`, `0xC10..0xC11`; read-only status uses zero-source read form | Selected internal CSR access; E8M0 scale uses SELI/SELD instead. |
| `atlas.trap` | `kind=ecall/ebreak` | Termination/trap word; general drain protocol remains open. |
| `atlas.fence` | none | Canonical FENCE word. Selected core shows it does not wait for pending scalar LW. |
| `atlas.scalar_load` | `kind=lb/lh/lw/lbu/lhu/seld/seli, dst, base, offset`; signed byte offset except SELI, which requires base zero and E8M0 code `0..255` | Scalar VMEM read or scale-register load. Scalar writeback may be asynchronous. |
| `atlas.scalar_store` | `kind=sb/sh/sw, src, base, offset`; signed 12-bit byte offset | Scalar VMEM store with selected byte/halfword/word mask. |

The current C++ verifier checks local field legality. It does **not** check
all physical lifetime/alias constraints, numerical-policy compatibility,
availability, or full instruction-stream execution semantics. The
[source discrepancy ledger](source-discrepancies.md) documents differences
between selected RTL, architecture text, and the inspected model.

## Pass and tool inventory

| Name | Kind | Implemented behavior |
| --- | --- | --- |
| `--verify-atlas-machine-stream` | `atlas-opt` module pass | Check local verifiers, flat state chain, selected word encoding, delay-slot adjacency, and in-block target confinement. Leave Atlas MLIR unchanged. It reuses encoder checks; it is not an independent hardware proof. |
| `--convert-atlas-to-llvm-calls` | `atlas-opt` module pass | Preserve each checked machine instruction as a separate `llvm.call @atlas_emit_*` with physical fields, encoded word, word index, conservative effects, and unknown availability. This intermediate requires finalization before LLVM IR translation; its calls are markers, not runtime functions. |
| `--finalize-atlas-llvm-calls` | `atlas-opt` module pass | Reconstruct and verify the typed Atlas stream from the LLVM calls, check every encoded word and control target, then emit one ordered LLVM inline-assembly block. Reject inconsistent fields, words, indexes, and malformed streams. |
| `--convert-atlas-to-llvm` | `atlas-opt` module pass | Replace checked stream with `llvm.func @atlas_program()` containing one side-effecting, ordered `llvm.inline_asm` word block and `llvm.return`. This is a reset-entry body, not a C-callable ABI. |
| `atlas-emit` | tool, not pass | Emit words or `--map-json` sidecar with source operation, attributes, word index, branch/delay-slot mapping, conservative effects, and `availability: unknown`. |
| `atlas-boot-pack` | tool, not pass | Package one checked reset-entry ELF `.text` section with a narrow authored memory layout and manifest. |
| `export_llvm_handoff.py` | exporter, not pass | Produce numbered Atlas/LLVM MLIR, LLVM IR, RV32 assembly, relocatable object, linked ELF, disassembly, word map, and hashes for the bounded MLP/attention fixtures. |

No timing annotation pass, delay scheduler, instruction selector, allocator,
Linalg-to-Atlas pass, or general callable ABI is implemented in this hand OOT
repository. Delays in the example programs are authored diagnostic spacing.

## Why the LLVM handoff uses inline assembly

The MLIR LLVM dialect can retain an inspectable machine stream without an LLVM
source change. `--convert-atlas-to-llvm-calls` now uses one `llvm.call` per
Atlas instruction, keeping the operation family, physical fields, checked word,
source location, and order. These calls are compiler markers. The finalizer
reconstructs the typed Atlas stream and checks the encodings again; translating
the marker calls directly to LLVM IR would leave unresolved external functions.

The selected unmodified LLVM RISC-V code generator does not define Atlas's
custom tensor instructions or its instruction-index PC behavior.
`--finalize-atlas-llvm-calls` therefore freezes the verified stream as one
side-effecting inline-assembly block of encoded words. Keeping it in one block
prevents LLVM from placing instructions between a branch and its selected delay slot or
changing the offsets of internal targets. The numbered handoff examples prove
that LLVM emitted the checked bytes; they do not give LLVM knowledge of the
Atlas operations or certify a callable function.

For a standalone Atlas program, a future object writer could put the checked
encoder's words directly into an ELF section and link them with `ld.lld`. That
would remove the inline-assembly bridge, but LLVM would no longer generate
those instruction bytes. Encoding the words as an LLVM global in an executable
section would also hide their meaning as data. An instruction-aware LLVM route
requires adding Atlas instruction definitions and lowering to an LLVM target;
that is a separate toolchain project, not a capability of this unmodified
LLVM build. Splitting the current block into one inline-assembly call per
Atlas operation would expose more boundaries in LLVM IR but could insert code
between selected instructions and invalidate branch and delay-slot placement.

The typed Atlas stream is the primary place for timing annotations and
scheduling. The structured LLVM-call stage is available for LLVM-dialect
analysis and metadata passes; any pass that changes the instruction stream
must update the checked fields, word, index, and source map consistently or
the finalizer rejects it. Keep the `atlas-emit --map-json` sidecar with the
resulting ELF so binary offsets remain traceable afterward.

Nicolas and Jeremy can add resource/availability annotation passes on the
structured LLVM-call stage, or on typed Atlas machine IR. The call attributes
expose each operation and its physical fields without modifying LLVM. A pass
that inserts or reorders instructions must also regenerate checked words,
indexes, and the operation-to-word map. That support is not implemented yet;
for scheduling work today, the typed Atlas stream already has the encoder and
map exporter. The first annotation pass should bind facts to a selected-RTL
source identity and leave unknowns explicit. A scheduler must preserve the
one-slot control rule and asynchronous scalar/transfer lifetimes. Its result
needs stream verification, fresh mapping, and selected-core checks. The final
inline-assembly block no longer exposes individual operations.

To add a pass, place its declaration under `include/Atlas/`, its implementation
under `lib/`, list the new source in `lib/CMakeLists.txt`, and register it in
`tools/atlas-opt.cpp`. `AtlasStreamVerification.cpp` is a minimal example of
that wiring. Operation definitions and local legality belong in
`include/Atlas/AtlasOps.td` and `lib/AtlasOps.cpp` if a new attribute or mode
is actually required. Preserve source-bound timing facts as attributes on
Atlas operations or structured LLVM calls. The word-map exporter carries
Atlas operation attributes to binary offsets; an LLVM-call transformation
that changes the stream also needs to update that map. The final block does
not retain individual operation attributes.

The current IR is a **flat word stream** with branch offsets, not a control-flow
graph. A scheduler must account for backward branches and multiple executions
of one static operation; the linear state token alone does not prove a
cross-iteration dependency safe. The verifier checks encoding and in-block
targets, but it does not validate an annotation's timing claim. Their pass
tests should include loops, asynchronous reads, changed delay slots, and
post-schedule selected-core runs before replacing `availability: unknown`.

```sh
build/bin/atlas-opt --verify-atlas-machine-stream \
  --convert-atlas-to-llvm-calls test/examples/handoff_mlp_tile.mlir \
  > out/mlp.structured-llvm.mlir
build/bin/atlas-opt --finalize-atlas-llvm-calls \
  out/mlp.structured-llvm.mlir > out/mlp.encoded-llvm.mlir
```
