# Atlas dialect and pass reference

The Atlas dialect targets the pinned RTL (`ucb-bar/atlas-npu`
`0079c0541111197741a231c002e3843fa6f545b2`). Ops are defined in
[`AtlasOps.td`](../include/Atlas/AtlasOps.td), verifiers and effects are in
[`AtlasOps.cpp`](../lib/AtlasOps.cpp), and word encoding and stream checks are
in [`AtlasEncoding.cpp`](../lib/AtlasEncoding.cpp). How the stages connect is
in the [compiler pipeline](compiler-pipeline.md); per-instruction RTL status
is in [hardware status](hardware-status.md). The parser and printer use MLIR
generic op syntax.

## IR stages

**Virtual IR** uses SSA values for 32×32 tiles (`!atlas.virtual_bf16`,
`!atlas.virtual_fp8`) and `!atlas.virtual_state` to order external effects.
`%t1` names a value, not register 1. Control flow uses `func.func`, `cf.br`,
and `cf.cond_br`; values, including the state token and loop-carried tiles,
merge through block arguments. `--verify-atlas-virtual-stream` checks that
each edge passes the current state, output indices are distinct, and no
machine op is mixed in.

**Machine IR** threads a physical `!atlas.state` token: `atlas.start` makes it
(no word) and every machine op consumes one state and returns the next. The
encoder requires one flat chain with at least one op and a non-redirecting
instruction in the delay slot after every branch or jump. Registers and slots
are checked integer attributes, so the encoder sees the chosen numbers.

The machine has 64 tensor registers of 1,024 bytes. A BF16 tile uses an
even-based pair in `0..62`; an FP8 tile uses one register in `0..63`. Each MXU
(`0`, `1`) has weight and accumulator slots `0..1`. Scalar registers are
`0..31`.

## Virtual operations

| Op (`atlas.virtual_*`) | Operands → results | Meaning |
| --- | --- | --- |
| `start` | → state | Begin the function; no word |
| `input_bf16`, `input_fp8` | state, `index` → state, tile | Read a declared input tile |
| `output_bf16` | state, tile, `index` → state | Write a declared output tile |
| `vpu_unary` | BF16, `kind` → BF16 | Any machine unary kind |
| `vpu_binary` | BF16, BF16, `kind` → BF16 | `add`, `sub`, `mul`, `min`, `max` |
| `vpu_reduce` | BF16, `row_sum/row_min/row_max` → BF16 | Each row's result fills that row; lowers to `atlas.vpu_reduce` |
| `vpu_splat` | `bits` `0..65535` → BF16 | Same raw bits everywhere; lowers to `atlas.vli` mode `all` |
| `xlu_transpose` | FP8 → FP8 | Element `(j, i)` is source `(i, j)` |
| `mxu_matmul` | FP8, FP8, `unit` → BF16 | One reset contraction; the unit affects the result |
| `pack_fp8` | BF16, `scale_code` → FP8 | Row-major FP8 tile; lowering accepts codes `122..128` and adds the relayout |
| `dma_load_fp8/bf16` | state, addr, len → state, tile | Atomic DRAM load |
| `dma_store_fp8/bf16` | state, tile, addr, len → state | Atomic DRAM store |
| `mxu_load_weight` | state, FP8, `unit` → state, weight | Replace the unit's current weight |
| `mxu_load_acc_fp8/bf16` | state, tile, `unit` → state, acc | Seed an idle accumulator (FP8 has no scale) |
| `mxu_reset` | state, FP8, weight → state, acc | Start a contraction; no live accumulator on the unit |
| `mxu_accumulate` | state, FP8, weight, acc → state, acc | Continue; consumes the accumulator version |
| `mxu_readout_bf16` | state, acc → state, BF16 | Read and consume the accumulator |
| `mxu_readout_fp8` | state, acc, scale → state, FP8 | Quantize with a raw scale code and consume |
| `scale_constant` | `code` `0..255` → scale | Pure, immutable E8M0 code |

Scalars come from `arith.constant` (i1/i32), `arith.addi` (wrapping i32;
overflow flags are rejected everywhere), and `arith.cmpi`. DMA rules
(whole tiles, constant addresses, alignment, range) are in the
[compiler pipeline](compiler-pipeline.md#dma-in-virtual-ir).

**Function ABI.** Lowering accepts one function per module. It carries
`atlas.input_dram_base` and `atlas.output_dram_base` and, when it takes
runtime scalar arguments, `atlas.control_dram_base`. Each base is a
1,024-byte-aligned 32-bit address; tile indices select 2,048-byte slots, and
an FP8 tile uses the first 1,024 bytes of its slot. The control mailbox holds
i1/i32 arguments as consecutive little-endian words; lowering records their
registers in `atlas.scalar_arg_regs`. This is a reset-entry program ABI, not
a C-callable one.

**CFG rules.** Block arguments may be i1/i32 controls and virtual BF16 tiles.
There is one `virtual_start`, one return block, and all outputs and stores are
in that block.

**MXU handles.** `!atlas.virtual_mxu_weight<u>` is a resident weight and
`!atlas.virtual_mxu_acc<u>` one accumulator version on unit `u`. Each unit
holds 2 weights and 2 live accumulators per block. The verifier rejects
overflow, stale or forked accumulator versions, accumulators live at block
exit, and handles crossing CFG edges. A weight stays resident until its last
use; readout leaves it in place. Scale constants may be used across blocks,
but there are no scale block arguments or runtime scales. MXU readout scaling
is its own hardware behavior and is not the same as VPU pack. The older
`virtual_mxu_matmul` still works but needs no live MXU handles on its
unit.

Examples in [`test/examples`](../test/examples): `virtual_bf16_ssa.mlir`,
`virtual_bf16_cfg.mlir`, `virtual_mxu_accumulation.mlir`,
`virtual_mxu_seeded_fp8.mlir`, `virtual_fp8_two_layer_mlp.mlir`, and
`virtual_fp8_two_layer_mlp_bias.mlir` (biases pre-broadcast to 32×32 tiles).

### Allocation

[`VirtualAllocationPlan`](../include/Atlas/AtlasVirtualAllocation.h) computes
all placements before any machine op is emitted: CFG interference coloring,
MXU slots, DMA channels and staging windows, and fixed scratch resources.
[`AtlasVirtualToMachine.cpp`](../lib/AtlasVirtualToMachine.cpp) only reads
the plan. The plan refers to source SSA values, so rebuild it after changing
the IR or its order.

| Resource | Policy |
| --- | --- |
| BF16 tiles, BF16 only | 31 even pairs; pair 62 is the copy temporary |
| BF16 and FP8 together | FP8 in registers `0..31`, BF16 pairs `32..60`, pair 62 temporary |
| Scalar controls | `x10..x26`, or `x18..x26` when a pack needs `x10..x17`; `x27` is the copy temporary |
| Pack scratch | VMEM words `32768..33279`; input indices below 64 |
| Boundary tiles | VMEM bank 0 stages inputs, bank 1 outputs |

A register is reused once its value is dead. Results may not overwrite their
operands. Block arguments become edge copies, and copy cycles go through the
temporaries. There is no spilling: excess pressure is an error.

Lowering calls `VirtualAllocationPlan::verify()` first. Independent checkers
recompute ownership from the source instead of trusting the allocator:
[registers](../lib/AtlasRegisterAllocationVerification.cpp) (liveness, ranges,
even pairs, scratch, entry staging, DMA helper setup),
[DMA](../include/Atlas/AtlasDMAAllocationVerification.h) (channel, staging, and
helper geometry), and [MXU](../include/Atlas/AtlasMXUAllocationVerification.h)
(per-unit slot ownership via [`MXUOwnership`](../include/Atlas/AtlasMXUOwnership.h)).

### Generated-artifact checks

Lowering marks its output `atlas.generated_from_virtual = "resource-contract-v5"`
with six contract attributes and `atlas.timing_state`. A module with no marker,
contracts, or tags is hand-written and gets no generated checks; any other mix
is rejected ([`AtlasGeneratedArtifact.h`](../include/Atlas/AtlasGeneratedArtifact.h)).
Contracts come from the source and checked placements, never from the plan.
Generated programs must fit 2,048 words after DMA waits and delays.

| Contract | Checks |
| --- | --- |
| `atlas.virtual_dma_contract` ([DMA](../include/Atlas/AtlasDMAContractVerification.h)) | Launch operands and completion of virtual DMA transfers; DRAM conflicts until the matching wait |
| `atlas.virtual_mxu_contract` + command tags ([MXU](../include/Atlas/AtlasMXUContractVerification.h)) | Every MXU command's operands, formats, slots, version, FP8 scale, and ownership on each path |
| `atlas.virtual_tile_contract` ([tile](../include/Atlas/AtlasTileContractVerification.h)) | Registers, addresses, and required predecessors of every VLOAD/VSTORE, DMA launch/wait, and mailbox LW |
| `atlas.virtual_cfg_contract` ([CFG](../include/Atlas/AtlasCFGContractVerification.h)) | Branch conditions, targets, op visits, and register contents across branches and loops |
| `atlas.virtual_source_memory_contract` ([memory](../include/Atlas/AtlasSourceMemoryEffectContract.h)) | Overlapping source DRAM effects complete before a later write-involving one |
| `atlas.virtual_buffer_contract` ([buffer](../include/Atlas/AtlasBufferContractVerification.h)) | Each VMEM reader sees its writer's words, including the PACK relayout (as the lowering models it, not taken from the RTL); each PACK scale register holds its code |

An explicit transfer is pending from its launch to its wait. In between, only
work that moves no memory a transfer could own may issue, plus whole PACK
relayout loops ([`AtlasPackLoop.h`](../include/Atlas/AtlasPackLoop.h)): one
block closed by its own helper back-edge, holding only that PACK's scalar copy,
entered only by falling into it. Decoding runs each loop from its entry
registers to find the VMEM it touches; the DMA memory and timing checks compare
that with every pending transfer. A transfer may stay pending across a block
boundary only into or out of such a loop, so a PACK does not complete earlier
transfers ([DMA across PACK](pack-dma-overlap.md)).

Scheduling and timing passes take `provider=<id>` (default: the module's
`atlas.timing_provider`, else `npu-model-rtl-match-v1`) from a registry that
holds only the built-in default ([`AtlasTimingProvider.h`](../include/Atlas/AtlasTimingProvider.h)).
The selected provider supplies virtual machine-operation costs, physical
dependence distances, issue gaps, reservations, DMA conflicts and verification.
Unknown ids, missing rules and disagreement with retained provider metadata fail.
Emission and LLVM handoff of generated programs need `"timed"` state and the
provider name; hand-written streams need no timing metadata.

## Machine operations

All take and return `!atlas.state`. Fields are typed attributes.

| Op | Fields and verified forms | Meaning and limits |
| --- | --- | --- |
| `atlas.start` | none | Begin stream; no word |
| `atlas.vload` | `dst, base, offset, format="raw"`; scalar base, signed 12-bit offset in 32-word units | Load one 1,024-byte VMEM tile |
| `atlas.vstore` | `src, base, offset, format="raw"`; same bounds | Store one tensor register to VMEM |
| `atlas.dma` | `direction=load/store, channel=0..7, reg, dram, size` (scalar registers) | Launch a VMEM/DRAM transfer; completion is separate |
| `atlas.dma_wait` | `channel=0..7` | Wait for that channel |
| `atlas.dma_config` | `channel=0..7, base_reg=0..31` | Set a channel base |
| `atlas.mxu_push` | `kind=weight_fp8/acc_fp8/acc_bf16, unit, src, slot`; BF16 source even | Push to a unit's weight or accumulator slot |
| `atlas.mxu_matmul` | `unit, src, weight_slot, acc_slot, accumulate` | Start a reset or continuing contraction; MXU0 and MXU1 differ |
| `atlas.mxu_pop` | `format=fp8/bf16, unit, dst, slot, scale_reg`; BF16 even dst, scale register 0 | Read an accumulator |
| `atlas.vpu_binary` | `kind=add/sub/mul/min/max, dst, lhs, rhs`; even pairs | BF16 binary op |
| `atlas.vpu_unary` | `kind=mov/recip/exp/exp2/square/cube/relu/sin/cos/tanh/log2/sqrt, dst, src`; even pairs | BF16 unary op |
| `atlas.vpu_pack` | `direction=bf16_to_fp8/fp8_to_bf16, dst, src, scale_reg` | E8M0 conversion; joins physical 16-lane BF16 rows into 32-byte FP8 rows, not row-major |
| `atlas.vpu_reduce` | `kind=col_sum/col_min/col_max/row_sum/row_min/row_max, dst, src` | Column modes scan 64 physical rows × 16 lanes; row modes span two 16-lane halves |
| `atlas.vli` | `mode=all/row/col/one, dst, immediate=0..65535`; all/row use even pair | Write raw 16-bit immediate under a broadcast mask |
| `atlas.xlu_transpose` | `dst, src` | Transpose a 32×32 FP8 byte tile |
| `atlas.alu_reg` | `kind=add/sub/sll/slt/sltu/xor/srl/sra/or/and, dst, lhs, rhs` | RV32 register ALU |
| `atlas.alu_imm` | `kind=addi/slti/sltiu/xori/ori/andi/slli/srli/srai, dst, src, immediate` | RV32 immediate ALU; signed 12-bit or shift `0..31` |
| `atlas.branch` | `kind=beq/bne/blt/bge/bltu/bgeu, lhs, rhs, offset_bytes`; even, `[-4096,4094]` | Target is word PC + `offset_bytes/2`; one delay slot |
| `atlas.jump` | `kind=jal/jalr, dst, base, offset`; JAL base 0, even offset `[-1048576,1048574]`; JALR word offset `[-2048,2047]` | Link and redirect after one delay slot; JALR target is a word index |
| `atlas.delay` | `cycles=0..4095` | Frontend stall; does not prove a unit is done |
| `atlas.upper` | `kind=lui/auipc, dst, immediate=0..1048575` | Upper immediate; AUIPC adds to the word PC |
| `atlas.csr` | `kind=rrw/rrs/rrc/rrwi/rrsi/rrci, dst, source, address`; `0xC00..0xC03`, `0xC10..0xC11` | Internal CSR access |
| `atlas.trap` | `kind=ecall/ebreak` | Halt or trap word |
| `atlas.fence` | none | FENCE word; does not wait for a pending scalar LW |
| `atlas.scalar_load` | `kind=lb/lh/lw/lbu/lhu/seld/seli, dst, base, offset`; SELI needs base 0 and code `0..255` | Scalar VMEM read or scale-register load; writeback may be late |
| `atlas.scalar_store` | `kind=sb/sh/sw, src, base, offset`; signed 12-bit | Scalar VMEM store |

The verifier checks each op's fields. It does not check lifetimes, aliasing,
numerical policy, or availability across the stream.

## Pass and tool inventory

| Name | Kind | What it does |
| --- | --- | --- |
| `--verify-atlas-virtual-stream` | pass | Check virtual types, state edges, output indices, and that no machine op is mixed in. |
| `--lower-atlas-virtual-inputs-to-dma` | pass | Convert boundary FP8/BF16 inputs to explicit DMA loads using the input DRAM base and 2-KiB slot stride; run before virtual scheduling to expose prefetching. |
| `--schedule-atlas-virtual` | pass | Reorder each virtual block before allocation by a pressure- and latency-aware list schedule, keeping SSA, handle, and memory order. `random-seed=N` picks a random legal order. `provider=<id>` selects machine-operation timing for costs; lowering recipes and DMA queue estimates remain compiler heuristics. Missing cost coverage fails. |
| `--lower-atlas-virtual-to-machine` | pass | Allocate and emit untimed machine IR with resource contracts: tiles, controls, MXU, pack, block-argument copies, branches, DMA launches with VLOADs moved to first use. Leaves waits to the next pass. `allocation-report=<file>` writes the plan as JSON, even on failure. |
| `--insert-atlas-dma-waits` | pass | Add each DMA transfer's wait right before the first instruction that needs it done (staging access, channel reuse, DRAM conflict, helper DMA, scalar memory or redirect outside a PACK relayout loop, a PACK loop that touches the staging window, CSR, halt, any other branch target). Transfers stay in flight through PACK. |
| `--verify-atlas-generated-schedule` | pass | Run the [generated-artifact checks](#generated-artifact-checks), wait matching, and delay-slot NOPs; on timed streams also recheck issue spacing. |
| `--verify-atlas-machine-stream` | pass | Check field verifiers, the flat state chain, encoding, delay slots, and in-block targets. |
| `--insert-atlas-delays` | pass | Time each block in order and add the minimum delays, each with an `atlas.reason`. Rejects input that already has delays, and rejects AUIPC, JALR, and linking JAL. A hazard only a DMA wait can fix is an error. |
| `--schedule-atlas-stream` | pass | List-schedule each block with the selected provider's dependence graph, issue gaps, reservations and DMA conflicts, then add delays. Ops move only within a block; delay slots are not filled and require one-cycle provider spacing. |
| `--verify-atlas-timing` | pass | Recheck issue cycles, dependencies, DMA waits, halt, and drained block boundaries for `provider=<id>`. |
| `--convert-atlas-to-llvm-calls` | pass | One `llvm.call @atlas_emit_*` marker per instruction with fields, word, and index. Must be finalized before translation. |
| `--finalize-atlas-llvm-calls` | pass | Rebuild and recheck the Atlas stream from the markers, then emit one ordered inline-assembly word block. |
| `--convert-atlas-to-llvm` | pass | Go straight from machine IR to `llvm.func @atlas_program()` with the word block. |
| `atlas-emit` | tool | Print words; `--map-json` word map; `--program-json` [program JSON](compiler-pipeline.md#physical-program-output). Needs timed state for generated programs unless `--allow-untimed`. |
| `atlas-boot-pack` | tool | Package one reset-entry ELF `.text` with a fixed memory layout and manifest. |
| `tools/export_llvm_handoff.py` | script | Build the full Atlas → LLVM → ELF bundle for the handoff examples. |
| `tools/lower_atlas_program.py` | script | Run the virtual-to-words pipeline and keep every stage. |

By default, scheduling and timing passes use npu-model's `rtl-match` rules, ported from
atlas-compiler-experiments `3ae2b5d`, with DMA VMEM addresses counted in
words; they are a model, not RTL timing. The LLVM handoff uses one
inline-assembly block so stock LLVM, which has no Atlas instructions, cannot
put code between a branch and its delay slot. There is no instruction
selector, whole-program allocator, Linalg-to-Atlas pass, or callable ABI here.

## Adding a pass

Declare the pass under `include/Atlas/`, implement it under `lib/`, add the
source to `lib/CMakeLists.txt`, and register it in `tools/atlas-opt.cpp`.
[`AtlasStreamVerification.cpp`](../lib/AtlasStreamVerification.cpp) is a
small example. New op fields or modes go in `AtlasOps.td` and `AtlasOps.cpp`.

Put timing facts as attributes on Atlas ops or marker calls. A pass that
inserts or reorders instructions on the marker stage must also update the
words, indices, and word map, or the finalizer rejects it. Machine IR is a
flat word stream with branch offsets, so a scheduler must handle backward
branches and ops that run many times; the state token alone does not prove a
cross-iteration dependency safe. Test new passes with loops, asynchronous
reads, and changed delay slots. The
[LLVM handoff](compiler-pipeline.md#llvm-handoff) commands show the marker
stage end to end.
