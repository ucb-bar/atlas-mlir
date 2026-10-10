# Atlas dialect and pass reference

This hand-authored OOT dialect targets selected RTL
`0079c0541111197741a231c002e3843fa6f545b2`. Its source of truth is
[`AtlasOps.td`](../include/Atlas/AtlasOps.td), with verifiers/effects in
[`AtlasOps.cpp`](../lib/AtlasOps.cpp) and word encoding/stream checks in
[`AtlasEncoding.cpp`](../lib/AtlasEncoding.cpp). The
[99-mode census](selected-variant-census.md) binds parameterized operation
modes to decoder rows. Encoding coverage does not establish full arithmetic,
timing, or integrated SoC qualification.

## IR stages and verification

The dialect has two checked stages. No check in this reference qualifies selected-core numerics, RTL timing or hardware completion. `!atlas.virtual_bf16` and
`!atlas.virtual_fp8` are unallocated 32×32 tiles. A name such as `%t1` identifies an MLIR SSA *value*;
the spelling and number do not select Atlas register 1. In this narrow
pre-allocation stage, `!atlas.virtual_state` tracks ordered external reads,
writes, and explicit MXU resource transitions, with names such as `%io1`. In the physical examples, `%s1` is an SSA
machine-state token, not scalar register 1. Pure VPU
candidate operations express their tensor dependencies through `%t` operands.
The `--verify-atlas-virtual-stream` pass checks the virtual state chain,
distinct output indexes, and that no physical operation is mixed into it.
For virtual control flow, the tool uses MLIR `func.func`, `cf.br`, and
`cf.cond_br`. SSA values merge through successor block arguments, including
loop-carried tiles and the state token; there is no separate `phi` operation.
The verifier checks each edge passes the current state, MLIR checks ordinary
SSA dominance and block-argument types, and the current CFG slice places all
external outputs in one return block. Its verifier does not choose an issue
order or prove numerical equivalence. See the
[virtual SSA interpreter contract](virtual-ssa-interpreter-contract.md) for
the incoming-edge binding rule, loop visits, state tokens, and interpreter
test cases.

After allocation, `!atlas.state` is the *physical* instruction-order token.
`atlas.start` produces it and emits no word. Each machine operation consumes
one state and returns the next. The encoder requires one flat linear chain,
at least one machine operation, and one nonredirecting instruction after each
branch or jump for the selected RTL's delay slot. Machine operations declare
conservative physical-state read/write effects. Their physical registers and
slots are checked integer attributes. This is intentional: after allocation,
the chosen physical numbers and aliases must be visible to the encoder.

The machine register file has 64 registers of 1,024 bytes. BF16 tiles occupy
even-based pairs in `0..62`; FP8 tiles use one register in `0..63`. Weight and
accumulator slots `0..1` belong to a specific MXU `0` or `1`. Scalar registers
are `0..31`. The parser/printer uses MLIR generic operation syntax. Semantic
tensor operations, dynamic shapes, host calls, and a callable ABI are absent.

## Pre-allocation BF16/FP8 SSA slice

The following operations provide a small, concrete hand-authored example of
virtual tensor-register def/use. They are instruction candidates, not an
automatic semantic selector. Boundary indexes name declared external
input/output tiles. One BF16 tile requires an even-based *pair* of 1,024-byte
physical registers on the selected RTL. The bounded CFG materializer below
chooses the pair and stages these tiles through the selected VMEM layout.

| Operation | Operands and result | Obligation |
| --- | --- | --- |
| `atlas.virtual_start` | `() -> !atlas.virtual_state` | Begin one virtual region; no word. |
| `atlas.virtual_input_bf16` | state, nonnegative input index -> next state, `!atlas.virtual_bf16` | Read a declared external tile; memory-read effect. |
| `atlas.virtual_input_fp8` | state, nonnegative input index -> next state, `!atlas.virtual_fp8` | Read a declared 32×32 FP8 tile; memory-read effect. |
| `atlas.virtual_vpu_unary` | `!atlas.virtual_bf16`, selected unary kind -> `!atlas.virtual_bf16` | Preserve the chosen operation and SSA dependency; no physical register assigned. |
| `atlas.virtual_vpu_binary` | two `!atlas.virtual_bf16` values, selected binary kind -> `!atlas.virtual_bf16` | Both sources remain explicit, including shared uses. |
| `atlas.virtual_mxu_matmul` | two `!atlas.virtual_fp8` values, unit `0` or `1` -> `!atlas.virtual_bf16` | One 32×32 reset contraction; the unit is part of the numerical semantics. |
| `atlas.virtual_pack_fp8` | `!atlas.virtual_bf16`, E8M0 scale code -> `!atlas.virtual_fp8` | Logical row-major FP8 tile; the current physical lowering admits code `127` only and explicitly relayouts the selected pack result. |
| `atlas.virtual_output_bf16` | state, `!atlas.virtual_bf16`, nonnegative output index -> next state | Write a declared output tile; memory-write effect. |

See [`virtual_bf16_ssa.mlir`](../test/examples/virtual_bf16_ssa.mlir): one
input `%t0` feeds both a unary operation and a later binary operation; two
results are retained. That shared use creates a liveness requirement. The
state `%io1` orders external I/O, whereas `%t0` carries tensor dataflow.
[`virtual_bf16_cfg.mlir`](../test/examples/virtual_bf16_cfg.mlir) shows a
diamond merge and a loop. MLIR may rename the textual `%t` and block labels
when printing; SSA identity is the value and CFG edge, not the spelling.
The CFG verifier accepts only i1/i32 controls and virtual BF16 tiles in block
arguments, a single `virtual_start`, one return block, and boundary outputs in
that block. The implemented lowering accepts exactly one such function per
module. The function has explicit DRAM input/output base attributes and, for
runtime scalar arguments, a separate control mailbox base. Each base is a
1,024-byte-aligned 32-bit address; tile indexes identify 2,048-byte BF16
tiles. The control mailbox holds i1/i32 arguments as consecutive little-endian
32-bit words. Lowering records their chosen scalar registers in
`atlas.scalar_arg_regs`. This is a reset-entry program ABI for the selected
standalone core, not a C-callable function ABI.

The implemented BF16-only slice colors CFG-live virtual BF16 tiles onto 31 usable
even-based register pairs and i1/i32 controls onto scalar x10..x26. Pair 62
and x27 are reserved for parallel-copy cycles. It reuses a location when the
source SSA value is dead, checks coloring against its interference graph, and
rejects excess pressure. Block arguments become edge copies; a cycle is
broken through the reserved temporary. VMEM bank 0 stages input tiles and
bank 1 stages outputs, with DMA completion before reads and stores. Lowering
emits an untimed stream; later passes add [timing](#generated-artifact-checks).
General layer tiling remains a separate obligation.
Generated VPU execution currently admits unary `mov`/`relu` and binary `add`;
other virtual modes receive an explicit
qualification error. `add` uses the selected VPU's FP32 sum followed by a
BF16 bit chop; it is not a generic BF16 round-to-nearest-even operation.
With FP8 values present, it reserves tensor registers 0..31 for FP8 and pairs 32..60
for BF16, with pair 62 as a temporary. A pack reserves scalar x10..x17 for
its VMEM relayout and colors runtime controls in x18..x26. The external input
ABI uses 2,048-byte slots; a 32×32 FP8 tile occupies the first 1,024 bytes of
its slot. Pack scratch uses VMEM words 32768..33279 and currently requires
input indexes below 64. These are conservative, explicit physical partitions,
not a general alias-aware allocator. Changing issue order would require
recomputing live ranges and checking the physical assignment again.

[`virtual_fp8_two_layer_mlp.mlir`](../test/examples/virtual_fp8_two_layer_mlp.mlir)
is a fixed 32×32 virtual SSA example: MXU0, BF16 ReLU, unit-scale pack,
MXU1, BF16 output. Its output is checked through selected standalone-core
execution with changed runtime weights. It has no bias, tails, Linalg import,
or captured model precision transformation.
[`virtual_fp8_two_layer_mlp_bias.mlir`](../test/examples/virtual_fp8_two_layer_mlp_bias.mlir)
adds BF16 VPU bias addition after each MXU. Its boundary biases are already
broadcast into physical 32×32 tiles; capturing a vector bias and preparing
that tile are separate frontend/ABI obligations.

### Allocation policy and instruction emission

[`VirtualAllocationPlan`](../include/Atlas/AtlasVirtualAllocation.h) computes placements before machine instruction emission. Its [implementation](../lib/AtlasVirtualAllocation.cpp) owns the existing CFG interference coloring, MXU handle slots, explicit DMA transfer placements and IDs, and fixed scratch registers, channels, and VMEM windows. [`AtlasVirtualToMachine.cpp`](../lib/AtlasVirtualToMachine.cpp) reads these placements to materialize values and emit instructions; it no longer colors registers or assigns DMA resources while emitting them.

The plan preserves the existing bounded placement policy, instruction order, and delays. It is an internal C++ allocation plan, not another dialect stage or standalone pass. It refers to verified source SSA values and must be rebuilt after changing that IR or its order.

Before emitting instructions, lowering calls `VirtualAllocationPlan::verify()`. The independent [register-assignment checker](../lib/AtlasRegisterAllocationVerification.cpp) recomputes CFG liveness from the source operations and edges, without using the allocator's interference graph. It checks complete, unique assignments, physical register ranges, even BF16 pairs, shared BF16/FP8 storage, reserved scratch registers, and entry-argument staging. Conflicts report the physical register span and the two source values. Scalar and tensor registers occupy separate banks; scale-register numbers do not reserve same-numbered scalar registers.

The checker rejects operand/result aliasing under the current no-in-place policy and checks writes from dead results and unused block arguments because lowering still emits them. Successor arguments must remain distinct from values live through the edge, while parallel-copy cycles may reuse incoming registers across the edge. Register reuse after a value dies remains valid. This checkpoint validates scalar/tensor assignments at SSA operation boundaries; MXU-slot ownership, DMA-window lifetimes, general allocation, spilling, scheduling, and hardware completion timing are outside this checker. Existing virtual-handle and generated-schedule checks remain in force.

The checker also rejects DMA helper setup copies that clobber live scalars or the uncaptured size operand and, for explicit-DMA functions, recomputes across CFG joins and loops the staging base retained until each await and the persistent 1,024-byte helper at implicit DMA reads; other fixed helpers are not covered.

Independent placement checkers recompute ownership from the source: the [DMA checker](../include/Atlas/AtlasDMAAllocationVerification.h) ([issue #10](https://github.com/ucb-bar/atlas-mlir/issues/10)) checks complete assignments, unique bounded IDs and channels, staging and helper-register geometry and block-local staging overlaps; the [MXU checker](../include/Atlas/AtlasMXUAllocationVerification.h) checks complete assignments, per-unit banks and slot ownership through the shared [`MXUOwnership`](../include/Atlas/AtlasMXUOwnership.h) transitions.

### Channel-free virtual DMA and scalar SSA

The DMA slice represents asynchronous whole-tile I/O without choosing a channel, VMEM address, or scalar register. `arith.constant` supplies i1/i32 values, `arith.addi` supplies wrapping i32 arithmetic, and `arith.cmpi` compares i32 operands to produce an i1 control value. These operations are admitted in both module streams and CFG functions. DMA addresses and lengths use those ordinary i32 SSA values; a separate scalar-register type is unnecessary.

Overflow flags on `arith.addi` are rejected throughout virtual streams and CFG functions, including arithmetic unrelated to DMA, tightening the prior CFG scalar admission.

| Operation | Operands after current virtual state | Results after next virtual state |
| --- | --- | --- |
| `atlas.virtual_dma_load_fp8` / `virtual_dma_load_bf16` | DRAM byte address, byte length | `!atlas.virtual_dma_load_fp8` / `!atlas.virtual_dma_load_bf16` pending handle |
| `atlas.virtual_dma_await_fp8` / `virtual_dma_await_bf16` | Matching pending-load handle | Ready `!atlas.virtual_fp8` / `!atlas.virtual_bf16` tile |
| `atlas.virtual_dma_store_fp8` / `virtual_dma_store_bf16` | Source tile, DRAM byte address, byte length | `!atlas.virtual_dma_store` pending handle |
| `atlas.virtual_dma_wait` | Pending-store handle | No additional result; the store is complete |

All seven operations advance the virtual state chain and have conservative read/write effects. A pending handle owns a private logical staging buffer until completion. A load exposes no usable tensor until its await. A store captures its source tile into transfer-owned staging; that staging must survive until the wait. Physical lowering implements this ownership; the virtual IR does not assign that buffer to a physical bank or window.

The current admission requires FP8 transfers of exactly 1,024 bytes or BF16 transfers of exactly 2,048 bytes. Addresses and lengths must be proven from i32 constants and unflagged constant-addition expressions. Address bits are interpreted unsigned, additions wrap modulo 2^32, and additions with overflow flags are rejected rather than evaluated as wrapped constants. DRAM addresses must be at least `0x80000000`, aligned to 32 bytes, and have a widened exclusive transfer end at most 2^32. The contract uses an upper DRAM address half of zero. These checks establish the bounded address form; external memory accessibility and initialization remain invocation requirements. Runtime DMA addresses/lengths and arbitrary sub-tile transfers are not admitted yet.

At most two transfers may be outstanding. Await/wait may complete them in either order and must consume each exact handle once; pending handles cannot cross blocks or survive block exit. Pure computation and explicit MXU operations may occur between issue and completion. Existing `virtual_input_*`, `virtual_output_bf16`, and `virtual_pack_fp8` are rejected while DMA is pending because their lowering uses implicit DMA or VMEM scratch. Completed store waits count as external outputs. In a CFG, both store issue and completion must be in the unique return block, as with the existing output convention.

The invocation must keep load-source memory stable and exclude conflicting external accesses to transfer ranges until completion. The stream verifier checks operations in this IR; it cannot enforce concurrent host behavior.

[`virtual_dma_tiles.mlir`](../test/examples/virtual_dma_tiles.mlir) demonstrates SSA address arithmetic and explicit load/store completion. Physical lowering assigns each pending transfer a private 2-KiB staging window and a free channel (loads prefer channel 0, stores channel 1). Loads launch at issue and await emits the matching wait before loading the tile; stores snapshot the tensor into staging before launching. Explicit DMA functions require the input/output base attributes, and transfers may not overlap the control mailbox.

The independent [DMA memory check](../include/Atlas/AtlasDMAMemoryVerification.h) recomputes CFG scalar constants and the effective byte ranges of emitted operands, permits read/read sharing, and requires proof for any access involving a write until the matching wait.

Generated explicit launches and waits carry matching `atlas.virtual_dma_transfer` IDs. Between them the generated-schedule checker permits independent scalar, VPU, MXU and proven-safe vector-memory work, and rejects channel reuse, mismatched waits, configuration changes and control-flow entry or exit.

### Explicit virtual MXU resources

`!atlas.virtual_mxu_weight<unit>` identifies a resident FP8 weight, and `!atlas.virtual_mxu_acc<unit>` identifies one accumulator version. Units are `0` or `1` and remain part of the selected arithmetic semantics. These handles have no physical register or slot numbers. Each unit admits two resident weights and two live accumulators within a block.

| Operation | Operands and results, in addition to the virtual state chain | Meaning |
| --- | --- | --- |
| `atlas.virtual_mxu_load_weight` | FP8 tile, `unit` attribute -> weight handle | Replace the selected unit's current weight. |
| `atlas.virtual_mxu_load_acc_fp8` | FP8 tile, `unit` attribute -> accumulator handle | Initialize an idle accumulator by decoding FP8; no scale operand. |
| `atlas.virtual_mxu_load_acc_bf16` | BF16 tile, `unit` attribute -> accumulator handle | Initialize an idle accumulator from a BF16 tile. |
| `atlas.virtual_mxu_reset` | FP8 activation, weight handle -> accumulator handle | Start a reset contraction; require no live accumulator on that unit. |
| `atlas.virtual_mxu_accumulate` | FP8 activation, weight handle, accumulator handle -> next accumulator handle | Continue contraction, consuming the current accumulator version. |
| `atlas.virtual_mxu_readout_bf16` | accumulator handle -> BF16 tile | Read the result and consume the current accumulator version. |
| `atlas.virtual_mxu_readout_fp8` | accumulator handle, virtual scale -> FP8 tile | Quantize the result with the selected raw scale code and consume the accumulator version. |

All seven operations consume and return the current `!atlas.virtual_state` and declare conservative read/write effects. The stream verifier rejects capacity overflow, stale or forked accumulator versions, and accumulators left live at block exit. A weight remains resident until its last source use; independent weights and accumulator chains may coexist. Accumulator loading preserves resident weights and can precede continuation without a reset contraction. Either readout leaves the current weight available for reuse. Handles cannot cross CFG edges, including implicit uses of a dominating handle in another block. BF16 readout values can use the existing BF16 block-argument convention.

`atlas.virtual_scale_constant {code = ... : i32}` produces an immutable `!atlas.virtual_scale` from a raw E8M0 code in `0..255`. It is pure and has no state operand or physical register number. FP8 readout explicitly consumes this scale value. The current slice accepts constant scale definitions, including dominating uses across blocks, but no scale block arguments or runtime scale inputs. These are selected hardware codes: the MXU readout's scaling behavior must not be inferred from VPU packing or replaced with a generic quantization rule.

Both readout forms intentionally use generated operand/result checks instead of a custom per-operation `verify()`. The accumulator type validates its unit; TableGen requires `!atlas.virtual_scale` for FP8 readout; the scale-constant producer validates its code. Readout has no separate unit attribute or second unit-bearing operand to compare. Current accumulator identity and consumption are checked by the stream verifier.

The existing reset-only `virtual_mxu_matmul` remains supported. Within a mixed stream it requires no live explicit weights or accumulators on its selected unit. Transformations of mixed streams must recheck this ordering contract; the existing convenience operation retains its original pure trait.

[`virtual_mxu_accumulation.mlir`](../test/examples/virtual_mxu_accumulation.mlir) demonstrates independent chains on both units. Lowering gives each weight and accumulator chain a free slot on its unit, outside tensor-register coloring, and rematerializes the FP8 scale code before every FP8 readout. [`virtual_mxu_seeded_fp8.mlir`](../test/examples/virtual_mxu_seeded_fp8.mlir) demonstrates accumulator initialization and FP8 readout.

### Generated-artifact checks

Lowering marks its output `atlas.generated_from_virtual = "resource-contract-v5"`, the only supported version, which must carry all six contract attributes (arrays may be empty) and an explicit `atlas.timing_state`. A module without marker, contracts or correspondence tags is a hand-written stream and receives no generated checks; every consumer applies [one classification](../include/Atlas/AtlasGeneratedArtifact.h) that rejects all other combinations, including the bare unit marker and earlier versions.

Contracts derive from live source, checked placements or earlier contracts, never from the lowering's plan.

| Contract and checker | Checks | Out of scope |
| --- | --- | --- |
| `atlas.virtual_dma_contract` ([DMA](../include/Atlas/AtlasDMAContractVerification.h)) | Launch-time operands and completion identities of explicit transfers | Implicit DMA, tile payloads |
| `atlas.virtual_mxu_contract` and `atlas.virtual_mxu_command` tags ([MXU](../include/Atlas/AtlasMXUContractVerification.h)) | Every MXU command's operands, formats, slots and logical version; FP8-readout scale contents; path-wise physical ownership | Tensor contents |
| `atlas.virtual_tile_contract` ([tile](../include/Atlas/AtlasTileContractVerification.h)) | Registers, addresses and required predecessor commands of every VLOAD/VSTORE, DMA launch/wait and mailbox LW on all emitted paths | Buffer contents between those commands, including PACK's scalar permutation |
| `atlas.virtual_cfg_contract` ([CFG](../include/Atlas/AtlasCFGContractVerification.h)) | Branch conditions, targets, operation visits and register contents, including each mailbox argument's i1 mask, across branches and loops | PACK memory layout |
| `atlas.virtual_source_memory_contract` ([source memory effects](../include/Atlas/AtlasSourceMemoryEffectContract.h)) | Earlier overlapping source DRAM effects of a block visit (explicit or implicit, with proven byte spans) complete before a later write-involving one issues | Anything but source DRAM effect order |
| `atlas.virtual_buffer_contract` ([buffer](../include/Atlas/AtlasBufferContractVerification.h)) | Each VMEM reader observes its writer's words (a VLOAD or mailbox LW its completed DMA load, a DMA store its VSTOREs, a PACK relayout VLOAD the row interleave of its raw VSTORE; any other VLOAD after a VSTORE is rejected), and each PACK conversion's scale register holds its source scale code | The asynchronous LW result (timing); RTL evidence for the interleave, which is the lowering's own PACK description |

Run `--lower-atlas-virtual-to-machine --insert-atlas-delays`, optionally with `--schedule-atlas-stream=insert-delays=false` between them, to time the untimed lowering output; both timing passes preserve contracts and command tags. Executable emission and LLVM handoff require `"timed"` state and the named `atlas.timing_provider`; hand-written streams need no timing metadata.

Timing passes resolve `provider=<id>` (default: the retained `atlas.timing_provider`, else `npu-model-rtl-match-v1`) through a process-wide registry, and timed output records it. Timing verification reconstructs issue cycles with that complete [`TimingProvider`](../include/Atlas/AtlasTimingProvider.h); missing rules, unknown ids and a mismatch with the retained provider fail. The registry is seeded only with the default; RTL-derived rules would register their own complete provider.

## Machine operations

All machine rows take and return `!atlas.state`. The fields shown are typed
attributes. Their verifier checks the listed forms, with mode-specific
bounded execution evidence in the census and source-discrepancy notes.

| Operation | Attributes and verified forms | Physical meaning and limit |
| --- | --- | --- |
| `atlas.start` | none | Begin stream; no encoded word. |
| `atlas.vload` | `dst, base, offset, format="raw"`; tensor destination, scalar base, signed 12-bit offset in 32-word units | Load one 1,024-byte VMEM tile into a tensor register. |
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
| `--verify-atlas-virtual-stream` | `atlas-opt` module pass | Check the bounded virtual BF16/FP8 stage's SSA types, CFG state edges, output indexes, and isolation from physical machine operations. It does not assign registers or emit words. |
| `--schedule-atlas-virtual` | `atlas-opt` module pass | Reorder virtual operations within each block before resource assignment, preserving SSA, handle and memory dependencies, by pressure-aware heuristic cost or a seeded random legal order. |
| `--lower-atlas-virtual-to-machine` | `atlas-opt` module pass | Verify one bounded virtual CFG; assign live BF16 pairs, FP8 registers, and scalar registers; stage input/output tiles and runtime controls; lower MXU and unit-scale pack; resolve BF16/scalar block-argument copies, branches and DMA waits; emit typed machine operations with resource contracts and `atlas.timing_state = "untimed"`, without fixed async, scalar-load or helper padding. |
| `--verify-atlas-generated-schedule` | `atlas-opt` module pass | Run the [generated-artifact checks](#generated-artifact-checks), matching DMA waits, protected transfer intervals, DMA memory conflicts and architectural NOP branch slots; timed streams also recheck actual issue spacing. Emission and LLVM conversion run the same checks. |
| `--verify-atlas-machine-stream` | `atlas-opt` module pass | Check local verifiers, flat state chain, selected word encoding, delay-slot adjacency, and in-block target confinement. Leave Atlas MLIR unchanged. It reuses encoder checks; it is not an independent hardware proof. |
| `--insert-atlas-delays` | `atlas-opt` module pass | Reject input that contains `atlas.delay`, then time each basic block in program order and insert the minimum delays the timing model requires, each with an `atlas.reason`. Branch and jump offsets are recomputed. The model is npu_model's `rtl-match` rules ported from atlas-compiler-experiments `3ae2b5d` (`AtlasTiming.cpp`), with DMA VMEM addresses counted in words as in Atlas RTL; it is not selected-RTL timing evidence. Fixed-latency work drains at block boundaries, a DMA wait may release at any time, and a hazard only a DMA wait can fix is an error (channel reuse is a warning). AUIPC, JALR, and linking JAL are rejected. `provider=<id>` selects the [timing provider](#generated-artifact-checks). |
| `--schedule-atlas-stream` | `atlas-opt` module pass | Same input, model, and checks as `--insert-atlas-delays`, but reorder each basic block with the greedy list scheduler ported from atlas-compiler-experiments `3ae2b5d` (`buildGraph`, `criticalHeights`, `scheduleBlock`), then insert the delays its issue cycles need. `insert-delays=false` reorders without padding and keeps the output untimed. Work moves only within a block and after everything it depends on; branches are re-aimed at the new first instruction of their target block. Delay slots are not filled. Under another `provider=<id>`, the model-graph schedule must pass that provider's verification or the pass fails. |
| `--verify-atlas-timing` | `atlas-opt` module pass | Check actual issue cycles, dependency distances, reservations, DMA wait handling, guarded halt and drained CFG boundaries with the selected `provider=<id>`, then retain timed state and provider provenance. Unknown timing states fail. |
| `--convert-atlas-to-llvm-calls` | `atlas-opt` module pass | Preserve each checked machine instruction as a separate `llvm.call @atlas_emit_*` with physical fields, encoded word, word index, conservative effects, and unknown availability. This intermediate requires finalization before LLVM IR translation; its calls are markers, not runtime functions. |
| `--finalize-atlas-llvm-calls` | `atlas-opt` module pass | Reconstruct and verify the typed Atlas stream from the LLVM calls, check every encoded word and control target, then emit one ordered LLVM inline-assembly block. Reject inconsistent fields, words, indexes, and malformed streams. |
| `--convert-atlas-to-llvm` | `atlas-opt` module pass | Replace checked stream with `llvm.func @atlas_program()` containing one side-effecting, ordered `llvm.inline_asm` word block and `llvm.return`. This is a reset-entry body, not a C-callable ABI. |
| `atlas-emit` | tool, not pass | Require timed state for generated artifacts; `--allow-untimed` permits inspection with resource and correspondence checks but no timing claim. Emit words or `--map-json` sidecar with source operation, attributes, word index, branch/delay-slot mapping, conservative effects, and `availability: unknown`. |
| `atlas-emit --program-json` | tool mode, not pass | Emit the checked physical words with typed fields and selected-PC control metadata for a separate functional model. It does not define numerical or timing semantics. |
| `atlas-boot-pack` | tool, not pass | Package one checked reset-entry ELF `.text` section with a narrow authored memory layout and manifest. |
| `export_llvm_handoff.py` | exporter, not pass | Produce numbered Atlas/LLVM MLIR, LLVM IR, RV32 assembly, relocatable object, linked ELF, disassembly, word map, and hashes for the bounded MLP/attention fixtures. |

No instruction selector, whole-target allocator, general Linalg-to-Atlas pass,
qualified timing model, or callable ABI is implemented here. The generated
virtual path is timed by the passes above. Other physical examples retain authored delays. Focused selected-core
tests cover a long SSA chain with physical pair reuse, a swap backedge,
branch merges, runtime mailbox control, two ordered outputs, actual
LLVM-produced object words, and memory guards. A 32-tile simultaneously live
case fails with an explicit
31-pair pressure diagnostic. These tests do not establish full target or SoC
qualification.

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
changing the offsets of internal targets. The handoff tests compare freshly
generated LLVM object bytes with the checked stream; they do not give LLVM knowledge of the
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

Contributors can add resource/availability annotation passes on the
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

### Adding a pass

Place the pass declaration under `include/Atlas/`, its implementation
under `lib/`, list the new source in `lib/CMakeLists.txt`, and register it in
`tools/atlas-opt.cpp`. `AtlasStreamVerification.cpp` is a minimal example of
that wiring. Operation definitions and local legality belong in
`include/Atlas/AtlasOps.td` and `lib/AtlasOps.cpp` if a new attribute or mode
is actually required. Preserve source-bound timing facts as attributes on
Atlas operations or structured LLVM calls. The word-map exporter carries
Atlas operation attributes to binary offsets; an LLVM-call transformation
that changes the stream also needs to update that map. The final block does
not retain individual operation attributes.

The physical IR is a **flat word stream** with branch offsets; the preceding
virtual IR has an MLIR control-flow graph. A future general scheduler must
account for backward branches and multiple executions
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
