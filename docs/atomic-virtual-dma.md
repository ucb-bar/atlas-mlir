# Atomic virtual DMA: one boundary between virtual and physical assembly

Branch: `feat/atomic-virtual-dma`, from `handwritten-implementation` at
`aa52abf` (PR #15). Status: Phase 1 is implemented.

## Goal

The two halves of the compiler should be separated by one clear boundary:

| | Virtual assembly | Physical assembly |
|---|---|---|
| Registers | SSA values, unlimited | Allocated physical registers, finite |
| Timing | None | `atlas.delay` from the timing model |
| DMA | Atomic: a load returns its tile, a store is complete | `atlas.dma` launch on a channel and staging window, later `atlas.dma_wait` |
| Control flow | CFG blocks with block arguments | Branches and jumps with an `addi x0, x0, 0` delay slot, edge copies |

Today three of the four rows hold. Delays and delay slots exist only after
`--lower-atlas-virtual-to-machine`, and virtual values are SSA. DMA does not:
the virtual IR exposes completion. This change makes virtual DMA atomic and
moves wait placement to the physical side, as its last step before timing.

## Where completion leaks today

PR #2 (`bcee90c`, "feat: add virtual DMA transfers and scalar validation",
Nicolas Reed, 2026-10-02) introduced explicit completion, "to make … DMA
completion explicit before physical placement". The PR presented it as an
initial interface for the virtualization contract discussion.

| Piece | What it does with completion |
|---|---|
| `include/Atlas/AtlasOps.td` | `virtual_dma_load_fp8/bf16` return a pending handle. `virtual_dma_await_fp8/bf16` turns the handle into the tile. `virtual_dma_store_*` return a handle that `virtual_dma_wait` completes. The handle types are in `AtlasTypes.td`. |
| `lib/AtlasVirtualVerification.cpp` (`VirtualDMATransfers`) | At most `kMaxPendingVirtualDMA = 2` pending. Completion before block exit. No pending transfer at a boundary tile or pack. Handles may not cross blocks. No pending transfers touching the same DRAM bytes. |
| `lib/AtlasVirtualScheduling.cpp` | `isLaunch` / `isCompletion`, plus a `SlotPool` of 2 that assigns slots in **source order** and turns them into hard edges. Implicit-memory edges pin packs among transfers. |
| `lib/AtlasVirtualAllocation.cpp` (`placeDMA`) | Channel and staging window held from launch to the completion op. |
| `lib/AtlasVirtualToMachine.cpp` (`completeDMA`) | Each completion op becomes `atlas.dma_wait` followed, for loads, by VLOADs from staging. |
| Generated-artifact checkers: tile, source memory effect, DMA allocation, register allocation (`DMAAwaitBasePolicy`) | Pair each launch with its virtual completion op. |
| `tools/atlas_virtual_evaluator.py`, `docs/virtual-ssa-interpreter-contract.md` | Model pending handles and the 2-transfer limit. |

The cost shows up in performance as well as in the contract. Measured on
`feat/combined-attention-mlp-dma-utilization`:
- **What it costs.** The producer must place every completion. The scheduler
  inherits source-order slot edges from those placements. A 4-tile program
  written tile by tile runs at 6,423 cycles per tile, against 4,716 when the
  producer hand-pipelines the same tiles. Each tile boundary stalls on the
  previous tile's store wait (about 1,000 cycles), then on the next tile's
  Kᵀ load (about 500).
- **What it's for.** Explicit handles let the evaluator and checkers reason
  about visibility. Those checks move to the physical side, where the waits
  are.

## Virtual IR after the change

```mlir
// Before: launch, then await
%io1, %ev = "atlas.virtual_dma_load_fp8"(%io0, %addr, %size)
    : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
%io2, %q = "atlas.virtual_dma_await_fp8"(%io1, %ev)
    : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)

// After: one atomic load
%io1, %q = "atlas.virtual_dma_load_fp8"(%io0, %addr, %size)
    : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
%io2 = "atlas.virtual_dma_store_bf16"(%io1, %y, %out, %bytes)
    : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
```

- **Removed:** `virtual_dma_await_fp8`, `virtual_dma_await_bf16`,
  `virtual_dma_wait`, and the types `!atlas.virtual_dma_load_fp8`,
  `!atlas.virtual_dma_load_bf16`, `!atlas.virtual_dma_store`.
- **Semantics are sequential.** A load returns the DRAM tile as of its
  position in the state chain; a store writes DRAM at its position. These are
  the semantics `virtual_input_*` / `virtual_output_bf16` already have.
- **The verifier keeps:** constant addresses, 32-byte alignment, whole-tile
  sizes, the address range, and no overlap with the control mailbox.
- **The verifier drops:** every pending-transfer rule, because nothing is
  pending at this level.

## Virtual scheduler: atomic DMA, scheduled as if it had a wait

Treating a load as instant would place its consumer right after it. The
physical wait would then stall the in-order frontend for the whole transfer.
So the scheduler keeps the load atomic in the IR, but costs it as what it
becomes: a one-cycle launch whose tile arrives later.

**Heuristic.** When a DMA load is followed by an MXU, VPU, pack or store that
uses its tile, put independent work between them. A DMA wait will be inserted
in front of that consumer.

In the existing `Machine` model (`AtlasVirtualScheduling.cpp`), a consumer
can't start until its producer's result is ready, and the list scheduler picks
the ready operation that starts earliest. So the heuristic comes down to the
cost:

```
cost(load)   frontend = 1 cycle (setup + atlas.dma)
             transfer starts = max(issue, DMA engine free)      // one serial engine
             ready = transfer start + dmaTransferCycles(bytes) + VLOAD latency
cost(store)  frontend = VSTORE staging; transfer queues on the engine
             its completion delays later conflicting DMA and the block end
start(op)    = max(frontend, unit free, ready(each operand's producer))
pick         = min(pressure overflow, start, -height, source index)
```

A consumer of an in-flight load therefore starts late, and any independent
operation with an earlier start is placed in front of it: other compute, other
loads, the next tile's work. If nothing independent is ready, the consumer is
placed anyway, and its physical wait stalls there, as the hardware would.

**Rules on top of the cost:**
1. **Launch early.** Critical-path height includes DMA latency, so loads that
   feed long chains go first. A load's start is the frontend time, so it's
   always eligible early.
2. **In-flight cap as a resource, not a rule.** Track loads and stores whose
   modeled completion is still in the future. A new launch starts no earlier
   than when fewer than `K` are outstanding (`K` = 8: channels and staging
   windows). This replaces the source-order `SlotPool`. It only keeps the
   physical side from forcing waits early, so exceeding it can't break
   correctness.
3. **DRAM ordering.** Two DMA ops are ordered only if their DRAM ranges
   overlap and one writes (`virtualDMAConflict` already computes this).
   Explicit DMA stays ordered against `virtual_input_*` /
   `virtual_output_bf16`, which use the ABI slots. The pack no longer orders
   against DMA at this level: whether a transfer may stay in flight across a
   pack is a physical question (see below).
4. **Register pressure.** A load's destination register is only needed from
   the VLOAD that the physical side sinks to the first use. Phase 1 counts the
   value live from the load, which is conservative. Phase 2 counts it from
   `ready`.

`feat/combined-attention-mlp-dma-utilization` already has most of the cost
model: its O2 change queues launches on the serial engine and makes awaits
start at transfer completion. With atomic loads, the await's cost moves into
the load's `ready`, and the `SlotPool` code goes away.

## Physical side: allocation, launches, then wait insertion

The pipeline after the change:

```
verify-atlas-virtual-stream
schedule-atlas-virtual            atomic DMA, latency-aware (above)
-------------------------------- virtual | physical -------------------------
lower-atlas-virtual-to-machine    registers, channels, staging; launches;
                                  VLOADs sunk to first use; branches with
                                  NOP delay slots; edge copies
insert-atlas-dma-waits            NEW: as-late-as-possible waits from hazards
insert-atlas-delays | schedule-atlas-stream
atlas-emit
```

### Allocation (`placeDMA`)

- **Channels and windows:** each transfer gets a channel and a staging window,
  least recently used first, from the 8 of each, in launch order.
- **No virtual end point needed:** wait insertion adds a wait before any reuse
  of a still-pending transfer's channel or window. With 8 of each and the
  scheduler's cap, reuse should be rare.
- **Destination registers:** Phase 1 allocates the tile's registers from the
  load op. Phase 2 lets the allocator define a DMA load's result at its first
  use.

### Lowering (`launchDMA`, replacing `completeDMA`)

- **Load:** at the op, set up registers and emit `atlas.dma`. Emit the staging
  VLOAD(s) into the destination register(s) right before the first machine
  instruction that reads them. If the first use is in another block, emit them
  at the end of the defining block in Phase 1; Phase 2 sinks them across
  blocks. If all eight staging windows hold unread loads, the oldest load's
  VLOADs are emitted before the launch that reuses its window.
- **Store:** VSTORE(s) into staging, then `atlas.dma`, at the op.
- **No waits for explicit DMA.** Launch and VLOADs carry
  `atlas.virtual_dma_transfer`, so checkers can pair them.

### `--insert-atlas-dma-waits` (new machine pass)

This pass runs a forward dataflow over the machine CFG, tracking the set of
pending transfers: channel, staging VMEM span, DRAM span, direction. For each
instruction, take its footprint from the same functions the checkers use
(`provider.footprint`, the DMA memory spans and, in Phase 2, the pack-loop
memory bounds). Insert `atlas.dma_wait ch` (tagged) immediately before the
first instruction that:
- reads or writes the transfer's staging span (the sunk VLOAD, a VSTORE into
  a reused window, pack scratch if not proven disjoint);
- launches on the same channel;
- is a DMA whose DRAM range overlaps, where either one writes;
- is boundary-tile or mailbox DMA;
- is the completion CSR or the halt;
- in Phase 1, ends the block. In Phase 2, pending sets propagate along CFG
  edges, and at joins the predecessors wait until their sets agree.

Today `checkAtlasStream` reports these exact conditions as errors ("a delay
cannot cover a DMA transfer, so add atlas.dma_wait first"). The new pass turns
that list of errors into a list of insertions, and the checkers stay as the
independent proof. `schedule-atlas-stream` can still move each wait later
within its block, toward its most critical user.

### Example

Kᵀ and Q loads feeding QKᵀ. The scheduler put the splat and constant setup
between the loads and their consumers. The wait pass then places each wait
just before the sunk VLOAD:

```
atlas.dma  ch0  Kᵀ          ; virtual_dma_load_fp8 (Kᵀ)
atlas.dma  ch1  Q           ; virtual_dma_load_fp8 (Q)
vli.all    m32, 0x3E83      ; independent work the scheduler moved up
...
atlas.dma_wait ch0          ; inserted: first staging read for Kᵀ
vload      m0, 0(x4)        ; sunk VLOAD
vmatpush.weight mxu0, m0
atlas.dma_wait ch1          ; inserted
vload      m1, 0(x4)
vmatmul    mxu0, m1
```

## Checkers and evaluator (Nicolas)

These checkers pair a launch with a virtual completion op, and need a new
attribution:
- the tile contract (`AtlasTileContractVerification.cpp`);
- the source memory effect contract (`AtlasSourceMemoryEffectContract.cpp`);
- the DMA allocation check (`AtlasDMAAllocationVerification.cpp`);
- the register allocation check (`DMAAwaitBasePolicy`);
- the CFG contract.

**Proposal:** a physical wait and its VLOADs take the launching load or store
as their source operation. A transfer then has one source and two or three
physical commands (launch, wait, VLOADs). The checks themselves don't change.
They become proofs that the inserted waits cover every hazard.

The evaluator (`tools/atlas_virtual_evaluator.py`) and
`docs/virtual-ssa-interpreter-contract.md` get simpler. A load reads DRAM and a
store writes DRAM in state order, with no pending bookkeeping, the same as the
boundary tiles. This is a contract change, so it goes to the virtualization
contract discussion that PR #2 deferred to.

## Migration

The rewrite is mechanical and preserves semantics:
- **Load:** delete each await, use the load's new tile result for the await's
  value, and replace the await's state result with its state operand. The load
  stays where it was launched, so its order against stores is unchanged.
- **Store:** delete each `virtual_dma_wait` and replace its state result with
  its state operand.

About 25 files reference the completion ops. The largest are
`test/test_virtual_dma.py` (24 references), `test/test_virtual_scheduling.py`
(20), `test/tile-contract-verification.cpp`,
`test/dma-allocation-verification.cpp`, `test/test_virtual_dma_lowering.py`,
`test/test_source_memory_contract.py`, `test/generated_fixture.py` and
`tools/atlas_virtual_evaluator.py`. The examples are `virtual_dma_tiles`,
`virtual_dma_mxu`, `virtual_dma_mxu_bf16` and `virtual_sched_overlap`.

- **Delete:** tests of the pending rules (2-transfer limit, completion before
  exit, barriers at boundary tiles and packs).
- **Add:**
  - scheduler tests that independent work lands between a load and its
    consumer;
  - wait-insertion tests, one per hazard kind, plus CFG joins;
  - random-schedule soundness runs through wait insertion.

## Phases

1. **Same behavior, new boundary.**
   - Atomic ops, verifier, evaluator, migration.
   - Scheduler cost model with load latency and the in-flight cap.
   - Lowering with sunk VLOADs.
   - Wait insertion per block, waiting before any pack (main has no
     pack-memory proof).
   - Expect main's current cycle counts or better; same outputs.
2. **Bring in the proofs from the DMA branch:**
   - CFG pending propagation and pack-loop memory bounds, so waits move across
     packs and blocks;
   - deferred register definitions;
   - delete `kMaxPendingVirtualDMA` and the `SlotPool` DMA path.
3. **Measure:**
   - the combined tile should match or beat 6,361 cycles (56.7%);
   - naive multi-tile programs should approach the hand-pipelined numbers
     (4,716 cycles per tile at 4 tiles, about 88% for a long stream).

**Base branch:** main doesn't yet have the combined-attention and
DMA-utilization work: VPU reduce/splat, attention examples, the pipeline
driver, DMA-across-pack proofs, the simulator. Implementing phase 2 on top of
that work, by merging it to main first or rebasing this branch onto it, avoids
doing the proofs twice.

## Other leaks across the boundary (not in this change)

- **MXU slots:** the virtual verifier enforces 2 weight and 2 accumulator
  slots per unit (`kVirtualMXUSlots`). That's a finite resource checked at
  the virtual level, like the DMA limit this change removes.
- **MXU unit choice:** the source picks the unit (`unit = 0 | 1`) on every MXU
  op.
- **Pack scale code:** the lowering admits scale code 127 only.

## Validation

- `ctest` passes.
- Bit-identical outputs, on random inputs, for every lowerable example before
  and after (the method in `docs/dma-utilization-scheduling-results.md` on the
  DMA branch).
- Cycle counts per example, before and after, in both timing modes.
- Wait insertion reports where each wait went and why (the hazard), the way
  `atlas.delay` records its `atlas.reason`.
