# Plan: let explicit DMA stay in flight across PACK

Branch: `perf/attention-mlp-bottlenecks`, from `handwritten-implementation` at
`aea28c5` (PRs #23 and #17 merged). Status: plan, not implemented.

## Goal

Today every explicit DMA transfer completes before a PACK starts, and none
starts until the PACK ends. A PACK lowers to `vpu_pack`, a VSTORE, a 32-row
scalar relayout loop and a VLOAD, about 1,490 cycles with the DMA engine idle.
In the combined attention + MLP tile, three PACKs per tile and five FP8 loads
plus one BF16 store run strictly one after the other:

| Program (npu_model main, `9e6313b`) | Cycles | DMA busy | DMA and PACK overlap | DMA + PACK |
|---|---|---|---|---|
| attention + MLP, 1 tile | 10,136 | 52.5% | 13 cycles | 97% of cycles |
| attention + MLP, 4 independent tiles | 39,405 | 53.4% | 29 cycles | 99% of cycles |
| attention + MLP, 4 tiles, shared K/V/W0/W1 | 30,654 | 38.7% | 17 cycles | 98% of cycles |

The virtual scheduler is already within about 3% of the DMA + PACK bound, so
no ordering change helps until PACK stops draining DMA. A throwaway experiment
that lifted the drain everywhere, with verification bypassed, gave the same
output bytes and:

| Program | Before | After | Speedup | DMA busy after |
|---|---|---|---|---|
| 1 tile | 10,136 | 8,643 | 1.17x | 60.4% |
| 2 independent tiles | 19,772 | 15,119 | 1.31x | 68.7% |
| 4 independent tiles | 39,405 | 27,182 | 1.45x | 76.5% |
| 4 tiles, shared weights | 30,654 | 23,723 | 1.29x | 49.6% |

Those are upper bounds for this change. This plan does it with proofs: every
checker that rejects the overlap today accepts it for PACK, and only for PACK,
when it can show the PACK cannot touch any pending transfer's VMEM.

## Why the overlap is safe

- ScalarCore captures a DMA command's register operands at launch
  (ScalarCore.scala:489-509, DMA.scala:223-229). The PACK's register writes
  (the SELI scale entry, the x6 and x8 bases, the x10-x14, x16 and x17 loop
  registers) cannot change a pending transfer, whose operands were x4, x7
  and x9.
- `vpu_pack` reads and writes tensor registers only.
- The PACK's memory is fixed scratch: the raw conversion VSTORE at byte
  `packWord * 4` (0x20000, 1 KiB) and the relayout loop's `lw` from there and
  `sw` to `packRelayoutWord * 4` (0x20400, 1 KiB), read back by one VLOAD.
  Explicit staging windows are 8 x 2 KiB from byte `stagingWord * 4`
  (0x80000-0x83FFF). The ranges are disjoint.
- They are also in different 256 KiB VMEM banks (bank 0 against bank 2). In
  npu_model's port of VMEM.scala an LSU access denies the DMA only on the same
  bank, so the relayout does not even slow the transfer.

Nicolas should confirm the last two points against RTL before Phase 4 turns
the overlap on.

## Where the drain is enforced today

| Stage | File | Rule |
|---|---|---|
| Generated interval policy | `AtlasGeneratedSchedule.cpp` (`canOverlapAtlasGeneratedDMA`), `AtlasVerificationContext.cpp` (`verifyGeneratedStructure`) | While DMA is pending, only scalar ALU, delays, MXU, VPU unary/binary/reduce, VLI, XLU, VLOAD, VSTORE and SELI may issue. `vpu_pack`, `lw`, `sw` and branches are "unexpected instruction while DMA is pending". A redirect target may not enter a pending interval. |
| DMA memory | `AtlasDMAContractVerification.cpp` (`verifyAtlasGeneratedDMAMemory`) | Walks one block at a time: "DMA memory transfer crosses a basic-block boundary". The relayout loop is its own block. |
| Timing | `AtlasStream.cpp` (`checkAtlasStream`, `verifyAtlasTimedStream`) | Timing is checked block by block, so a transfer must complete in the block that starts it. |
| Wait insertion | `AtlasDMAWaitInsertion.cpp` | Completes every transfer before an instruction outside the interval policy, and at every branch target. Rejects any pending transfer at a delay slot. |
| Virtual scheduler | `AtlasVirtualScheduling.cpp` (`drainsDMA`) | Models PACK as waiting for every queued transfer. |
| Allocation | `AtlasVirtualAllocation.cpp` (`placeDMA`) | Completes every transfer at a PACK, freeing its channel and, for a store, its window. |

The stream scheduler (`AtlasScheduling.cpp`) uses the same interval policy
to order instructions after earlier waits. Its dependence graph already
handles a wait for a transfer launched in an earlier block (`IncomingDma` in
`AtlasTiming.cpp`), though nothing passes it the incoming footprints yet.

## Approach: admit a PACK region, nothing broader

Lowering already tags every instruction it emits for a PACK with
`atlas.virtual_cfg_helper = "pack"` and the PACK's `atlas.virtual_cfg_source`.
The buffer contract already recognizes the relayout as one block closed by its
own helper back-edge. It runs that loop from known entry registers (at most
256 iterations) and proves every `sw` lands inside the relayout scratch.

A **PACK region** is the run of instructions tagged with one PACK's source,
from its SELI through its relayout VLOAD, plus any `atlas.delay` the timing
pass places among them. It is admitted beside pending explicit DMA when:

1. **Shape.** It contains exactly one copy loop of the shape the buffer
   contract recognizes. It contains no DMA launch, wait or config, no other
   redirect, no JALR, CSR or halt. The loop's back-edge and target both lie
   inside the region, and nothing outside the region targets it.
2. **Memory.** Its VSTORE and VLOAD tile records and every `lw`/`sw` address,
   proved by the buffer contract's loop execution, lie inside the PACK's two
   scratch spans. Both spans are disjoint from every pending transfer's
   staging span.
3. **Intervals.** Every pending interval that reaches the region contains all
   of it. A transfer is never launched or completed inside a PACK region.

Anything else stays rejected as today: scalar memory and branches outside a
PACK region, a PACK whose scratch overlaps a pending staging window, a branch
into the loop from outside, a DMA inside the loop, and boundary-tile helper
DMA, which still drains.

One shared recognizer, `findAtlasPackRegions(const AtlasStream &)` in a new
`AtlasPackRegion.{h,cpp}`, returns each region's PC range, loop block,
source PACK and scratch spans. Every stage below calls it, so the checkers
and the passes cannot drift apart.

## Changes by stage

### Verification (Nicolas reviews)

- **Interval policy** (`verifyGeneratedStructure`, `canOverlapAtlasGeneratedDMA`):
  admit `VPUPackOp` outright (register-only). Admit `lw`, `sw`, the helper
  branch and its delay slot when they lie inside a PACK region. Narrow "redirect
  target enters a pending DMA interval" to allow a target whose redirect lies
  in the same PACK region, inside the same intervals. Reject a redirect that
  leaves an interval.
- **DMA memory** (`verifyAtlasGeneratedDMAMemory`): replace the per-block walk
  with a forward walk that carries pending transfers across fallthrough edges
  into and out of a PACK copy loop. The loop starts and completes nothing, so
  the pending set at its entry is the one at its exit. At a region, compare
  its two scratch spans with every pending staging span, as VLOAD and VSTORE
  are compared today. Any other block boundary with a pending transfer is
  still an error.
- **Buffer contract** (`AtlasBufferContractVerification.cpp`): add the
  missing bound on the copy loop's `lw` addresses: each must lie in the raw
  or relayout scratch span. Stores are already bounded. This is what makes
  the memory rule's span comparison cover every access the region makes.
- **Timing** (`checkAtlasStream`, `verifyAtlasTimedStream`): carry each pending
  transfer's completion footprint (its VMEM span and channel) into the
  successor blocks inside a PACK region. Check every instruction there against
  it with the provider's existing `dmaConflict`, as instructions in the
  launching block are checked now. Reservation tables stay per block: DMA
  completion is never timed by delays, only by `dma.wait`.

### Passes

- **`--insert-atlas-dma-waits`**: treat PACK-region instructions as admitted.
  At the copy loop's header, complete nothing. Change the delay-slot rule to
  reject only a completion that would land in a delay slot, not any pending
  transfer.
- **Stream scheduler** (`AtlasScheduling.cpp`): use the shared admission rule
  for its generated-DMA ordering edges. Pass `IncomingDma` footprints into
  `buildGraph` for the blocks after a PACK region, so a wait whose transfer
  started before the region constrains only the accesses that conflict with it.
- **Virtual scheduler**: `drainsDMA` keeps only boundary tiles. The PACK cost
  and the load lookahead are unchanged; with the drain gone, the lookahead
  search can place a load before a PACK whose result is used after it.
- **Allocation** (`placeDMA`): a PACK no longer completes transfers. Channels
  and windows stay held across it, so 8 channels and 8 windows cover more
  transfers at once. The existing eviction rules (oldest channel, oldest store
  window, else oldest load with its VLOADs flushed) are unchanged.
- **Lowering**: no change. Tags and scratch placement already give the
  recognizer what it needs. The register-allocation checker
  (`collectStagingLoads`) already models staging VLOADs at first use, which
  may now come after a PACK.

## Tests

Update, since they assert the drain:

- `test_virtual_dma_lowering.py::test_pack_drains_an_independent_load` becomes
  "a PACK leaves an independent load in flight": the wait comes after the
  relayout VLOAD, before the load's first use.
- `test_virtual_scheduling.py::test_stream_scheduling_keeps_every_wait_before_a_pack`
  becomes a check that both transfers may stay pending across the PACK and
  still schedule and verify.
- `test_dma_contract_verification.py::test_pending_configuration_memory_and_control_restrictions_remain`
  keeps its cases (a `lw`, a branch and a config outside a PACK region stay
  rejected) and gains PACK-region cases.

Add, each against a lowered stream with a load pending across a PACK:

- Accepted by every checker, timed by both timing passes, and identical
  across `--schedule-atlas-virtual=random-seed=N` orders.
- Rejected:
  - a relayout `sw` base moved into the pending staging window (buffer
    contract);
  - a `lw` moved there (new load bound);
  - a fixed placement whose pack scratch overlaps a staging window (DMA memory);
  - a branch from outside into the copy loop (interval policy);
  - a DMA launch or wait inserted inside the loop;
  - a VLOAD of the pending window inside the region (timing and DMA memory);
  - a completion forced into the loop's delay slot.
- C++ verification tests for the recognizer: a copy loop with one extra
  instruction, a second loop, or a missing helper tag is not a PACK region.

## Validation

1. Full ctest (`ATLAS_LLVM_BIN`, `ATLAS_LLD` wrapper), 104 skips as on main.
2. npu_model equivalence: every example main lowers, plus the combined
   attention + MLP programs (1, 2, 4 tiles, independent and shared weights),
   simulated on both the pinned npu_model and npu_model main with the same
   random inputs. Output bytes must match current main, and cycle counts must
   not regress anywhere.
3. Report cycles and DMA busy for the combined programs against the table
   above. The target is the experiment's numbers, 1.17x to 1.45x. Anything
   short of that points at a rule that is still too strict.

## Phases (one commit each, everything passing after each)

1. **Recognizer and checkers.** `AtlasPackRegion`, the interval policy, DMA
   memory walk, buffer contract load bound and timing carry. Tested with
   hand-edited streams that move a wait past a PACK. The compiler still emits
   the drain, so nothing it produces changes.
2. **Wait insertion and stream scheduler.** The admission rule, the loop
   header and delay-slot changes, `IncomingDma`. Where the virtual order
   already leaves a transfer pending at a PACK (a load before a PACK, used
   after it), its wait now moves past the PACK; update the drain tests here.
   Allocation still assumes the transfer completes at the PACK, which only
   makes it reuse that channel or window sooner; the wait inserter then
   completes the transfer before the reuse, so the stream stays correct.
3. **Allocation.** `placeDMA` keeps channels and windows held across PACK,
   so more transfers can be in flight at once.
4. **Virtual scheduler.** `drainsDMA` drops PACK; loads now move ahead of
   PACKs. Run the equivalence and cycle measurements.
5. **Docs.** `dialect-reference.md` and `virtual-program-lowering.md` describe
   the PACK region rule; this file records the measured results.

## Risks and open questions

- **RTL confirmation** of PACK-only scratch access and per-bank LSU/DMA
  arbitration (Nicolas), before Phase 4.
- **Timing across blocks.** Carrying footprints is new to the timing checkers.
  Phase 1 keeps it to PACK regions; general cross-block DMA (producer tile
  loops) is a separate change that can reuse it.
- **Pending stores.** A DMA store reads its staging window beat by beat until
  it completes. The PACK must write nothing there, which the span check covers
  for writes as well as reads.
- **Window pressure.** Holding windows across PACK may exhaust the 8 staging
  windows sooner. The allocator then flushes the oldest load's VLOADs early,
  which is correct but may cost some overlap. The cycle measurements will show
  whether that happens in the combined programs.

## Not in this change

- Making the relayout loop faster (temporaries, load/store order). That is
  the next item, and it keeps the loop a loop.
- Admitting transfers across other loops, such as a producer's tile loop.
- Overlapping boundary-tile helper DMA.
