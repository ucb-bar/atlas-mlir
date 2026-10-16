# Explicit DMA in flight across PACK

Branch: `perf/attention-mlp-bottlenecks`, from `handwritten-implementation` at
`aea28c5` (PRs #23 and #17 merged). Status: implemented.

## Goal

Every explicit DMA transfer used to complete before a PACK started, and none
started until the PACK ended. A PACK lowers to `vpu_pack`, a VSTORE, a 32-row
scalar relayout loop and a VLOAD, about 1,490 cycles with the DMA engine idle.
In the combined attention + MLP tile (three PACKs, five FP8 loads and one BF16
store per tile) DMA and PACK ran strictly one after the other:

| Program (npu_model main, `9e6313b`) | Cycles | DMA busy | DMA and PACK overlap | DMA + PACK |
|---|---|---|---|---|
| attention + MLP, 1 tile | 10,136 | 52.5% | 13 cycles | 97% of cycles |
| attention + MLP, 4 independent tiles | 39,405 | 53.4% | 29 cycles | 99% of cycles |
| attention + MLP, 4 tiles, shared K/V/W0/W1 | 30,654 | 38.7% | 17 cycles | 98% of cycles |

The virtual scheduler was within about 3% of that DMA + PACK bound, so no
ordering change could help until PACK stopped draining DMA.

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
  (0x80000-0x83FFF). The checkers prove this per program rather than assume
  it (below).
- The ranges are in different 256 KiB VMEM banks (bank 0 against bank 2). In
  npu_model's port of VMEM.scala an LSU access denies the DMA only on the same
  bank, so the relayout does not slow the transfer. Nicolas should confirm the
  bank behaviour against RTL.

## Design

One concept, the **PACK relayout loop** ([`AtlasPackLoop.h`](../include/Atlas/AtlasPackLoop.h)),
replaces "PACK drains DMA" everywhere:

- **Recognition** (`findAtlasPackLoops`) works on the flat instruction list, so
  the structural verifier can use it before decoding. A loop's back-edge is a
  `"pack"` helper branch to an earlier instruction with the exact NOP in its
  delay slot; everything from its target to the branch is that PACK's scalar
  loads, stores and ALU work, or a delay; no other redirect lands in it; and a
  stream with a JALR has none. Such a loop launches and completes no DMA and has
  no other entry, so the transfers in flight when it starts stay in flight
  through every iteration and when it exits.
- **Decoded facts.** `readAtlasStream` records each loop in
  `AtlasStream::packLoops` with its block, the registers it is entered with (the
  exit of the block that falls into it) and `access`, the VMEM it reads and
  writes over all iterations, proved by running it (`proveAtlasPackLoopAccess`)
  with the timing model's own scalar address rule. `AtlasStream::dmaMayCross`
  says whether explicit DMA may be in flight from one block into the next:
  only into or out of a PACK loop.
- **One loop interpreter**, `runAtlasPackLoop`, serves the access proof, the
  timing check and the buffer contract, which used to carry its own copy.

The stages then ask those two questions instead of assuming a transfer ends in
its block:

| Stage | Change |
|---|---|
| Interval policy (`canOverlapAtlasGeneratedDMA`, now PC-aware) | Admits `vpu_pack` and every instruction of a PACK loop beside a pending transfer. |
| Structural verifier (`verifyGeneratedStructure`) | Uses that policy; a PACK loop's back-edge may land inside the intervals that hold the loop, which they always hold whole since intervals start and end at DMA instructions. |
| DMA memory (`verifyAtlasGeneratedDMAMemory`) | Carries pending transfers across `dmaMayCross` boundaries and compares each PACK loop's proved access with every pending staging span; an unknown access fails as "cannot prove DMA memory disjointness". |
| Timing input check (`checkAtlasStream`) | Carries pending footprints the same way; inside a PACK loop it runs the loop and checks each instruction, with that iteration's registers, through the provider's own footprint and DMA conflict rules. |
| Timed verification (`verifyAtlasTimedStream`) | Carries pending transfers across `dmaMayCross` boundaries. |
| `--insert-atlas-dma-waits` | Decodes the stream for its loops; at a loop's first instruction it completes only the transfers whose staging the loop touches, never at its header otherwise; it rejects a completion that would land in a delay slot, not any pending transfer there. |
| Stream scheduler | Uses the PC-aware policy for its generated-DMA ordering edges. |
| Allocation (`placeDMA`) | A PACK no longer completes transfers; channels and windows stay held across it. |
| Virtual scheduler | `drainsDMA` is gone: only boundary tiles (`isImplicitMemory`) drain, so loads move ahead of PACKs. |
| Lowering | No change: PACK's helper tags and fixed scratch already give the recognizer what it needs. |

Everything else stays rejected as before: scalar memory and branches outside a
PACK loop, a loop whose VMEM meets a pending staging window or cannot be
proved, a loop holding a DMA instruction or entered from elsewhere, and
boundary-tile helper DMA, which still drains.

### Differences from the plan

- The plan had the buffer contract bound the copy loop's `lw` addresses, so
  the DMA memory check could compare the PACK's scratch tiles. Instead the
  decoded stream proves what each loop actually touches, and the DMA memory and
  timing checks compare that directly, so no checker depends on another's
  lemma.
- Checkers and wait insertion landed together: they share one admission rule,
  and splitting them would have needed a temporary half rule.
- The stream scheduler does not yet receive `IncomingDma` footprints for
  transfers launched before a PACK loop. A wait for such a transfer orders
  every later VMEM access after it, which is conservative and correct.

## Tests

- `pack-loop` C++ suite: recognition of the lowered loop, before and after
  timing; the facts the decoded stream adds (block, entry registers, 1 KiB
  raw-scratch reads and relayout writes, `dmaMayCross`); and each refusal
  (untagged or foreign body instruction, forward branch, non-NOP delay slot,
  scale load, second entry), plus an undecidable loop's unknown access.
- `test_dma_contract_verification.py::PackOverlapVerificationTest`: a load
  pending across a PACK is accepted at every boundary and by both timing
  passes; rejected when the relayout writes or reads the pending window, when
  its row limit is unknown, when a wait sits in its body and when a branch
  enters it.
- `test_virtual_dma_lowering.py::test_pack_leaves_an_independent_load_in_flight`
  and `test_virtual_scheduling.py::test_transfers_stay_pending_across_a_pack_through_stream_scheduling`
  replace the two tests that asserted the drain.
- `test_virtual_scheduling.py::test_loads_move_ahead_of_independent_packs`
  fails on the previous scheduler and passes now.

## Results

Each program lowered through `tools/lower_atlas_program.py` (every verifier,
`--timing schedule-atlas-stream`) with main (`aea28c5`) and with this branch,
then simulated with the same random inputs on the pinned npu_model
(`0c4a1f9`) and on npu_model main (`9e6313b`, beat-level DMA calibrated
against VCS). Output bytes match main for every program on both models.

Combined attention + MLP, npu_model main:

| Program | Main | Branch | Speedup | DMA busy, main → branch |
|---|---|---|---|---|
| 1 tile | 10,136 | 8,610 | 1.18x | 52.5% → 60.8% |
| 2 independent tiles | 19,772 | 15,119 | 1.31x | 53.4% → 68.7% |
| 4 independent tiles | 39,405 | 27,182 | 1.45x | 53.4% → 76.5% |
| 2 tiles, shared weights | 16,964 | 13,745 | 1.23x | 44.2% → 54.0% |
| 4 tiles, shared weights | 30,654 | 23,690 | 1.29x | 38.7% → 49.8% |

These equal the upper bound measured earlier by lifting the drain with
verification bypassed. On the pinned npu_model the same programs gain 1.12x
to 1.40x.

Two-layer MLP tiles with explicit DMA, npu_model main: 1.07x (1 tile) to
1.30x (6 tiles) with per-tile weights, reaching 99.9% DMA busy; 1.07x to 1.28x
with shared weights.

Examples: programs without a PACK, or whose transfers already completed before
it, compile exactly as on main. `virtual_fp8_two_layer_mlp(_bias)` gain about
1%. `virtual_attention_tile` gains 0.6% on npu_model main and loses 0.8% (35
cycles) on the pinned model: its virtual schedule is unchanged, and the stream
scheduler now interleaves V's wait and staging VLOAD with the PACK's prologue.

## Next

- With DMA overlapping PACK, PACK itself bounds shared-weight programs: each
  still costs about 1,490 cycles. The relayout loop can drop to about 1,200
  with more temporaries and its loads ahead of its stores, keeping it a loop.
- `IncomingDma` footprints for the stream scheduler (above).
- Admitting transfers across other loops, such as a producer's tile loop,
  can reuse `dmaMayCross` and the loop interpreter.
