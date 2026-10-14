# Raising DMA utilization of the combined attention + MLP tile

Branch: `feat/combined-attention-mlp-dma-utilization`, from
`feat/combined-attention-mlp-lowering` at `106c9bb`.

**Status (2026-10-13):** O1, O2, O3 and the in-loop part of O4 are committed
on this branch. The combined tile runs in 6,361 cycles at 56.7% DMA
utilization with the default `schedule-atlas-stream` timing, up from 8,817
cycles and 40.9%. O5 was measured slower and rejected; O6 is open. Results
and verification: [`dma-utilization-scheduling-results.md`](dma-utilization-scheduling-results.md).

This is a work list for parallel agents. Each item below can be picked up on
its own: it says where the code is, what was measured, what to change, and
when it is done. Read the shared sections first (context, measurement,
analysis, conventions).

## Context

- **Program:** [`test/examples/virtual_attention_mlp_tile.mlir`](../test/examples/virtual_attention_mlp_tile.mlir),
  a stand-in for the combined program until it lands: the attention tile
  (QKᵀ on MXU0 → softmax → FP8 pack → PV on MXU1) → unit-scale BF16→FP8 pack →
  the two-layer MLP (MXU0 → ReLU → pack → MXU1). It loads five 1 KiB FP8 tiles
  (Q, Kᵀ, V, W0, W1) and stores one 2 KiB BF16 tile.
- **Pipeline:** [`tools/lower_atlas_program.py`](../tools/lower_atlas_program.py)
  runs verify → `--schedule-atlas-virtual` → `--lower-atlas-virtual-to-machine`
  → `--insert-atlas-delays` (or `--schedule-atlas-stream`) → `atlas-emit`.
- **Simulator:** [`tools/simulate_atlas_program.py`](../tools/simulate_atlas_program.py)
  runs the emitted words in npu_model. It decodes each word back into an
  npu_model instruction and checks that it re-encodes to the same word, then
  reports cycles and per-unit utilization. `--profile` attributes every cycle to
  the virtual operation holding the in-order S1 stage. It adapts three npu_model
  behaviors to the RTL; see its docstring.

### The metric

DMA utilization = DMA0 busy cycles / total cycles. The busy cycles are fixed
by the bytes moved: 5 × 516 (1 KiB loads) + 1,028 (2 KiB store) + 1 (config)
= **3,609 cycles**. npu_model's DMA engine is a single in-order queue, so
transfers on different channels do not overlap. **Utilization rises only by
cutting total cycles.** Report cycles and utilization together. Changes that
move fewer bytes (FP8 output, weight reuse across tiles) lower utilization
but are still wins. Judge them by cycles, or cycles per tile.

## Measure

Use the venv Python (`~/.venvs/atlas-mlir/bin/python`, which has npu_model
`main` at `0c4a1f9`). The driver needs a new, empty output directory each run.

```sh
ninja -C build
python tools/lower_atlas_program.py test/examples/virtual_attention_mlp_tile.mlir \
  --output-dir out/dma/<name> [--timing insert-atlas-delays] [--atlas-bin-dir <bin>]
python tools/simulate_atlas_program.py out/dma/<name> --profile [--json out/dma/<name>/sim.json]
```

The simulator exits nonzero unless the program halts on ECALL with zero
runtime errors. Also run the attention-only tile
(`test/examples/virtual_attention_tile.mlir`, baseline 4,465 cycles, 57.7%) to
check that a change does not regress it.

## Results so far

| Configuration | Cycles | DMA util. | Notes |
|---|---|---|---|
| Baseline (`insert-atlas-delays`) | 8,817 | 40.9% | |
| `--timing schedule-atlas-stream` | 8,458 | 42.7% | item O3 |
| Pack barrier relaxed (prototype) | 7,946 | 45.4% | item O1, unsound prototype |
| Pack barrier relaxed + `schedule-atlas-stream` | 7,612 | 47.4% | O1 + O3 |
| O1 + O2 + O3 + O4, `insert-atlas-delays` (committed) | 6,574 | 54.9% | sound O1, pack loop reordered |
| O1 + O2 + O3 + O4, `schedule-atlas-stream` (committed default) | **6,361** | **56.7%** | |
| O5 alone, `schedule-atlas-stream` | 8,553 | 42.2% | rejected: slower than baseline |
| Source: W0/W1 loaded and awaited before attention | 8,919 | 40.5% | worse: the serial loads delay QKᵀ |
| Source: W0 issued early, awaited after the pack | rejected | | "must complete pending DMA before implicit memory operations" |

**Ceilings for one tile.** Two resources bound the tile: the DMA engine
(3,609 busy cycles) and the in-order frontend.
- **With the pack at about 1,130 cycles (committed):** the frontend is
  occupied about 4,400 cycles, of which the packs take about 3,380. On top of
  that it must wait about 1,550 cycles: for the first tiles to load, and for
  the final store. That puts the ceiling at about 60%.
- **The relayout loop's floor:** in the timing model a scalar `lw` issues at
  most every 3 cycles, so a row of 8 loads takes at least about 27 cycles, or
  about 1,000 cycles per pack. That leaves only about 350 cycles across three
  packs. (This doc first estimated 800 per pack and 67%, assuming one `lw`
  per cycle; that was wrong.)
- **With no relayout** (a pack result the MXU can use directly; needs a
  hardware/ISA change): about 95%. The DMA then runs continuously, except for
  about 190 cycles of MLP1 and output staging before the store.
- **Across tiles:** the Q/K load at the start (about 1,030 cycles) and the
  store at the end (about 1,130) are exposed for one tile no matter what.
  Only overlapping several tiles hides them (O6).

## Analysis at the end of each pass (baseline)

**1-virtual (source).** W0/W1 are loaded after the attention part.
The virtual verifier forbids any explicit DMA pending at a
`virtual_pack_fp8` (`lib/AtlasVirtualVerification.cpp`, "must complete
pending DMA before implicit memory operations"). So a load can only overlap the
short compute stretches between packs.

**2-scheduled (`--schedule-atlas-virtual`).**
- **What it improved:** it loads K before Q, so the K weight push runs while
  Q arrives.
- **V:** V is issued after Q's await, because at most two transfers may be
  pending (`kMaxPendingVirtualDMA = 2`).
- **W0/W1 stay put:** they remain between the attention-output pack and the
  MLP, because `isImplicitMemory` in `lib/AtlasVirtualScheduling.cpp` adds an
  edge from every pack to every transfer launch and completion. That pins each
  pack in its source position among the transfers.

Under today's rule this schedule is essentially optimal. The only other legal
placement, before the first pack, measured worse.

**3-allocated (`--lower-atlas-virtual-to-machine`).** The program fits with
room to spare: BF16 4/15, FP8 2/32, scalar 7/9, MXU 1 weight + 1 accumulator
slot per unit, 2 DMA in flight of 8 channels, 195 of 2,048 untimed words. Each
pack lowers to `vpu_pack`, a VSTORE of the raw result, and then a 32-row scalar
relayout loop. Each row is 8 × (`lw`, `sw`) + 3 pointer `addi` + counter
`addi` + `blt` + delay-slot NOP, through two temporaries (x16, x17). x15 is
inside the pack's reserved x10..x17 but unused.

**4-timed (`--insert-atlas-delays`).**
- **Inside the relayout loop:** every `lw` is followed by `delay 2` (RAW
  into its `sw`), 8 per row and 256 per pack.
- **Before each loop:** a 26-cycle drain ("this block finishes before the
  next one starts").
- **Elsewhere:** the softmax chain and MXU ops are separated by their
  latencies (30–64 cycles each).

`--schedule-atlas-stream` reorders within blocks and saves 359 cycles.

**Simulation (`--profile`), where the 8,817 cycles go:**

| Phase | Cycles | DMA busy during it |
|---|---|---|
| Three packs (`vpu_pack` + relayout loop), about 1,610 each | 4,828 | 0 |
| DMA.WAIT stalls: K/Q head 963, V 51, W0 511, W1 357, final store 1,028 | 2,910 | all |
| Attention MXU + softmax | about 700 | yes (V loading) |
| MLP matmuls + ReLU | 195 | partly |
| Output VSTOREs before the store launch | 100 | no |

The avoidable DMA exposure is V + W0 + W1, about 920 cycles. The DMA is idle
for the whole of every pack. Packs take 55% of the runtime.

## Work items

Priority order. O1, O2 and O5 all touch `lib/AtlasVirtualScheduling.cpp`.
O1, O3 and O4 all affect timing. Work on separate branches and expect to
rebase. To see compounding gains before O1 lands, apply the
[prototype patch](#appendix-o1-prototype-patch) locally. It is for
measurement only and must not be committed.

### O1. Let explicit DMA stay pending across `virtual_pack_fp8`

**Measured:** 8,817 → 7,946 cycles (45.4%); 7,612 (47.4%) with O3. The profile
shows W0 loading during the first pack, W1 during the second, and only the
Q/K head and store tail still exposed.

**Why it is safe in principle.**
- **VMEM:** the pack touches only its scratch, VMEM words 32768..33279 (raw
  stage and relayout). DMA staging windows start at word 131072, 512 words
  each.
- **Registers:** the pack uses x3, x6, x8, x10–x14, x16 and x17. The DMA
  helpers are x4, x7 and x9. DMA captures its operands at launch anyway.
- **Implicit DMA:** the pack issues no DMA. `virtual_input_*` and
  `virtual_output_bf16` do, so they keep the barrier.

**Seven places enforce the barrier.** The prototype changed each of them; see
the appendix.
1. `lib/AtlasVirtualVerification.cpp` (~line 238): the virtual rule. Drop
   `VirtualPackFP8Op` from it.
2. `lib/AtlasVirtualScheduling.cpp` `isImplicitMemory`: the pack↔transfer
   ordering edges.
3. `lib/AtlasGeneratedSchedule.cpp` `canOverlapAtlasGeneratedDMA`: the
   machine ops allowed while DMA is pending. It excludes `vpu_pack`, `lw`,
   `sw` and branches.
4. `lib/AtlasVerificationContext.cpp` `verifyGeneratedStructure`: "generated
   redirect target enters a pending DMA interval". The loop back-edge
   targets inside the interval.
5. `lib/AtlasDMAContractVerification.cpp` `verifyAtlasGeneratedDMAMemory`:
   per block, "DMA memory transfer crosses a basic-block boundary". The loop
   splits the machine block.
6. `lib/AtlasStream.cpp` `checkAtlasStream`: per-block pending footprints,
   "may still be in flight when its block ends".
7. `lib/AtlasStream.cpp` `verifyAtlasTimedStream`: "timing verification
   requires DMA completion before a block boundary". Also review the
   cross-block wait handling in `lib/AtlasTiming.cpp` (~lines 900–935).

**What the sound version needs, unlike the prototype.**
- **Prove VMEM disjointness.** The relayout loop's `lw`/`sw` base registers
  are loop-carried, so their addresses are unknown and the timing check flags
  them as conflicting with any pending transfer. The prototype exempts every
  instruction tagged `atlas.virtual_cfg_helper`. Replace that with a proof.
  Either give the pack loop a summarized footprint, the pack scratch range
  already recorded in `atlas.virtual_buffer_contract`, or bound x10–x12 with a
  simple induction range. Then check that footprint against every pending
  staging window.
- **Handle loops in the machine CFG.**
  - Allow a pending interval to span a loop only if the loop contains no
    launch or wait for that interval.
  - Allow a back-edge only if both the branch and its target are inside the
    same interval.
  - Carry pending transfers along CFG edges, not in layout order as the
    prototype does.
- **Admit `vpu_pack` and pack-footprint `lw`/`sw`, not every `lw`/`sw`.**
- **Keep the barrier for `virtual_input_*` and `virtual_output_bf16`.**

**Done when:**
- `ctest` passes.
- New tests cover:
  - a positive lowering in which a transfer stays pending across a pack;
  - negative cases: a VMEM window overlapping the pack scratch, a pending
    transfer at an input or output tile, and a launch or wait inside the loop;
  - `--schedule-atlas-virtual=random-seed=N` orders, which must still verify.
- The combined tile measures about 7,950 cycles or fewer.

**Coordinate:**
- Nicolas's CFG and tile contract checkers model DMA intervals and may need the
  same change.
- Open RTL question: does AtlasCore allow scalar LSU and VLOAD/VSTORE access to
  one VMEM bank while a DMA writes another? Get RTL evidence before the FPGA
  run.

### O2. Model the DMA queue in the virtual scheduler, and fix its cost inputs

`Machine::start` in `lib/AtlasVirtualScheduling.cpp` gates every operation on
its unit being free. So the model says a DMA launch cannot issue until the
engine finishes the previous transfer, and in-order issue then delays
everything after it. The hardware queues up to 8 commands; only the transfers
serialize.

- **Model the queue:** a launch takes 1 frontend cycle. Its transfer starts at
  max(issue, engine free) and completes `dmaTransferCycles` later.
- **Check the priority:** the list scheduler's key orders by modeled start
  time before critical-path height. Make sure launches go as early as possible
  and awaits as late as possible.
- **Update the pack cost:** `costOf(VirtualPackFP8Op)` estimates vpack +
  vstore + 256 × (lw + sw latency) + vload. The measured cost is about 1,610
  cycles (about 1,510 with `schedule-atlas-stream`). The estimate leaves out
  the per-row loop overhead and the block drain. Calibrate it against the
  simulator.
- **The same goes for the await cost:** the await is modeled on the LSU
  only, but the DMA.WAIT stalls the frontend.

**Measure first.** With the O1 prototype, today's scheduler already placed the
W0 and W1 launches before the packs. The gain may appear only in programs with
more transfers than pending slots. Report before and after with the O1 patch
applied.

**Done when:** scheduler tests pass (`test/test_virtual_scheduling.py`), the
random-seed soundness runs still verify, and no example program gets slower.

### O3. Make the machine stream scheduler the default timing pass

`--timing schedule-atlas-stream` saves 359 cycles on the baseline and 334 on
top of O1, and passes `--verify-atlas-timing` and
`--verify-atlas-generated-schedule`. Most of the gain is in the pack loop
(delay per pack drops from about 631 to 544 cycles).

**To do:**
1. Confirm it passes the whole test suite as the default in
   `tools/lower_atlas_program.py`. Check `test/test_lower_atlas_program.py` for
   anything that assumes `insert-atlas-delays`.
2. Find what it reorders in the loop body. If that can be expressed in the
   lowering's instruction order instead, fold it into O4.

This is machine-level and cheap.

### O4. Cut the pack relayout cost (lowering)

This is the largest single lever, though it sits below the virtual stages.
Packs are 55% of runtime. The code is `lowerPack` in
`lib/AtlasVirtualToMachine.cpp`, about lines 542–605. The loop stays a loop:
do not unroll it (see [Conventions](#conventions)).

**Where the time goes.** Each of the 32 rows issues 22 instructions: 8 `lw`,
8 `sw`, 3 pointer `addi`, a row-counter `addi`, `blt` and the delay-slot NOP.
It also issues 8 `delay 2`, one for the load-use gap between each `lw` and its
`sw`. That is about 44 cycles per row. With only two temporaries (x16, x17),
each `sw` has to wait for the `lw` just before it.

**Options:**
- **More temporaries.** x15 is reserved for the pack and unused. With three or
  more temporaries, each `sw` can trail its `lw` by three instructions, so
  loads and stores alternate with no `delay`. Either emit that order in the
  lowering, or let `--schedule-atlas-stream` (O3) find it. It already reorders
  this loop body; reusing two temporaries is what limits it today.
- **Fewer induction instructions.** The two source halves are 512 bytes apart.
  One source pointer with offsets 0–12 and 512–524 can replace two, which also
  frees x11 as another temporary. Exiting on a pointer compare instead of a
  row counter removes the counter `addi`. That leaves 16 memory ops + 2 `addi`
  + `blt` + NOP = 20 instructions per row.
- **Remove the 26-cycle drain before the loop**, if the VSTORE hazard can be
  covered another way.

**Target (corrected):** a scalar `lw` issues at most every 3 cycles in the
timing model, so a row floors at about 27 cycles and a pack at about 1,000.
The 20-cycles-per-row figure assumed one load per cycle and is not reachable
without faster scalar loads.

**Status:** done within the loop. The committed loop rotates x16/x17/x15
through the loads and stores and keeps one row per iteration. It measures
about 1,130 cycles per pack (from about 1,610), within about 4 cycles per row
of the floor.

**Limits:**
- `atlas.virtual_buffer_contract` must still verify the row interleave.
- Update the scheduler's pack cost (O2) to match.

**Done when:** the pack-related lowering tests and the buffer-contract tests
pass, and each pack measures ≤ about 1,000 cycles in `--profile`.

### O5. Split legacy `virtual_mxu_matmul` before scheduling

The MLP uses the legacy `virtual_mxu_matmul`. It is costed as a blocking
push → multiply → pop (98 and 65 frontend cycles here), and `effectsOf` gives
it a write of its whole MXU unit (`kUnit0`/`kUnit1`). The scheduler therefore
cannot push W0/W1 into the weight buffer during the preceding pack.

Rewrite it before `--schedule-atlas-virtual` into
`virtual_mxu_load_weight` / `virtual_mxu_reset` / `virtual_mxu_readout_bf16`,
the form the attention part already uses. That could be a canonicalization or
a step at the start of the scheduler. The weight push can then run as soon as
its tile is awaited.

**Expected gain:** small, roughly 30–60 cycles per matmul. It compounds with
O1, since the weight push can follow the await before the pack.

**Watch:** the MXU slot limits (2 per unit), and keep the legacy op valid as
input.

**Done when:** `test/test_virtual_mxu*.py` and the lowering tests pass, and
the combined tile gets faster with the O1 patch applied.

### O6. Overlap tiles (program structure)

For one tile, about 2,100 cycles of DMA stay exposed: the Q/K head and the
store tail. Real workloads run several tiles: heads, query blocks, tokens.
The fix is to prefetch tile i+1's inputs while tile i computes, and store
tile i while tile i+1 runs. W0/W1 stay resident, and K/V stay resident across
query tiles.

**Scope.** The tile structure comes from whatever produces the virtual
program, not from atlas-mlir. atlas-mlir does not unroll or software-pipeline
the program's loops. Its job here is to accept and schedule programs in which
one tile's transfers overlap another tile's compute.

**Current limits in atlas-mlir:**
- Pending DMA handles cannot cross CFG blocks, and must complete before block
  exit (`lib/AtlasVirtualVerification.cpp`).
- `kMaxPendingVirtualDMA = 2` (`include/Atlas/AtlasVirtualVerification.h`),
  while the allocation report lists 8 channels and 8 staging windows.

**Steps:**
1. Hand-write a two-tile example,
   `test/examples/virtual_attention_mlp_2tile.mlir`, standing in for what a
   producer would emit: tile 2's loads interleaved into tile 1's compute. It
   needs O1. Measure cycles per tile, and record which atlas-mlir rules
   reject it.
2. Lift those limits:
   - Raise the pending-transfer limit. Check the staging and channel
     allocation in `lib/AtlasVirtualAllocation.cpp`.
   - For producers that emit a loop over tiles, allow transfer handles to be
     carried across the loop's back-edge, without unrolling it.
3. Agree the program shape with the producer (Agustin).

Report cycles per tile as well as utilization, because weight residency lowers
DMA busy cycles.

## Conventions

- **Build:** `ninja -C build`. A separate worktree builds in about 11 seconds:
  `cmake -S <tree> -B <tree>-build -G Ninja
  -DMLIR_DIR=~/Projects/llvm-project/build/lib/cmake/mlir
  -DCMAKE_BUILD_TYPE=Release -DLLVM_USE_LINKER=lld
  -DPython3_EXECUTABLE=~/.venvs/atlas-mlir/bin/python`.
- **Test:** `ATLAS_LLVM_BIN=~/Projects/llvm-project/build/bin ctest --test-dir build`.
  The ELF-link tests need `ATLAS_LLD` pointing at an lld wrapper that adds
  `--image-base=0` (lld 22/23 and `-Ttext=0`).
- **Scope:** atlas-mlir does not unroll loops. Loop and tile structure belongs
  to whatever produces the virtual program. Loops atlas-mlir emits itself,
  such as the pack relayout, stay loops.
- **Don't fake proofs:** don't relax a checker without replacing what it
  proved. Every rule above exists because a generated-artifact check depends
  on it.
- **Commits:** one-line commit messages, with no co-author trailer.
- **Hand-off:** put the before/after cycles and utilization for the combined
  tile and the attention tile in the PR description.

## Caveats

- npu_model's timing (`AtlasTiming` ports it) has not been checked against
  the RTL.
- The combined program is a stand-in. Its interface and BF16→FP8 scale code may
  differ from the real one. The DMA busy time is the same for any program that
  moves these six tiles.

## Appendix: O1 prototype patch

This patch is for measurement only. It is unsound: it exempts every pack
helper from DMA conflict checks, and it carries pending transfers across blocks
in layout order instead of along the CFG. Apply it with `git apply` on
`106c9bb` to reproduce the 7,946 and 7,612 cycle results.

<details><summary>Prototype diff</summary>

```diff
diff --git a/lib/AtlasDMAContractVerification.cpp b/lib/AtlasDMAContractVerification.cpp
index d26458e..bd68981 100644
--- a/lib/AtlasDMAContractVerification.cpp
+++ b/lib/AtlasDMAContractVerification.cpp
@@ -263,7 +263,7 @@ LogicalResult mlir::atlas::verifyAtlasGeneratedDMAMemory(
           return transfer.launch.getChannel() == wait.getChannel();
         });
         if (found == pending.end())
-          return op->emitOpError("DMA memory wait has no captured transfer");
+          continue;  // PROTOTYPE: launched in an earlier block
         pending.erase(found);
       } else if (isa<VLoadOp, VStoreOp>(op)) {
         const Instr &in = s.instrs[i];
@@ -287,9 +287,7 @@ LogicalResult mlir::atlas::verifyAtlasGeneratedDMAMemory(
       // results conservatively become unknown; SELI leaves scalar regs alone.
       applyScalar(s.instrs[i], regs);
     }
-    if (!pending.empty())
-      return pending.front().launch.emitOpError(
-          "DMA memory transfer crosses a basic-block boundary");
+    (void)0;  // PROTOTYPE: transfers may cross the pack loop's blocks
   }
   return success();
 }
diff --git a/lib/AtlasGeneratedSchedule.cpp b/lib/AtlasGeneratedSchedule.cpp
index 816f884..d7601ab 100644
--- a/lib/AtlasGeneratedSchedule.cpp
+++ b/lib/AtlasGeneratedSchedule.cpp
@@ -18,10 +18,10 @@ bool mlir::atlas::canOverlapAtlasGeneratedDMA(Operation *op) {
   // ScalarCore captures DMA operands at launch; later scalar writes are safe.
   if (isa<DelayOp, MXUPushOp, MXUMatmulOp, MXUPopOp, VPUUnaryOp, VPUBinaryOp,
           VPUReduceOp, VLIOp, XLUTransposeOp, VLoadOp, VStoreOp, ALURegOp,
-          ALUImmOp, UpperOp>(op))
-    return true;
+          ALUImmOp, UpperOp, VPUPackOp, ScalarStoreOp, BranchOp>(op))
+    return true;  // PROTOTYPE: pack relayout beside pending DMA, no VMEM proof
   if (auto load = dyn_cast<ScalarLoadOp>(op))
-    return load.getKind() == "seli";
+    return load.getKind() == "seli" || load.getKind() == "lw";
   return false;
 }
 
diff --git a/lib/AtlasStream.cpp b/lib/AtlasStream.cpp
index cf9e3d1..81e968b 100644
--- a/lib/AtlasStream.cpp
+++ b/lib/AtlasStream.cpp
@@ -409,9 +409,10 @@ LogicalResult mlir::atlas::checkAtlasStream(
   auto name = [&](size_t i) {
     return s.ops[i]->getName().getStringRef().str();
   };
+  // PROTOTYPE: pending transfers carry across blocks in layout order.
+  std::array<std::vector<std::pair<size_t, Footprint>>, 8> pending;
   for (size_t b = 0; b < s.starts.size(); ++b) {
     RegValues regs = s.entry[b];
-    std::array<std::vector<std::pair<size_t, Footprint>>, 8> pending;
     for (size_t i = s.starts[b]; i < s.blockEnd(b); ++i) {
       const Instr &in = s.instrs[i];
       const OpInfo &op = *in.op;
@@ -428,7 +429,9 @@ LogicalResult mlir::atlas::checkAtlasStream(
         for (int ch = 0; ch < 8; ++ch)
           if (!pending[ch].empty())
             return s.ops[i]->emitOpError("completion publication requires DMA.WAIT for pending channel ") << ch;
-      for (int ch = 0; ch < 8; ++ch)
+      // PROTOTYPE: pack helpers touch only the pack scratch window.
+      bool packHelper = s.ops[i]->hasAttr("atlas.virtual_cfg_helper");
+      for (int ch = 0; ch < 8 && !packHelper; ++ch)
         for (const auto &[k, dma] : pending[ch]) {
           auto conflict = provider.dmaConflict(dma, f);
           if (!conflict.error.empty())
@@ -461,12 +464,6 @@ LogicalResult mlir::atlas::checkAtlasStream(
     }
     // Timing is checked one block at a time, so a transfer must complete in
     // the block that starts it, as verifyAtlasTimedStream requires.
-    for (int ch = 0; ch < 8; ++ch)
-      if (!pending[ch].empty())
-        return s.ops[pending[ch].back().first]->emitOpError(
-            "may still be in flight when its block ends; timing requires DMA "
-            "completion before a block boundary, so add atlas.dma_wait in "
-            "this block");
   }
   return success();
 }
@@ -656,9 +653,7 @@ LogicalResult mlir::atlas::verifyAtlasTimedStream(
         return s.ops[i]->emitOpError("timing provider issue gap exceeds positive cycle domain");
       cycle += gap.value;
     }
-    for (Operation *launch : pendingDMA)
-      if (launch)
-        return launch->emitOpError("timing verification requires DMA completion before a block boundary");
+    (void)pendingDMA;  // PROTOTYPE: transfers may cross the pack loop's blocks
     if (!s.endsInBranch(block) && !s.endsInHalt(block)) {
       int idle = 0;
       if (s.fallsThrough(block))
diff --git a/lib/AtlasVerificationContext.cpp b/lib/AtlasVerificationContext.cpp
index 7a2524c..0285e48 100644
--- a/lib/AtlasVerificationContext.cpp
+++ b/lib/AtlasVerificationContext.cpp
@@ -77,7 +77,8 @@ static LogicalResult verifyGeneratedStructure(ModuleOp module) {
       }
       if (target)
         for (const AtlasDMAInterval &interval : intervals)
-          if (*target > interval.launchPC && *target <= interval.waitPC)
+          if (*target > interval.launchPC && *target <= interval.waitPC &&
+              !(currentPC > interval.launchPC && currentPC < interval.waitPC))
             return op.emitOpError(
                 "generated redirect target enters a pending DMA interval");
     }
diff --git a/lib/AtlasVirtualScheduling.cpp b/lib/AtlasVirtualScheduling.cpp
index 28c7c66..247b7db 100644
--- a/lib/AtlasVirtualScheduling.cpp
+++ b/lib/AtlasVirtualScheduling.cpp
@@ -65,8 +65,7 @@ bool isCompletion(Operation *op) {
 // Boundary tiles and the pack move memory without a transfer handle, so they
 // run with no explicit transfer pending.
 bool isImplicitMemory(Operation *op) {
-  return isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op,
-             VirtualPackFP8Op>(op);
+  return isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op>(op);
 }
 
 // An explicit table: `Pure` is not trusted, because legacy virtual_mxu_matmul
diff --git a/lib/AtlasVirtualVerification.cpp b/lib/AtlasVirtualVerification.cpp
index 775e529..7ee263e 100644
--- a/lib/AtlasVirtualVerification.cpp
+++ b/lib/AtlasVirtualVerification.cpp
@@ -236,8 +236,7 @@ LogicalResult verifyVirtualBlock(Block &block, bool entry, bool cfg,
             "pending DMA handles cannot cross CFG blocks");
     }
     if (!dma.pending.empty() &&
-        isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op,
-            VirtualPackFP8Op>(operation))
+        isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op>(operation))
       return operation.emitOpError(
           "must complete pending DMA before implicit memory operations");
     if (auto start = dyn_cast<VirtualStartOp>(operation)) {
```

</details>
