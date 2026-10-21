# Code quality review

Reviewed `handwritten-implementation` at `22de8eb` (after PRs #29 and #30), on
branch `review/code-quality`. Scope: `lib/` and `include/` (15k lines of C++),
`tools/` and `python/` (12k lines of Python), `runtime/`, and the test and
build setup. Each finding below was checked against the code; line numbers
refer to `22de8eb`.

Severity:

- **High**: a correctness risk, or rules that have already diverged with
  measurable effect.
- **Medium**: duplication or design that will cost time on the next change.
- **Low**: readability, style, docs.

Several findings come from our own recent changes. They are marked
*(recent: ours)*, so they get the same scrutiny as the rest.

## Top issues

| # | Issue | Severity |
|---|---|---|
| 1 | The pass pipeline is defined four times, and the copies have diverged. The driver's copy takes 1.6x the cycles on a 16-tile program. | High |
| 2 | Weight reuse runs hidden inside three unrelated passes and probes the allocator by trial and revert. | High |
| 3 | Two virtual ops are declared `Pure` but have ordering obligations. | High |
| 4 | "Does this value take a register?" is decided in 11 places, and the exceptions are applied unevenly. | High |
| 5 | Boundary-ABI checks and slot-address math are written out in three transforms, with drift. | Medium |
| 6 | Hand-maintained mirrors of lowering: contract expansion tables, scheduler cost formulas, the staging-point rule. | Medium |
| 7 | Machine and virtual ops are built from name strings instead of typed builders. | Medium |
| 8 | Hardware constants are defined repeatedly under different names. | Medium |
| 9 | Contract verifiers are hard to read: 380-character lines, one diagnostic for 33 failures. | Medium |
| 10 | Test infrastructure: the whole Python suite is one ctest test, there's no CI, and a third of the suite skips. | Medium |

---

## 1. Duplicated rules and sources of truth

### 1.1 Four pipeline definitions, already diverged (High)

The order of passes from virtual IR to machine code is spelled out by hand in four places:

| Where | Boundary→DMA | MXU spill | Schedule | Spill | Notes |
|---|---|---|---|---|---|
| `python/atlas_mlir_native/binding.py:151-165` "mandatory" | no | no | no | yes | **the default** |
| `python/atlas_mlir_native/binding.py:156-163` "optimized" | yes | yes | yes | yes | only with `ATLAS_NATIVE_PIPELINE=optimized` |
| `tools/compile_atlas_linalg_tile.py:811-816` | yes | no | yes | yes | |
| `tools/lower_atlas_program.py:40-63` | **no** | opt-in | yes | opt-in (`--spill`) | stage-preserving driver |
| `tools/export_llvm_handoff.py:73` | no | no | no | no | |

The divergence has a measurable cost. Nicolas's `virtual_fp8_mlp_16_tiles.mlir`,
lowered with the driver's passes (no boundary conversion, delay insertion),
takes **58,102** cycles on npu_model main (63.6% DMA busy). The full pipeline
(`--atlas-compile`) takes **36,055** (99.3% DMA busy), with identical output. The default native pipeline schedules
nothing, and the faster one is behind an environment variable
(`binding.py:169-175`). The stage table in `docs/compiler-pipeline.md` is a
fifth description of the order.

**Fix:** register named pipelines once in C++ with `PassPipelineRegistration`
(for example `atlas-virtual-to-machine` and `atlas-virtual-to-machine-debug`)
and have every tool invoke them by name. The driver can still split them into
stages for artifacts, but should take its stage list from the same
registration. Make the optimized pipeline the default and drop the
environment variable, or turn it into an explicit option.

**Status on this branch: done.** `--atlas-compile` is registered
(`lib/AtlasPipeline.cpp`) as five registered stages, which the driver runs one
per artifact. The compile script, the binding's optimized pipeline and
`export_llvm_handoff.py` run it whole. The binding's unscheduled qualification
pipeline is registered too, as `--atlas-compile-unscheduled`, so no tool
spells out a pass list. The driver's `--spill` and `--timing` options are gone:
it always runs the full pipeline. Output is unchanged on all 21 compilable
examples, except for the driver and the exporter, which now produce the full
pipeline's result. Left for a team decision: which pipeline the binding uses
by default (`ATLAS_NATIVE_PIPELINE`).

### 1.2 "Does this value take a register?" decided in 11 places (High)

Seven files classify values as scalar or tile independently:

- `lib/AtlasCFGContractVerification.cpp:26,36`
- `lib/AtlasRegisterAllocationVerification.cpp:42`
- `lib/AtlasSpilling.cpp:293,388,566`
- `lib/AtlasVirtualAllocation.cpp:372,689`
- `lib/AtlasVirtualScheduling.cpp:506`
- `lib/AtlasVirtualToMachine.cpp:114,233`

There are three exceptions to apply: DMA immediates take no register,
memory-resident controls take no register, and loaded tiles take registers
only at staging. They are applied unevenly:

| Exception | Files that apply it |
|---|---|
| `isAtlasMemoryControl` | CFG contract, register checker, spill, allocator, lowering. **Not the scheduler**: its pressure model counts memory-resident entry controls as scalar registers. |
| `isVirtualDMAImmediate` | 6 files |
| `isVirtualDMALoadedTile` | 4 files *(recent: ours)* |

**Status on this branch: done.** `registerKindOf(Value)` in
`AtlasVirtualAllocation.h` now replaces all eleven sites, with identical output
on every example. The scheduler now leaves memory-resident controls out of its
pressure count, and the spill pass no longer considers DMA immediates as
victims.

**Fix:** one function, for example `std::optional<RegisterKind> registerKindOf(Value)`
in `AtlasVirtualAllocation.h`, that encodes every exception. All seven files
call it. Independent checkers can keep their own algorithms, but should share
this *specification* of which values exist.

### 1.3 Boundary ABI validation in three transforms (Medium)

The same rules are re-implemented in each: base at or above `0x80000000`,
1 KiB aligned, under 2³², 2 KiB slots, input and output windows disjoint.

- `lib/AtlasVirtualBoundaryToDMA.cpp:31-63` (`readBoundary`) *(recent: ours)*
- `lib/AtlasImmutableWeightReuse.cpp:51-76` (`boundaryBases`)
- `lib/AtlasVirtualToMachine.cpp:293-333` (`readABI`)

They have already drifted. Each starts from different slot counts:

| Copy | Input slots | Output slots |
|---|---|---|
| Boundary conversion (`:34`) | 0 | 1 |
| Weight reuse (`:51`) | 1 | 1 |
| Lowering (`:309`) | at least 1 | at least 1 |

Only the lowering also checks the VMEM window and PACK-scratch index limits. The slot address `base + index * 2048 + half * 1024` is computed
in at least four files (`AtlasRegisterAllocationVerification.cpp:474`,
`AtlasTileContractVerification.cpp:327`, `AtlasImmutableWeightReuse.cpp:96,101`,
`AtlasVirtualBoundaryToDMA.cpp:104,108`).

**Fix:** a `BoundaryABI` struct with `read(func::FuncOp)`, `inputSlotAddress(i, half)`
and `outputSlotAddress(i, half)`, used by the three transforms. The tile
contract verifier can keep its independent check.

**Status on this branch: done.** `AtlasBoundaryABI` (`AtlasBoundaryABI.h`)
reads and checks both bases, counts the slots, checks the DRAM spans
(`verifySpans`) and computes slot addresses. All three transforms use it, with
one slot rule: each window spans at least one slot, as lowering already
required. The lowering keeps its own VMEM-window and mailbox checks. The tile
contract and register checkers keep their independent address arithmetic.
Output is unchanged on every example; one diagnostic now reads "DRAM boundary
ABI requires explicit atlas.input_dram_base and atlas.output_dram_base".

### 1.4 The DMA staging-point rule in three places (Medium)

"A loaded tile's VLOADs go before its first use, before a later launch that
reuses its staging window, or at the block's terminator" is coded:

- in lowering, `lib/AtlasVirtualToMachine.cpp:91-99` and `582-619` (`flushLoad`, `flushReusedWindow`)
- in the register checker, `lib/AtlasRegisterAllocationVerification.cpp:326` (`collectStagingLoads`)
- in the allocator, `lib/AtlasVirtualAllocation.cpp:322` (`stageLoads`) *(recent: ours, `a6a4143`)*

The checker's copy is intentional: it is the independent check. The
allocator's and lowering's copies must agree by construction, though, and
nothing enforces that except the checker catching a mismatch.

**Fix:** lowering reads the staging points from `VirtualAllocationPlan` (add an
accessor) instead of re-deriving them.

### 1.5 Hand-maintained mirrors of what lowering emits (Medium)

Several tables duplicate the lowering's output by hand and must be edited
whenever lowering changes:

- **Tile command expansion counts** per source op, `lib/AtlasTileContractVerification.cpp:191-207`
  (`{6, 2}`, `{3, 1}`, …).
- **CFG record field counts** keyed by op-name strings, `lib/AtlasCFGContractVerification.cpp:304-305`.
- **Ops that "issue nothing"**, `lib/AtlasCFGContractVerification.cpp:32` *(recent: ours)*.
- **Scheduler costs** that restate lowering shapes: the PACK relayout loop
  formula (`lib/AtlasVirtualScheduling.cpp:478`), boundary tiles, staging,
  and spill costs. The spill-victim estimate mirrors the spill pass's
  longest-span heuristic *(recent: ours)*.

Commit `73f68af` ("refresh PACK counts") is an example of this upkeep.

**Fix:** where the contract is meant to be an independent specification, keep
it, but add one test that lowers every virtual op and compares the emitted
counts with the table, so drift fails loudly in one place. For the
scheduler's costs, derive estimates from a shared description of each op's
expansion rather than restating it.

### 1.6 Hardware constants defined repeatedly (Medium)

| Fact | Definitions |
|---|---|
| 8 DMA channels and windows | `AtlasDMAAllocationVerification.cpp:18`, `AtlasVirtualAllocation.cpp:27`, `AtlasVirtualScheduling.cpp:318` (`kMaxQueuedDMA`) |
| VMEM size | `AtlasTiming.h:73` (`kVmemBytes`), `AtlasDMAAllocationVerification.cpp:17` (`0x180000 / 4`) |
| 2 KiB tile, 1 KiB half | `AtlasVirtualToMachine.cpp:38-39`, `AtlasVirtualBoundaryToDMA.cpp:18`, `AtlasDMAAllocationVerification.cpp:19`, plus bare `2048`/`1024` literals across `lib/` |
| PACK reserves x10–x17, so scalars start at x18 | `AtlasVirtualAllocation.cpp:43,552` (literal `18`), `AtlasRegisterAllocationVerification.cpp:142,168` (literals `10`, `17`, `26`). None of these derive from `packSourceRegs` and the other fields of `selectedResources()`. |
| Register capacities 15 / 32 | literals in `AtlasVirtualAllocation.cpp:44-47` |

**Fix:** one `AtlasTarget.h` with the selected target's constants. Derive the
reserved scalar range from `FixedResourcePlacement`.

**Status on this branch: done.** `AtlasTarget.h` defines VMEM size, the DMA
channel count, tensor register count and size, and BF16 tile size. The
duplicate definitions are gone, and the bare tile-size literals in `lib/` use
it. `FixedResourcePlacement` now holds the scalar pool (`firstScalarValue`,
`lastScalarValue`) and derives the PACK span from its PACK registers, so the
allocator and the register checker read one definition. The capacity
diagnostics are built from the computed capacity. Output is unchanged on every
example.

### 1.7 Liveness implemented three times (Low)

The allocator (`AtlasVirtualAllocation.cpp:~400-439`), the register checker
(`AtlasRegisterAllocationVerification.cpp:~255-275`) and the CFG contract
builder (`AtlasCFGContractVerification.cpp:~210-217`) each run a hand-written
backward fixed point. The scheduler uses `mlir::Liveness`. The checker's copy
is justified as an independent check; the allocator and contract builder
could share one utility or use `mlir::Liveness` with a value filter (from
1.2).

### 1.8 Python duplication (Medium)

- `_sha` is defined **10 times** across `tools/`, for example `atlas_check_mailbox_dump.py:24`
  and `atlas_graph_model_bridge.py:13`.
- The BF16 round-to-nearest-even helper `_bf16_rne` is in both
  `python/atlas_mlir_native/exp_lane.py:29` and `graph_precision.py:103`.
  `_pow2` is in `exp_lane.py:25` and `reference.py:382`, and five more test
  modules. Numerics code should exist once.
- The nine `tools/atlas_*_host_data.py` and `*_bridge.py` scripts share about
  25–30% boilerplate pairwise (manifest reading, hashing, JSON I/O).
  **Fix:** a `tools/atlas_host_common.py`.

---

## 2. Design problems

### 2.1 Weight reuse is a hidden side effect inside three passes (High)

`reuseAtlasImmutableWeights(function)` is called from inside
`--lower-atlas-virtual-boundary-to-dma` (`AtlasVirtualBoundaryToDMA.cpp:97`),
`--schedule-atlas-virtual` (`AtlasVirtualScheduling.cpp:952`) and
`--lower-atlas-virtual-to-machine` (`AtlasVirtualToMachine.cpp:1097`). Its
own registered pass, `--reuse-atlas-immutable-weights`, is in no pipeline.
The problems:

- A scheduler and a lowering pass silently change program semantics,
  merging loads. Pass names no longer describe what passes do, and the result
  depends on which of the three runs first.
- It decides by **cloning the function, running the full allocator, and
  keeping the rewrite only if allocation succeeds** (`AtlasImmutableWeightReuse.cpp:172-186`).
  That's the same probe-and-revert pattern just removed from the scheduler,
  and it contradicts the spill-based design: reuse that needs one cheap spill
  is rejected outright. It can cost up to three extra full allocations per
  compile.

**Fix:** make it an explicit pipeline step before scheduling, and remove the
three hidden calls. Decide by estimated cost, or always reuse and let
spilling handle pressure.

### 2.2 Ops declared `Pure` that are not (High)

`virtual_mxu_matmul` (`include/Atlas/AtlasOps.td:138`) and `virtual_pack_fp8`
(`:220`) carry `[Pure]`. The scheduler's own comment says this can't be
trusted, because they have ordering obligations: fixed MXU slots and fixed
PACK scratch (`AtlasVirtualScheduling.cpp:72-73`). The scheduler works around
it with a private effect table, but any standard MLIR pass (CSE,
canonicalize, LICM, DCE) is entitled to reorder, merge or drop these ops.

**Fix:** model their effects truthfully, either memory effects on a
named resource or the state token the other ordered ops use. Then the
scheduler's table can be derived from the IR.

**Status on this branch: done.** Both ops now declare read and write memory
effects (`lib/AtlasOps.cpp`), as the boundary and machine ops do, with
identical output on every example. The scheduler keeps its finer table, which
lets DMA overlap PACK, and its comment now says why.

### 2.3 Legacy MXU matmul maintained in parallel (Medium)

The pre-handle `virtual_mxu_matmul` (fixed slots) is still special-cased in:

- MXU allocation verification (`AtlasMXUAllocationVerification.cpp:165-169`)
- MXU contract (`AtlasMXUContractVerification.cpp:345-361`)
- MXU ownership (`AtlasMXUOwnership.h:45-47`)
- CFG contract (`AtlasCFGContractVerification.cpp:447`)
- the scheduler's effect table

Every MXU change pays for both paths. **Fix:** decide whether it's still
needed. If not, rewrite it to handle-based ops in a canonicalization and
delete the legacy paths.

### 2.4 Ops built from strings instead of typed builders (Medium)

- **Lowering:** queues machine ops as `PlannedOp{name, attrs}`
  (`AtlasVirtualToMachine.cpp:41-46`) through 57 call sites of the form
  `add("atlas.scalar_load", loc, {{"kind", str("lw")}, {"dst", …}})`. `add()`
  attaches MXU and tile-command tags by comparing op-name strings
  (`:250-256`).
- **Other passes:** spilling (`AtlasSpilling.cpp:218,236,251,380`, including
  `"arith.constant"`), MXU spilling (`AtlasMXUSpilling.cpp:53,86,208,249`),
  wait insertion (`AtlasDMAWaitInsertion.cpp:269`) and boundary conversion
  (`AtlasVirtualBoundaryToDMA.cpp:73`, *recent: ours*) build ops through
  `OperationState` by name.

The ODS-generated builders already exist. With strings, a misspelled op or
attribute name compiles and only fails in a verifier at run time, if one
covers it. **Fix:** use the typed builders (`ScalarLoadOp::create(...)`,
`arith::ConstantIntOp::create(...)`). For lowering's deferred emission, keep
`PlannedOp` but give it typed construction callbacks.

**Status on this branch: done.** Lowering creates each machine op with its
typed builder as it goes (`emit<ScalarLoadOp>(loc, "lw", /*dst=*/...)`) and
sets branch offsets once labels resolve, so `PlannedOp` is gone, and its MXU
and tile-command tags come from the op type instead of name comparisons. The
spill, MXU spill, wait insertion and boundary passes use typed builders, with
shared `createAtlasSpillStore` and `createAtlasReload`. Only LLVM finalization
still builds ops by name, because it rebuilds arbitrary ops from markers.
Output is unchanged on every example.

### 2.5 Verification repeated on every pass boundary (Medium)

Every virtual pass runs the full `verifyAtlasVirtualModule` on entry and
again on exit: boundary conversion (`:94,110`), the scheduler (`:946,965`),
spilling (`:449,470`), MXU spilling (`:279,288`), weight reuse (`:162,166`)
and lowering (`:1086`). The pipelines also run explicit
`--verify-atlas-virtual-stream` passes. That's about ten full verifications
per optimized compile. It is defensive, but costs compile time and blurs which
pass owns which guarantee.

**Fix:** verify at pipeline boundaries, plus an opt-in "verify after every
pass" mode, as MLIR's own `-verify-each` does. Keep explicit entry checks
only where a pass relies on a property its predecessor doesn't establish.

### 2.6 Spill pass limits (Medium)

- Only accepts a module with one function (`AtlasSpilling.cpp:454`), so every
  pipeline inherits that restriction.
- The victim is the **longest-lived value anywhere in the function**
  (`:556-575`), not a value live at the point that failed. That can spill
  values that don't relieve the conflict, with one reload per use. We
  measured 2 spills turning into 15–17 reloads.
- Each iteration reruns the full allocator and rescans every op and every
  use (`:481` onward). The cost is O(spills × program size), with
  `max-spills` defaulting to 256.
- MXU spilling happens in two passes: `--spill-atlas-mxu-state`, and again
  inside `--spill-atlas-virtual` (`:465`).

**Fix:** have the allocator report the failing interference clique and pick
the victim from it, preferring the value whose next use is furthest away.
Then spill incrementally.

### 2.7 Operand-index arithmetic instead of named accessors (Medium)

- `isVirtualDMAImmediate` checks `use.getOperandNumber() + 2 >= user->getNumOperands()`
  (`AtlasVirtualVerification.cpp:504`) *(recent: ours)*.
- `onlyWeightUses` checks `use.getOperandNumber() == 1` (`AtlasImmutableWeightReuse.cpp:32-37`).
- `dmaSpan` picks operands by `store ? 2 : 1` (`:41-42`).

Each silently breaks if an op's operand order changes. **Fix:** compare
against `getDramByte()`, `getSizeBytes()` and `getWeight()`.

**Status on this branch: done.** Typed accessors replace operand positions:
`virtualDMATransfer` (a virtual DMA's address, size and tile),
`virtualMXUReadoutAcc`, `atlasAccessedSlot`, `atlasSlotValue`,
`atlasSpillSaved` and `atlasSpillTile`. The three sites above, and the same
pattern in the MXU contract, allocation, scheduler, spilling, tile contract
and lowering, use them or the ops' named accessors. Only the state chain,
every stateful op's first operand by convention, is still read by position.
Output is unchanged on every example.

### 2.8 Fixed registers aliased without explanation (Low)

`selectedResources()` (`AtlasVirtualAllocation.cpp:50-82`) gives x1 to both
`haltReg` and `inputDramReg`, and x3 to both `outputDramReg` and `scaleReg`.
The PACK reserves x10–x17, but x15 is unused: the checker notes "x15 is
intentionally reserved" (`AtlasRegisterAllocationVerification.cpp:141`) without
saying why. The register checker's content model proves the sharing is safe,
but the table should say *why* the roles can share a register.

### 2.9 Behavior chosen by environment variables (Low)

Pipeline choice (`ATLAS_NATIVE_PIPELINE`) and tool locations
(`ATLAS_OOT_BIN_DIR`, `binding.py:3447`) come from the environment, with
defaults that pick the slower path. Prefer explicit arguments with logged
values.

---

## 3. Readability

### 3.1 Contract verifiers are hard to read (Medium)

Lines over 120 characters: `AtlasCFGContractVerification.cpp` 71 (longest
380), `AtlasBufferContractVerification.cpp` 22, and
`AtlasSourceMemoryEffectContract.cpp` 22. Examples:
`AtlasCFGContractVerification.cpp:164,196,308,330`. Line 308 is a
16-condition `if` on one line, and line 196 builds a seven-field record in
one expression. Several lines also hold more than one statement.

**Fix:** run `clang-format` (see 4.3) and split record building and parsing
into small named helpers.

### 3.2 One diagnostic for 33 failure conditions (Medium)

`AtlasCFGContractVerification.cpp:245` defines
`bad() = "CFG contract malformed source records"`, returned from 33
different checks. A failure doesn't say which field, record or condition
broke. **Fix:** say which condition failed, for example "operation record 12
is missing tile_commands".

### 3.3 Very long functions (Medium)

| Lines | Function |
|---|---|
| 371 | `AtlasCFGContractVerification.cpp:240` (contract verification) |
| 270 | `AtlasTiming.cpp:403` (footprint construction) |
| 234 | `AtlasScheduling.cpp:23` `scheduleBlock` (ported from atlas-compiler-experiments) |
| 232 | `AtlasVirtualAllocation.cpp:362` `colorValues` |
| 225 | `AtlasTileContractVerification.cpp:223` `buildAtlasTileContract` |
| 197 | `AtlasVirtualVerification.cpp:180` (virtual function verification) |
| 178 / 175 | `AtlasBufferContractVerification.cpp:42` `derive`, `:288` verify |
| 177 | `AtlasDelayInsertion.cpp:23` `timeBlock` |
| ~280 | `AtlasVirtualToMachine.cpp:700` `lowerOperation`, one `dyn_cast` chain |

In Python, `binding.py` is a single 3,887-line module: `compile` is 804
lines (`:3084`), `profile` 353 (`:2729`), `prepare_candidate` 303 (`:3274`).

**Fix:** split by phase, for example parse, check and build for the contract
verifiers, and request matching, selection, emission and verification for
`binding.py`, with one function per op family in `lowerOperation`.

### 3.4 Op identity compared as strings (Low)

There are 12 comparisons such as `name == "atlas.virtual_save"`:
`AtlasCFGContractVerification.cpp:304,447,454,459`,
`AtlasTileContractVerification.cpp:94,96`,
`AtlasBufferContractVerification.cpp:82-83`, and
`AtlasVirtualToMachine.cpp:250,253,1027`. Use `isa<>` on ops, or compare
against `Op::getOperationName()` where only a name is available, as the
tile expansion table already does.

### 3.5 Per-op facts spread over parallel switches (Low)

Adding a machine op means editing:

- the encoding switch (`AtlasEncoding.cpp:66`)
- the op-to-timing-mnemonic switch (`AtlasStream.cpp:22`)
- the string-keyed timing table (`AtlasTiming.cpp:40-73`)
- the dialect, and sometimes `atlas-emit`

Consider TableGen per-op metadata (encoding format, timing class) so one
definition drives all of them.

---

## 4. Tests and infrastructure

### 4.1 The whole Python suite is one ctest test (Medium)

`CMakeLists.txt:25-28` registers `unittest discover` as a single test,
`atlas-dialect`. Any one failure marks all 762 tests as failed, and there's
no per-module timing. `PYTHONPATH` doesn't include `python/`, so five tests
fail to import on a clean checkout (`atlas_mlir_native`). Apart from those
five, the suite passes at `22de8eb`. **Fix:** add
`python/` to the environment, and register one ctest entry per test module.

**Status on this branch: done.** Each test module is its own ctest test
(`atlas-python-<module>`), run from `test/` with `python/` prepended to
`PYTHONPATH`, which fixes the five import errors. The three modules written as
plain pytest functions run under pytest; under `unittest discover` they had
never run. Running modules separately exposed one hidden dependency:
`test_virtual_boundary_to_dma` relied on another module having put `tools/` on
the path. The suite runs in parallel now, in about 9 seconds instead of 36.
`ENVIRONMENT_MODIFICATION` needs CMake 3.22, so that is the new minimum.

### 4.2 A third of the suite skips by default, with no CI (Medium)

231 of 762 Python tests skip, even with `ATLAS_LLVM_BIN` set as ctest
expects:

| Tests | Skip reason |
|---|---|
| ~123 | a Merlin checkout |
| ~58 | ARC, ModeLIR or RTL paths |
| ~44 | an independent assembler root |
| ~6 | other (npu_model pin, handoff archive, clang) |

There's no record of which configuration last ran them, so the RTL and Merlin
paths can rot unnoticed.

### 4.3 No CI, formatting or lint configuration (Medium)

There's no `.github/workflows`, `.clang-format`, `.clang-tidy`, or ruff, flake8
or mypy configuration. **Fix:** a CI job for build, ctest and the skip counts,
and adopt `clang-format` (LLVM style) plus `ruff`. Format in one dedicated
commit so blame stays readable.

### 4.4 Test helpers live in test modules (Medium)

- **Cross-imports:** 73 of 119 test modules import helpers from other test
  modules. `test_mxu_reference` is imported by 27 of them, and
  `test_virtual_lowering` by 20.
- **Copy-pasted helpers:** `_emitted` is defined in 17 modules, 7 of them
  exact copies. A subprocess wrapper named `run` is defined in 5.
- **Shared support exists but is thin:** `verification_support.py` is the
  only shared module.

**Fix:** a `test/support/` package for program builders, tool runners and
decoders.

### 4.5 Tests that edit generated code as text (Low)

Many verification tests locate and rewrite lowered machine IR by regex and
line index, for example `test_dma_contract_verification.py`,
`test_virtual_dma_lowering.py` and `test_source_memory_contract.py`. They
encode incidental codegen details such as which register holds what, or
which instruction comes first. Three recent compiler changes (DMA immediates,
staged tiles, the scheduler) each forced fixture edits unrelated to the
behavior under test. Where possible, assert on structured output (the
allocation report, `--program-json`) or mutate through a small typed helper
rather than raw regexes.

### 4.6 Dead fixture (Low)

`test/examples/handoff_attention_mlp_boot_layout.json` isn't referenced by
any test, tool, doc or glob.

**Status on this branch: done.** Deleted.

---

## 5. Documentation

- **Branch logs filed as reference docs (Low).**
  `docs/pack-dma-overlap.md:3` and `docs/pack-relayout-loop.md:3` open with
  "Branch: … Status: implemented" and contain dated result tables. Keep
  `docs/` as reference for current behavior, and move measurements and history
  into PR descriptions or a changelog.
- **Pipeline order described in several places (Low).** See 1.1. The docs
  should point at the single registered pipeline once it exists.

---

## 6. What is in good shape

- Fixed resources (registers, VMEM windows, channels) are centralized in
  `selectedResources()`, apart from the gaps in 1.6.
- The register, DMA and MXU allocation checkers and the generated-artifact
  contracts recompute facts independently instead of trusting the allocator.
  That is a strong safety net, and it caught every mistake made in recent
  work.
- The C++ consistently uses `FailureOr`, `LogicalResult` and op-anchored
  diagnostics; there are no exceptions or silent `assert`-only checks on input.
- The DMA timing model has one implementation, `timing::dmaTransferCycles`,
  shared by the schedulers and machine timing.
- The runtime is small with little duplication: the two most similar files
  share 34 lines.
- The Python tools have no bare `except` clauses and no regex rewriting of
  MLIR.

## 7. Suggested order of work

1. **One registered pipeline** (1.1). It is cheap, and it fixes a measured 1.6x
   slowdown in the driver along with the native default.
2. **Shared definitions:** `registerKindOf` (1.2), `AtlasTarget.h` (1.6) and
   `BoundaryABI` (1.3).
3. **Weight reuse** becomes an explicit pass with no trial allocation (2.1).
4. **Truthful effects** for `virtual_mxu_matmul` and `virtual_pack_fp8` (2.2),
   and a decision on the legacy matmul (2.3).
5. **Typed op construction** in lowering and the spill passes (2.4), and
   named accessors (2.7).
6. **Contract verifier readability:** `clang-format`, split functions and
   specific diagnostics (3.1–3.3), with a test pinning the expansion
   tables to what lowering emits (1.5).
7. **Test infrastructure:** `PYTHONPATH`, per-module ctest entries, CI, and a
   shared `test/support/` package (4.1–4.4).
8. **Python cleanup:** split `binding.py` and share the host-tool helpers (1.8).
