# Verification of RTL timing facts

These checks connect extracted facts, compiler admission and hardware simulation for selected operations/configurations. See the [provider contract](README.md) for assumptions and limits, and the [extractor reference](../../tools/rtl_extract/README.md) for commands.

## Extraction and compiler cross-check

[test_rtl_extract_synthetic.py](../../test/test_rtl_extract_synthetic.py) tests interpretation, coupled recipes, responders and spec validation. Unsupported control dependencies must remain unresolved.

[test_rtl_extract_hw.py](../../test/test_rtl_extract_hw.py) checks retained HW IR, engine/DMA timings and one-cycle responses. With `ATLAS_HW_IR` and `ATLAS_HW_EXPORTER`, expectations must match extraction. Stage insertion/bypass and payload-dependent control must change event ages or leave records unresolved.

Its `CompilerCrossCheck.test_compiler_rules_and_fact_coverage` uses the `rtl-timing-facts-probe` target from `ATLAS_OOT_BIN_DIR` (`ATLAS_TIMING_PROBE` overrides it). The [facts map](../../test/rtl-timing-facts-map.yaml) requires every nonempty event group/storage access compared or explicitly unmodeled; perturbed facts must report differences. This checks built-in constants; loaded-fact consumers are tested below.

[test_rtl_extract_pairs.py](../../test/test_rtl_extract_pairs.py) checks stable gaps and rejection immediately below, ignored launches, event changes, busy guards, interchangeable ports and per-command sequence attribution. Pair-fact tests reject unchecked records, mismatched hardware, unresolved blocks, duplicates and inconsistent windows. Missing or spec-inconsistent operand metadata fails; generated-spec canonical digests are pinned in `sources`.

Selected evidence contains 841 VPU pairs (402 at gap 1), 196 disjoint-bank and 536 shared-operand same-MXU pairs: 1,573 total. Sweeps check 177 AtlasCore assertions. Eighteen VPU chains check 11 VectorEngineTop assertions; 172 MXU chains check 177 AtlasCore assertions. Counts come from sweep `invariants`; chains test spacing, with MXU chains outside selected capacity evidence.

[test_rtl_extract_regen_check.py](../../test/test_rtl_extract_regen_check.py) checks cache invalidation, output inventories, generated-spec dependencies, stage selection, environment emission, replay counts and comparison. Every gap, record field and metadata item is substantive; SHA pins must match before narrowly normalizing paths.

## Compiler admission

[test_rtl_timing.py](../../test/test_rtl_timing.py) tests selection/digests, unsupported domains, missing/unresolved facts, memory boundaries and drainage under insertion and scheduling. Emission and both LLVM paths recheck streams after timing-marker removal or mutation. Concrete-entry tests cover known/unknown branches, strided loops, worst-context joins, invariant tiles beyond 4096 entries and rejection of moving tiles. `PairFactsSelectionTest` checks pair identity, export/reload, gaps, capacities and unknown blocks. [timing-provider-verification.cpp](../../test/timing-provider-verification.cpp) checks registry/common-verifier isolation and physical scheduling with a non-model provider.

[test_rtl_engines.py](../../test/test_rtl_engines.py) checks facts-derived admission boundaries and one cycle below: engine overlap, VSTORE retention, changed facts/operands, pending/queued DMA, matching waits and final boundaries. Unselected MXU timing retains its default-provider regression.

[test_rtl_pairs.py](../../test/test_rtl_pairs.py) checks both consumers, every same-MXU pair's access rules/sweep bound, capacity two, VPU slot reuse and XLU exclusion against every live VPU. Missing/unusable facts require isolated holds; inapplicable facts preserve schedules. Every VPU chain matches minimum uniform spacing. MXU chains reject below stable gaps; at least 109 match exactly, with capacity/access diagnostics explaining later admission.

[test_rtl_scheduler.py](../../test/test_rtl_scheduler.py) tests candidate orders, DMA hints, default ties, joins and loops. [test_rtl_export.py](../../test/test_rtl_export.py) checks words, digests, applicability, block origins and virtual lowering/reordering, costs, queued DMA, shared-bank rejection and FP8 relayout contexts.

### Gap coverage

[test_rtl_gap_coverage.py](../../test/test_rtl_gap_coverage.py) builds programs from recorded overlays, including empty ones, and finds minimum verified gaps. System words must re-encode identically; unmapped records fail and unsupported pairs require provider rejection. Gaps must meet [the sweep bound](README.md#gap-coverage). DMA extraction tests also compare the other operation's events directly because sweep diagnostics report only the first mismatching group.

`ATLAS_PAIR_SWEEPS` supplies platform-path-separated paths; defaults include `pairs-*.json` beside `ATLAS_OP_TIMING`, so add system-directory sweeps explicitly. `ATLAS_PAIR_FACTS` enables pair admission. Reports include bounds, compiler gaps/slack and informational default-provider comparisons; rejection counts depend on selected sweeps/operands.

## Verilator replay

[replay-ee290-compute.py](../../tools/replay-ee290-compute.py) schedules, verifies, emits and runs cases on AtlasCore using supplied RTL or a prebuilt model with the current [driver](../../test/ee290-compute-replay.cpp). [Reproduce](README.md#reproduce) regenerates it. [ee290_replay_cases](../../tools/ee290_replay_cases/__init__.py) registers instruction/virtual programs, exact initial/expected memory, command counts and overlap requirements. Pair cases require `--pairs`; declared consumers run separately.

Replay follows branches/delay slots, requires drained entries and attributes events to individual VPU/MXU commands. Binding checks provider/digests, words, dynamic block offsets, resources, addresses, ages, response data and lifetimes. VPU own-unit hold end plus one equals the first post-issue edge clearing its opcode's `issueBusy`; with live companions it may be earlier because they can delay clearance. MXU own-unit holds match release; policy holds cover constrained activity. DMA completion binds to matching waits. Scalar/scale accesses use concrete-entry footprints.

Exact references cover transpose, BF16 multiply, supported VPU operations, FP8 conversion and MXU tiles/accumulations. Control/virtual cases cover loops, both diamond arms, concurrent DMA and two-layer FP8 MLPs with per-iteration relayout. Required overlap must occur. Memory comparison includes inputs and guards; unsupported exact VPU references fail.

[test_ee290_compute_replay.py](../../test/test_ee290_compute_replay.py) rejects mutated words/digests/contexts, moved/misattributed accesses, incomplete responses, borrowed releases, missing drains and corrupt accumulator/pop events. `VerilatorReplayTest.test_vpu_issue_busy_clear_must_match_its_own_unit_hold` rejects early/late clearance. Portable tests cover decoding, exact references and VCD sampling. Hardware requires `ATLAS_EE290_COMPUTE_MODEL`, `ATLAS_OOT_BIN_DIR`, `ATLAS_OP_TIMING` and optionally `ATLAS_PAIR_FACTS`.

DMA completion remains external. Sweeps/replay cover queued launch order, distinct channels and compute beside transfers; disjoint windows, channel reuse and eight-slot admission remain compiler obligations.

## Gates

With selected artifacts, run these modules; skips provide no execution evidence. Gap coverage can be expensive.

```sh
source build/env.sh
unset ATLAS_TIMING_PROBE
export PYTHONPATH="$PWD/tools:$PWD/test:$PYTHONPATH"
"$VPY" -m unittest test_rtl_timing test_rtl_engines test_rtl_pairs \
  test_rtl_scheduler test_rtl_export test_rtl_gap_coverage \
  test_selected_core_runtime test_ee290_compute_replay
"$VPY" -m unittest test_rtl_extract_hw test_rtl_extract_pairs \
  test_rtl_extract_synthetic test_rtl_extract_regen_check
```
