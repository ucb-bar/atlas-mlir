# Verification of RTL timing facts

The checks below connect extracted facts, compiler admission and simulated hardware. They qualify only the extracted operations and configurations. See [the provider contract](README.md) for assumptions and limits, and the [extractor reference](../../tools/rtl_extract/README.md) for extraction commands.

## Extraction and compiler cross-check

[test_rtl_extract_synthetic.py](../../test/test_rtl_extract_synthetic.py) checks the control interpreter, coupled recipes, responder timing and specification validation on small circuits. Unsupported control dependencies must remain unresolved rather than produce a timing fact.

[test_rtl_extract_hw.py](../../test/test_rtl_extract_hw.py) checks retained HW IR, engine timings, DMA and the retained-IR tool. With `ATLAS_HW_IR` and `ATLAS_HW_EXPORTER`, engine tests compare extracted values with specification expectations. Mutation tests insert register stages, bypass a valid stage or make control depend on payload: event ages must change or the record must become unresolved. Response tests check the one-cycle memory response assumption.

Its `CompilerCrossCheck` compares facts against compiler footprints through the build's `rtl-timing-facts-probe` CMake target. It finds the probe through `ATLAS_OOT_BIN_DIR`; `ATLAS_TIMING_PROBE` overrides that path. The [facts map](../../test/rtl-timing-facts-map.yaml) states comparisons and explicit exceptions. Tests require every nonempty event group and storage access to be compared or declared unmodeled, and perturb facts to check that differences are reported. This checks built-in footprint constants; selection and scheduling against loaded facts have separate tests below.

[test_rtl_extract_pairs.py](../../test/test_rtl_extract_pairs.py) covers pair sweeps, sequences, pair-fact construction and pins of all generated specs. Pair window tests reproduce stable gaps and the rejection immediately below them; synthetic tests distinguish ignored launches, changed event streams, busy guards and interchangeable ports. Sequence tests apply guards to every command after the first and check event attribution across a chain.

The VPU evidence comprises 841 ordered pairs, including 402 with stable gap 1, and 18 chain records. The pairs use AtlasCore with 177 checked assertions; the chains use VectorEngineTop with 11. Same-MXU evidence comprises 196 disjoint-bank and 536 shared-operand pairs, plus 172 chain records, checked through AtlasCore with 177 assertions. These assertion counts come from the sweep records' `invariants` fields. Pair-fact tests check a VPU live capacity of two, the MXU live-window convention, accumulator and weight-slot relations, and rejection of unchecked records, mismatched hardware, unresolved blocks, duplicate records and inconsistent windows. The selected pair-fact document contains 1,573 pair records.

## Compiler admission

[test_rtl_timing.py](../../test/test_rtl_timing.py) covers provider selection, digest checks, unsupported domains, missing or unresolved facts, memory boundaries, terminals and block drains. Both delay insertion and scheduling are checked with `--verify-atlas-timing`. Emission and both LLVM conversion paths recheck the actual stream, so stripping a timing marker or rewriting a scheduled stream cannot bypass verification.

The same module checks blocks under each concrete entry context: known branches select their arm; unknown branches retain both arms; strided loops check each iteration; joins take the worst context's required spacing. Tests accept an invariant tile beyond the 4096-entry bound and reject a moving one with the bound named. Its `PairFactsSelectionTest` checks pair-fact digests, hardware and operation-fact identity, export and reload, malformed gaps and capacities, and unknown blocks. [timing-provider-verification.cpp](../../test/timing-provider-verification.cpp) checks the registry and common verifier's provider isolation, including use of a non-model provider by the physical scheduler.

[test_rtl_engines.py](../../test/test_rtl_engines.py) checks engine admission and register-access boundaries at the facts-derived gap and one cycle below it, including VSTORE source retention. It checks cross-engine overlap, altered facts and operands, pending DMA, queued transfers, matching waits and final-boundary verification. The unselected MXU check retains a regression for the default provider's rules.

[test_rtl_pairs.py](../../test/test_rtl_pairs.py) checks pair-fact admission under both consumers. It checks every same-MXU pair against the compiler's access rules and sweep bound, capacity two, VPU slot reuse, and XLU exclusion against every live VPU command. Missing or unusable facts fall back to isolated holds; facts with no applicable records preserve schedules. Every VPU chain must have the sweep's smallest accepted uniform spacing. Every MXU chain rejects all spacings below its stable gap; at least 109 match exactly, and a capacity or access-rule diagnostic must explain a later compiler admission.

[test_rtl_scheduler.py](../../test/test_rtl_scheduler.py) checks RTL candidate ordering and DMA hints, unchanged default-provider ties, join contexts and loop footprints. [test_rtl_export.py](../../test/test_rtl_export.py) checks exported words, digests, applicability and block-relative cycle origins. Its virtual program tests check selection across lowering and reordering, facts-derived costs, queued transfers, shared-bank rejection and the FP8 relayout loop's entry contexts.

### Gap coverage

[test_rtl_gap_coverage.py](../../test/test_rtl_gap_coverage.py) reverses the pair sweeps: it builds each compiler pair with the specification's operand overlays and finds the smallest selected-provider gap accepted by `--verify-atlas-timing`. System instructions must re-encode to the sweep's words. Unmapped records fail the test; unsupported pairs must produce provider rejections. The accepted gap must meet [the sweep bound](README.md#gap-coverage), which treats DMA and DRAM event movement separately from binding failures. The DMA tests in [test_rtl_extract_hw.py](../../test/test_rtl_extract_hw.py) also compare the other operation's events directly: a sweep record reports only the earliest mismatching event group.

`ATLAS_PAIR_SWEEPS` supplies sweep paths separated by the platform path separator. Its default is every `pairs-*.json` beside `ATLAS_OP_TIMING`; include the system-directory sweeps explicitly to cover them. `ATLAS_PAIR_FACTS` enables pair-fact admission for the same check. The report lists each required bound, accepted compiler gap and slack, plus an informational comparison with the default provider. Rejection counts depend on the selected sweeps and operand mapping; they are not a provider guarantee.

## Verilator replay

[replay-ee290-compute.py](../../tools/replay-ee290-compute.py) schedules cases with selected facts, verifies, emits and runs them on AtlasCore. It accepts a prebuilt model or builds from supplied RTL; the model must use the current [ee290-compute-replay.cpp](../../test/ee290-compute-replay.cpp). Replay commands are in [Reproduce](README.md#reproduce).

Cases live in [ee290_replay_cases](../../tools/ee290_replay_cases/__init__.py), [mxu.py](../../tools/ee290_replay_cases/mxu.py) and [vpu_ops.py](../../tools/ee290_replay_cases/vpu_ops.py). A case supplies instructions or a virtual program, initial and expected memory, command counts and optional overlap checks; new groups register explicitly in `load()`. Pair-fact cases require `--pairs`. Delay insertion and scheduling run each case under its declared consumers; final streams bind footprints to a serialized reference.

The validator follows branches with their delay slot, requires drained block entries and attributes events to individual live commands, including two VPU slots and pipelined MXU commands. The binder checks provider and evidence digests, emitted words, dynamic block offsets, access resources, addresses and ages, response data and lifetimes. Each VPU command's own-unit hold end plus one must equal the age of the first post-issue edge where its opcode's `issueBusy` bit clears, or be no greater than that age with live overlapping VPU commands, since a companion may delay the clear. MXU own-unit holds must end at the observed release; policy holds must cover the activity they constrain. DMA completions bind to their matching waits rather than a latency estimate. Scalar loads, stores and scale loads bind to footprints for their concrete entry context.

Exact-data cases cover transpose, BF16 multiply, supported VPU lane operations, FP8 pack/unpack and MXU tiles and accumulations. Control-flow and virtual cases cover loops, both diamond arms, concurrent DMA and FP8 two-layer MLPs, including each iteration's relayout addresses. Overlap cases require the intended engines or same-unit commands to be live together. Memory comparison includes initialized input and guard bytes as well as outputs; the driver rejects VPU operations without an exact supported reference for the supplied data.

[test_ee290_compute_replay.py](../../test/test_ee290_compute_replay.py) checks these bindings and mutates traces and exports: wrong words, digests or block contexts, moved or misattributed accesses, incomplete responses, borrowed releases, missing drains and corrupt accumulator or pop events must fail. Its portable tests check decoding, exact references and VCD sampling without a hardware model. Hardware tests require `ATLAS_EE290_COMPUTE_MODEL`, `ATLAS_OOT_BIN_DIR` and `ATLAS_OP_TIMING`; `ATLAS_PAIR_FACTS` enables overlap cases that require pair facts.

DMA completion remains external. Assertion-checked pair sweeps and replay cover queued launch order, distinct channels and compute beside pending transfers; disjoint address windows, channel reuse and the eight-slot limit are compiler obligations. See [Not covered](README.md#not-covered) for pair record metadata limitations.

## Gates

With the prepared environment from `build/env.sh`, run the compiler and extractor modules separately. Hardware-dependent tests require the selected artifacts; skips are not execution evidence. The gap sweep can be expensive.

```sh
source build/env.sh
unset ATLAS_TIMING_PROBE
export PYTHONPATH="$PWD/tools:$PWD/test:$PYTHONPATH"
"$VPY" -m unittest test_rtl_timing test_rtl_engines test_rtl_pairs \
  test_rtl_scheduler test_rtl_export test_rtl_gap_coverage \
  test_selected_core_runtime test_ee290_compute_replay
"$VPY" -m unittest test_rtl_extract_hw test_rtl_extract_pairs test_rtl_extract_synthetic
```
