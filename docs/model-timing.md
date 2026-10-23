# Model timing

atlas-mlir has two timing providers. `builtin`, the default, uses Atlas's handwritten timing rules. The opt-in `model` provider uses isolated operation timings, reservation releases and curve/DMA scalar costs derived from the [NPU Model](https://github.com/ucb-ee194-tapeout/npu_model), then selects schedules by executing verified candidates on that model. Overlap admissions use a supplied simulation-measured catalog plus handwritten structural rules. Architecture and operand vocabulary are derived from the selected model and checked against compiler capabilities; frontend semantics and conservative compiler policies remain handwritten; this is not a fully source-derived timing environment.

## Derive and validate facts

Derivation is an explicit step. Run it in the selected NPU Model's Python environment, using a new output directory:

```sh
python -B tools/atlas_model_facts.py --npu-model /path/to/npu_model --out /path/to/new-facts
```

`tools/atlas_model_facts.py` derives issue/completion windows and reservation releases from model source and configuration using `atlas_derive_mxu.py`, `atlas_derive_vpu.py`, `atlas_derive_memory.py` `atlas_derive_curve.py`, `atlas_derive_arch.py` and `atlas_derive_vocabulary.py`. Reviewed source structures are recorded in `tools/model-timing.lock.json`; unsupported source structures or derivation capabilities stop derivation. The reviewed AST contracts use Python 3.14. Derivation reads the selected model's `DefaultHardwareConfig`; successful derivation does not imply compiler support for a changed configuration. Compilation accepts only the reviewed default configuration: 32×32 geometry and 64 MREGs; register/weight/CSR and VMEM/DRAM sizes and bases; fetch width; DMA channels, beat/transfer/in-flight limits; VMEM bank/bus and off-chip link geometry and pacing; RTL numerics, deterministic initialization and fatal runtime errors; the exact execution-unit classes and `IsaSpec`; and the curve backend with only `curves=ee290sim_vcs_probe.json` and null latency/beat overrides. Compilation and overlap evidence require the lock's configuration identity.

The operation probes come from `IsaSpec`. Source-derived facts are checked against independent isolated executions with zero/nonzero inputs and instrumentation on/off. Control traces, progress, drain, event ages and releases must agree, and runtime errors remain fatal. The driver also verifies that model source, numerical data and derivation/observer tools stay unchanged during the run. Only successful validation produces the final `model-timing.json` and `facts.lock.json`; intermediate diagnostic files can remain after a failed validation.

Curve facts include unloaded latency, response pacing for every read/write mix, queue capacity, issue-history window, channel/frequency scaling, DMA startup, request interval and rounding/reset semantics. The supported controller uses unloaded latency with converge=1 and peak pacing. `atlas_validate_curve.py` compares an independent fact interpreter with native backend streams and isolated/mixed DMA runs, including backpressure, translation and scaling. These checks validate proposal costs; they do not establish completion of an arbitrary compiled stream.

Pass `model-timing.json` and its `op_timing_sha256` from `facts.lock.json` explicitly when compiling. Compilation does not derive facts or manage a facts cache. Missing coverage, unsupported operands, stale identities and malformed evidence are rejected. The compiler checks the selected facts digest and schema; validation-report hashes are retained inside the facts, and external report bytes are not separately authenticated by the compiler.

## Architecture and vocabulary provenance

`architecture` is generated from the effective `ArchStateConfig` and default hardware configuration: VMEM capacity, bank bytes/count, MREG count/rows/row bytes/register bytes, weight-buffer and scalar/exponent/CSR capacities, DMA channels/beat/transfer/in-flight limits, DRAM address range, fetch width and bus/link geometry. `bank_conflict.py:BankConflictChecker.access_mreg` supplies the physical bank modulus and row mapping; its reservation guard supplies the logical register limit. BF16 pair count/stride and tile bytes come from the bank-conflict pair helper and `ArchState.read_mrf_bf16_tile`/`write_mrf_bf16_tile`; scalar word width comes from `ArchState.write_xrf`. Extraction checks the allocation and VMEM banking implementation against those configuration fields.

The model provider requires these facts and checks them against `AtlasTarget.h`/`AtlasTiming.h` capabilities and the explicit configuration. This also guards fixed layout assumptions in the separately maintained overlap consumers (32 MREG banks, 32 rows, 1024-byte registers, six 262144-byte VMEM banks). Those consumers are not generalized to arbitrary geometry by this change. Builtin keeps the same constants and behavior. DMA command-ring depth is intentionally absent from `architecture`: `kAtlasDMACommandSlots` is an explicit RTL contract that the model cannot derive.

`operand_vocabulary` is generated from `configs/isa_definition.py` operand types/roles and the VPU/MXU unit planners. The derived metadata determines read/write direction, MREG register counts, offsets and row-pass counts, and MXU accumulator/weight/exponent resources. `isa_metadata` records model operand types, named roles, accesses, execution units and compiler syntax tokens. The compiler compares generated vocabulary with its compiled snapshot and checks all supported metadata operand signatures and engines against its instruction table. The checked-in JSON is a generated compatibility snapshot, not an independent source of model operand semantics; fresh derivation refuses a mismatch before simulation.

Compiler-authored adapters retain serialization/query syntax, timing-block and C++ class names, operand/token and resource-label translations, event-group names and conditional second-source markers. Model types, roles, dispatch and planner accesses determine their applicability. Exponent reads retain issue-time sampling; matmul weight reads retain the conservative lifetime policy. Representative operand numbers are extraction probes, not hardware constants.

Regenerate the compatibility snapshot after reviewing an intentional adapter change with `python -B tools/atlas_derive_vocabulary.py --npu-model /path/to/npu_model --output tools/model-timing-vocabulary.json`, then run the normal facts command. The normal command always rederives and checks the snapshot before validation; stale operand semantics cannot silently override the selected model.

## Overlap evidence

Isolated operation facts do not admit overlap. Pair, chain and composed evidence is a supplied simulation-measured catalog pinned to model revision `9e6313bddecb8ba936206eb347f5cd4e5335282c` and its locked configuration; it is not derived from source. Separate evidence covers exact ordered pairs, complete chains and bounded composed classes, with passing gaps and rejecting controls including the exact one-short gap. `tools/atlas_model_pairs.py` projects confirmed pairs from a sealed master and its validation receipt. `atlas_model_composition.py`, `atlas_model_chain_composition.py` and `atlas_model_composed_projection.py` replay and project composed evidence; `lib/AtlasModelPairs.cpp` checks it.

Evidence is bound to the exact facts-file SHA-256, not a semantic timing digest. The published selections used facts `896e1de72245a0f7eb02af46b3d689b789eaeb5837494bf67cc1ab750984341f`. Fresh derivation can reproduce the timing payload while changing the file digest: tool hashes and the validation report hash (which includes elapsed times) enter the facts bytes. Re-project pairs and replay/project chains and composed evidence against the new digest before combining them with regenerated facts. These tools reuse the supplied sealed master, successful receipts, query sets and holdouts; this repository cannot generate a new measured catalog. On another model revision or configuration, overlap evidence is unavailable: compile without `--pairs` and `--composed`, using isolated lifetimes. Published overlap-enabled results do not transfer to that fallback. Pair admission requires the covered operation/relation and gap. Larger live sets require complete chain or admitted composed coverage, all live pairs and structural capacity. Handwritten capacity, port-window, shared-engine, weight/accumulator and alias rules further constrain admission; matching the supplied measurements validates those rules without deriving them. Register, port, VMEM and DMA obligations remain enforced; waits invalidate earlier overlap eligibility. Scalar PACK overlap additionally requires the covered immutable aligned LW/SW address proofs. Without overlap evidence the compiler uses isolated resource lifetimes.

## Compile and select schedules

The lowering driver runs candidate selection:

```sh
python -B tools/lower_atlas_program.py input.mlir \
  --output-dir selected --atlas-bin-dir build/bin \
  --provider model --npu-model /path/to/npu_model \
  --op-timing /path/to/new-facts/model-timing.json --op-timing-sha256 OP_SHA
```

Replace `OP_SHA` with the lock's digest. Add `--pairs FILE --pairs-sha256 SHA` for confirmed pair/chain evidence and `--composed FILE --composed-sha256 SHA` for its composed side file. Composed evidence also requires its complete pinned base-pair file. Use `--model-region FILE` for representative DRAM input: JSON with `address`, `initial_hex` and optional `expected_hex`; repeat it for disjoint regions. Output directories must be empty. Derivation and evidence tools reject parent traversal and symlinks in input/output ancestry; model manifests fail if an excluded path name is present instead of silently omitting it.

`lib/AtlasModelTiming.cpp` implements the provider interface in `include/Atlas/AtlasTimingProvider.h`. Its costs guide virtual scheduling, allocation proposals, physical scheduling, delay slots, DMA waits and virtual-to-machine lowering. Scalar-slot save/restore/copy proposals combine address-materialization instruction costs with derived scalar load/store timing and releases. Missing facts cause refusal.

DMA proposals replay cold isolated transfers using derived startup, request interval, endpoint latency, fractional response pacing, rounding and finite pending capacity. This assumes pure read or store history and no VMEM grant denial; full native scoring decides the winner. Completion still requires explicit matching waits. The separate RTL target contract has eight DMA command slots: launch n reuses launch n−8's slot. The NPU Model does not enforce this ring, so the compiler moves a still-pending launch n−8's wait immediately before launch n and rejects streams that could overwrite a pending slot. `builtin` does not apply this ring rule; its pre-existing ring findings remain.

The stream scheduler fills a branch/jump delay slot with the block's last-issued single-cycle scalar ALU operation if it completes at issue and nothing later depends on it; otherwise it emits a NOP.

`tools/atlas_model_selection.py` compiles and verifies each candidate, then executes verified word streams on the unmodified native NPU Model in parallel processes. The scorer checks source, numerical-data and effective configuration identities against emitted evidence. Runtime errors, bank conflicts and supplied expected-output mismatches are fatal. Scores are cold cycles from first issue through drained ECALL. The emitted program is the scored program. `--workers` defaults to the smaller of 12 and the available CPU count (the library default is 12); lower it on shared hosts. Each scoring process uses one Torch thread.

Initial candidates cover the default, lookaheads 1/2/4, eight fixed random seeds, source-position waits, DMA staging-window variants and seeded legal MREG allocation ranks. The `full` preset adds single-operation and dependency-group reorderings, followed by PACK reload and selective-wait preload alternatives. Groups contain at most four nodes. Every candidate uses normal allocation, lowering and verification. A compiler-refused candidate records its error stage and cannot win; refusal of every default/lookahead/seed candidate stops with “no legal candidate.” A verified candidate's scoring failure stops compilation.

`--search-budget full` is the default: at most 32 initial, 96 single-operation and 204 group attempts. Decisions use deterministic batches of 12; a strict improvement restarts the neighborhood, and 96 consecutive nonimproving attempts stop a phase. `--search-budget selection` allows up to 24 initial attempts with no order search or PACK alternatives. Individual budget/Boolean flags override presets. Budgets count refused attempts and bound attempts rather than wall time.

Under curve, legal order proposals are ranked by cold replay of derived operation/DMA costs and exposed frontend waiting. Each order may try default and source-position wait placement, letting ready PACK/current-tile work precede future-transfer waits. Retained complete orders are bounded by the remaining attempt budget; estimating eligible orders still scales with neighborhood size. This ranking cannot choose a winner without native scoring.

Profiles define the objective. `curve`, the default, uses the selected model's default curve memory backend. `--profile fifo43` is a separately authored synthetic target: fixed-memory latency 43, one cycle per beat, and request acceptance on cycles 0–3 of each 11 (43/1/11/4), intended to match the standalone AtlasCore harness. These values are compiler literals, not extracted model-default facts. Exact curve-cycle ties are scored under FIFO43; equal scores on both profiles keep the earlier canonical candidate. This comparison applies to every incumbent and search restart. Explicit FIFO43 selection minimizes FIFO43 cycles alone. A curve improvement can increase FIFO43 cycles.

`--guarded-pack` converts eligible PACKs to MXU operations before selection. PACK overlap/reload/preload alternatives apply to PACKs that remain. `selection.json` records candidates, scores, failures, decision parents, accepted moves, tie participants and compile times; selected and candidate artifacts are retained.

## Updating for NPU Model changes

If derivation refuses or validation fails, identify the changed source and classify the change: a refactor may need extractor/source-shape updates; a timing change needs newly derived costs and validation; a new operation needs probe, schema, derivation and compiler mapping coverage. Review the relevant extractor and source contract together, and refresh accepted source shapes only after validating the revised extraction against native executions. Updating a pin alone does not establish support.

Derive fresh facts. For the same locked revision and configuration with unchanged timing content, re-project and replay the supplied pair/chain/composed catalog against the new facts digest using the sequence below, including positive/negative controls and required holdouts. For a new revision or configuration, compile without overlap evidence; generating a replacement measured catalog is outside this repository. Run compiler/evidence tests, representative selection and builtin identity checks.

For re-projection, obtain the supplied catalog's sealed master, validation receipt, query JSON and holdout-summary JSON. These external artifacts are required inputs and are not generated by the commands below. Set `MASTER`, `VALIDATION`, `QUERIES` and `HOLDOUTS` to those files; set `FACTS_DIR` to the freshly validated facts directory and `EVIDENCE_DIR` to a new output directory. For the published catalog the master digest is `c681093e34bcf7564d66328436dac2b4fb97789517578d9468cac99b4b43db03` and validation digest is `43e9d5381177dab8398b5b67156ce82a2196fe4672fb9c70ff192bbd6474e638`; use the digests in its supplied projection receipt, not defaults from an older catalog.

```sh
FACTS="$FACTS_DIR/model-timing.json"
OP_SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["op_timing_sha256"])' "$FACTS_DIR/facts.lock.json")
MASTER_SHA=c681093e34bcf7564d66328436dac2b4fb97789517578d9468cac99b4b43db03
VALIDATION_SHA=43e9d5381177dab8398b5b67156ce82a2196fe4672fb9c70ff192bbd6474e638
python -B tools/atlas_model_pairs.py \
  --master "$MASTER" --validation "$VALIDATION" --per-op "$FACTS" \
  --master-sha256 "$MASTER_SHA" --validation-sha256 "$VALIDATION_SHA" \
  --per-op-sha256 "$OP_SHA" --output "$EVIDENCE_DIR/base"
python -B tools/atlas_model_composition.py \
  --master "$MASTER" --per-op "$FACTS" --output-dir "$EVIDENCE_DIR/pair-replay"
python -B tools/atlas_model_chain_composition.py \
  --master "$MASTER" --per-op "$FACTS" --output "$EVIDENCE_DIR/chain-replay"
python -B tools/atlas_model_composed_projection.py \
  --per-op "$FACTS" --per-op-sha256 "$OP_SHA" \
  --base-pairs "$EVIDENCE_DIR/base/model-pairs.json" \
  --pair-audit "$EVIDENCE_DIR/pair-replay/pair-summary.json" \
  --chain-audit "$EVIDENCE_DIR/chain-replay/summary.json" \
  --queries "$QUERIES" --holdouts "$HOLDOUTS" --output "$EVIDENCE_DIR/composed"
```

Use the generated base receipt's `facts_sha256` with `--pairs "$EVIDENCE_DIR/base/model-pairs.json" --pairs-sha256 SHA`, and the composed receipt's `sha256` with `--composed "$EVIDENCE_DIR/composed/model-composed.json" --composed-sha256 SHA`; retain their successful replay/projection receipts. Re-projection changes the facts binding, not the supplied measured timings, and must not authorize changed timing semantics.

When adopting a second pinned model revision, also remeasure selected programs against the previous revision on matching inputs and profiles; recheck relevant RTL/SoC behavior when programs or timing change. Retain fresh identities and receipts before publishing new results. No facts cache or automatic overlap regeneration is involved.

## Results and limits

The provider still uses compiler-authored encoding/resource adapters, one-cycle ordinary issue and branch/delay conventions, and a one-cycle scratchpad response assumption shared by derivation and observation. Weight reads conservatively cover the isolated compute envelope unless admitted overlap supplies a tighter relation; unknown addresses reserve conservative resources. Waits invalidate overlap eligibility and block/control-flow boundaries drain work. Cold DMA and order replays are proposal heuristics with empty queues, pure read/store history, no VMEM grant denial, serialized engine approximations and handwritten edge offsets; native complete-program scoring chooses winners. PACK costs approximate the lowering template and available delay-slot filling; supported scale codes retain an empirical target gate. Search budgets, seeds, allocation ranks, 100000-cycle compiler bounds, 2048-word generated-code policy, spill-rewrite bounds and the 200000-cycle scorer timeout are engineering limits, not newly derived hardware timing.

The normal facts driver derives then validates its numeric payload, but the compiler's selected digest/schema checks do not establish that an arbitrary supplied artifact came from that driver; legacy/hand-authored isolated facts remain accepted by the interface. Hashes bind selected bytes and identity claims, and external validation-report bytes are not independently authenticated at compile time.

NPU Model cycles below compare matching `builtin` and `model` programs with the `full` preset on model revision `9e6313b`. Curve columns use curve-selected programs; FIFO43 columns use separately FIFO43-selected programs. Guarded columns apply guarded PACK to both providers. Percentages show cycle changes relative to matching builtin. EE290 SoC reports device cycles on EE290GenericIOSimConfig for all 44 matching builtin/curve-selected rows (11 benchmarks × two providers × plain/guarded PACK); every program hash matches the integrated r5 receipts. No FIFO43-selected SoC results are reported. Native model cycles and both-profile numerical checks are retained measurements transferred through exact program/input/model identity to r5 emissions, rather than fresh native measurements with the final binary.

| Benchmark | Curve | EE290 SoC | FIFO43 | Curve, guarded PACK | EE290 SoC, guarded PACK | FIFO43, guarded PACK |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention | 4,778 → 4,723 (-1.2%) | 5,440 → 5,406 (-0.6%) | 2,245 → 2,127 (-5.3%) | 4,778 → 4,723 (-1.2%) | 5,440 → 5,406 (-0.6%) | 2,245 → 2,127 (-5.3%) |
| MLP | 4,416 → 4,360 (-1.3%) | 5,030 → 5,033 (+0.1%) | 1,884 → 1,786 (-5.2%) | 3,958 → 3,905 (-1.3%) | 4,529 → 4,523 (-0.1%) | 1,071 → 983 (-8.2%) |
| Attention → MLP | 11,115 → 10,905 (-1.9%) | 11,805 → 11,650 (-1.3%) | 8,110 → 7,789 (-4.0%) | 9,593 → 9,174 (-4.4%) | 10,314 → 9,938 (-3.6%) | 3,986 → 3,755 (-5.8%) |
| Attention, 2 tiles | 6,389 → 6,223 (-2.6%) | 7,054 → 6,967 (-1.2%) | 3,820 → 3,524 (-7.7%) | 6,389 → 6,223 (-2.6%) | 7,054 → 6,967 (-1.2%) | 3,820 → 3,524 (-7.7%) |
| Attention, 4 tiles | 12,735 → 10,116 (-20.6%) | 13,569 → 11,088 (-18.3%) | 6,780 → 6,128 (-9.6%) | 12,735 → 10,116 (-20.6%) | 13,569 → 11,088 (-18.3%) | 6,780 → 6,128 (-9.6%) |
| Attention, 8 tiles | 21,051 → 18,870 (-10.4%) | 22,475 → 20,172 (-10.2%) | 12,720 → 11,741 (-7.7%) | 21,051 → 18,870 (-10.4%) | 22,475 → 20,172 (-10.2%) | 12,720 → 11,741 (-7.7%) |
| Attention, 16 tiles | 37,629 → 35,972 (-4.4%) | 39,690 → 38,213 (-3.7%) | 24,544 → 23,577 (-3.9%) | 37,629 → 35,972 (-4.4%) | 39,690 → 38,213 (-3.7%) | 24,544 → 23,577 (-3.9%) |
| MLP, 2 tiles | 7,074 → 6,044 (-14.6%) | 7,933 → 6,904 (-13.0%) | 3,326 → 2,963 (-10.9%) | 6,072 → 5,840 (-3.8%) | 6,751 → 6,648 (-1.5%) | 1,555 → 1,399 (-10.0%) |
| MLP, 4 tiles | 11,272 → 10,116 (-10.3%) | 12,293 → 11,100 (-9.7%) | 5,680 → 5,171 (-9.0%) | 10,347 → 10,116 (-2.2%) | 11,311 → 11,136 (-1.5%) | 2,886 → 2,259 (-21.7%) |
| MLP, 8 tiles | 19,555 → 18,869 (-3.5%) | 20,883 → 20,313 (-2.7%) | 10,586 → 9,640 (-8.9%) | 18,667 → 18,667 (0.0%) | 20,071 → 20,140 (+0.3%) | 5,350 → 3,887 (-27.3%) |
| MLP, 16 tiles | 36,110 → 35,769 (-0.9%) | 38,118 → 37,758 (-0.9%) | 20,475 → 18,957 (-7.4%) | 35,770 → 35,769 (0.0%) | 37,864 → 37,800 (-0.2%) | 11,433 → 7,812 (-31.7%) |

The 44 model-selected programs pass retained both-profile numerical checks with same-Core inputs 1,2,1 and the DMA ring checks. Builtin emitted words match the baseline on the checked 115-source corpus and 11 benchmarks in plain/guarded modes; metadata now records `builtin`, and the builtin delay-slot predicate was narrowed to single-cycle candidates without changing these checked outputs. This is corpus identity, not an identity guarantee for all programs. The results demonstrate bounded-search outcomes, without establishing global optimality, compile-time speedup or physical timing.

The curve backend models DRAM bandwidth without address sensitivity. Close model scores can reverse on the SoC: the measured plain MLP selection reduces model cycles from 4,416 to 4,360 while SoC device cycles rise from 5,030 to 5,033. Optimize and validate against the intended profile and target.
