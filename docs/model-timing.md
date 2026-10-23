# Model timing

`builtin`, the default provider, uses handwritten Atlas timing rules. The opt-in `model` provider derives operation timing, reservation releases, architecture, operand vocabulary, frontend timing, MXU weight-use lifetimes and curve/DMA costs from the selected [NPU Model](https://github.com/ucb-ee194-tapeout/npu_model) source and effective configuration. Supported overlap relations come from reviewed source semantics qualified by a bounded native gate. Verified complete candidates run on that model to select schedules. Compiler encoding adapters, capability limits, numerical gates and target assumptions remain explicit.

## Regenerate and validate

In the selected model's Python environment, run:

```sh
python -B tools/atlas_model_facts.py --npu-model /path/to/npu_model --out /path/to/new-facts
```

The output directory must be new. This derives the complete timing bundle, validates isolated operations and curve costs against independent native executions, then validates frozen source-derived overlap rules. Successful qualification publishes `model-timing.json`, `model-overlap.json` and `facts.lock.json`; failures can leave diagnostics. No measured overlap catalog or benchmark winner is a derivation input.

The `atlas_derive_*` extractors check reviewed source structures and effective `DefaultHardwareConfig` values. Timing source shapes live in `tools/model-timing.lock.json`; architecture, vocabulary, frontend, scratchpad and overlap extraction enforce AST/planner contracts. These reviewed contracts use Python 3.14. Unsupported structures or capabilities refuse. Compilation retains supported 32×32 geometry, register/bank layout and memory map, ISA/execution-unit classes, RTL numerics, deterministic initialization, fatal runtime errors and curve-controller semantics.

Before importing or executing Core, facts generation and native scoring also run `atlas_model_semantics.check_model_semantics`. Its reviewed `tools/model-semantics.json` inventory independently fingerprints every helper in 17 frontend/storage/ISA modules, plus their imports, globals, class bases, signatures, decorators and bindings. Unknown helper bodies, missing/added helpers and late overrides refuse with `unsupported semantics: FILE:HELPER`; checking `ignore_runtime_errors=False` alone is insufficient. The full `Core._handle_runtime_error` implementation is guarded, including the false-flag branch. Scalar width extraction checks the complete final store and X0 behavior, while the shared guard covers initialization, reset and transitive storage helpers.

The inventory appears in generated `lineage.semantic_dependencies`. The existing `model.source_sha256` field in the `atlas.derived_facts.v1` semantic payload binds the actual helper source bytes; generator identities bind the interpreter and reviewed inventory. Strengthening these compatibility/validation checks adds no scheduling input, so the supported unchanged model retains its semantic digest. The inventory is not a proof of arbitrary Python semantics or an authentication mechanism for externally authored facts.

| Contract dependency | Explicit source guard owner |
| --- | --- |
| Fatal errors, Core initialization/reset, S1 dispatch/hold, load and termination | Complete `hardware/core.py`, including `_handle_runtime_error`, recovery helpers and `exu_map` construction; configuration checks exact execution-unit membership/order. |
| Fetch, synchronous instruction memory, pipeline validity/claim/stall, instruction/PC handoff | Complete `hardware/ifu.py`, `hardware/stage_data.py`, `software/instruction.py`, `software/program.py` and base `hardware/hardware.py`. |
| Scalar completion and shared staged execution | Complete `hardware/exu.py`, including base/scalar constructors, reset, completion state, `StagedExecution` and `_Memory`. |
| Deterministic storage initialization/reset, XRF/ERF/CSR/flags, redirect, BF16 underlying reads/writes, DRAM/VMEM bounds and sparse storage | Complete `hardware/arch_state.py` and `hardware/sparse_memory.py`; architecture extraction then reads supported capacities/mappings. |
| ISA dispatch, operand construction and scalar execution helpers/constants | Complete `isa.py`, `isa_types.py`, `isa_patterns.py`, `configs/isa_definition.py` and package initializers; includes `_sign_extend`, `_MASK32`, memory helpers and registration/metaclass behavior. |
| Banking, deferred releases, VMEM ownership, accumulator/weight acquisition | Existing whole `bank_conflict.py` AST guards in VPU/MXU/memory/overlap derivation cover every helper independently of architecture's mapping extraction. |
| Engine constructors/reset/admission/read/commit and memory backend | Existing VPU whole-source guard after latency-table extraction, MXU whole-source guard after planner-expression extraction, LSU/XLU whole-source guards, and DMA/backend source contracts in curve derivation; shared EXU/storage callees are covered above. |

The new normalization ignores comments, locations and genuine leading docstrings. It accepts renamed nonargument locals only in simple scopes without closures, comprehensions, pattern/exception bindings or namespace reflection. It accepts reordered adjacent ordinary methods only in reviewed classes with safe signatures/defaults; class assignments and complex decorators/annotations remain order barriers. Arguments, annotations, logging calls, executable statements and external bindings remain checked. Existing engine guards still conservatively refuse the audit's VPU local rename and method reorder; their tolerance was not broadened. Configuration interpretation, Python/PyTorch behavior and the selected numerical routines remain explicit external boundaries, with source/data identity and existing numerical validation rather than a new arithmetic proof.

`runtime-validation.json` records five positive and five negative full native Core programs: VPU issue-busy, VMEM and MRF conflicts, invalid VPU pair operands and illegal control flow in a branch delay slot. Positive controls must reach drained ECALL; negative controls must propagate the specified native exception with the fatal-error flag unchanged. The 512-cycle bound, operands and drain slack are hand-chosen validation coverage, never timing facts or derivation authority. The production facts command regenerates this report and requires it before publishing a lock; source guards establish compatibility before these tests run.

Operation probes come from `IsaSpec`. Validation checks zero/nonzero inputs with instrumentation enabled/disabled, control traces, progress, drain, event ages and releases. Curve validation compares an independent fact interpreter with native backend streams and isolated/mixed DMA, including backpressure, translation and scaling. The driver checks source, numerical data and derivation/observer tools remain unchanged. These gates qualify reviewed semantics, not arbitrary-program completion.

Production requires `atlas.derived_facts.v1`: successful derivation/validation records bound to deterministic semantic inputs, vocabulary, complete releases and MXU weight-use inputs. The payload includes `overlap_rules_sha256`, computed canonically from `{schema, age_convention, contract, source_contract, pairs, composition}`, excluding `producer_sources` from `source_contract`. The compiler independently hashes those semantic fields from the overlap artifact and checks the payload binding; pair rows are not duplicated in timing facts. Changes to rules therefore invalidate the bound qualification. Raw observations and isolated synthetic fixtures are restricted to test/validation APIs.

The lock records selected file and semantic digests. Elapsed times belong in `timing-receipt.json`, outside semantic identity. Archive manifests, frozen candidates, sample descriptors and native reports. Hashes and internal records provide integrity and traceability; compilation neither reexecutes derivation nor independently authenticates an external author or native report.

## Provenance and assumptions

Architecture extraction combines configuration with allocation/banking implementation: VMEM capacity/banks; MREG geometry/count; weight/scalar/exponent/CSR capacities; DMA channels/beat/transfer/in-flight limits; DRAM range; fetch width; and bus/link geometry. `BankConflictChecker` supplies bank/row mapping and logical-register guards; BF16 helpers supply pair layout; `ArchState.write_xrf` supplies scalar width. Consumers still check their supported layout, including 32 physical MREG banks, 32 rows, 1024-byte registers and six 262144-byte VMEM banks.

Operand vocabulary comes from ISA operand types/roles and VPU/MXU planners, including directions, register counts/offsets, row passes and accumulator/weight/exponent accesses. ISA metadata checks compiler instruction signatures and engines against the generated compatibility snapshot. Compiler adapters retain encoding/query syntax, names and token/resource translations; exponent sampling remains at issue and weight lifetimes use the source-derived row windows below. After reviewing an adapter change, regenerate the snapshot with `python -B tools/atlas_derive_vocabulary.py --npu-model /path/to/npu_model --output tools/model-timing-vocabulary.json`, then regenerate facts.

Frontend extraction checks Core/scalar-unit execution, IFU/architecture redirect ordering and scalar ISA implementations. It derives ordinary issue timing, scalar accesses, branch/JAL delay-slot behavior and DELAY semantics. The reviewed model gives `1 + (imm & 0xFFF)`; compiler scheduling currently supports ordinary/base gap one and that 12-bit mask, refusing other values until adapters support them.

Curve facts carry unloaded latency, every read/write mix's response pacing, queue/history limits, channel/frequency scaling, DMA startup/request intervals and rounding/reset behavior. The reviewed controller uses unloaded latency, converge=1 and peak pacing. Queue capacity differs from engine in-flight capacity. Canonical configuration, source and curve bytes bind compilation/scoring; hashed dormant overrides do not drive the selected backend.

Scratchpad response latency remains the explicit one-cycle assumption in `scratchpad_assumptions()`. VPU `_read`, MXU `_sample`, LSU and XLU synchronously sample tensors on request edges; no independent SRAM response event exists. DMA store staging uses `cycle + 2`, without separating SRAM response from queue staging. Extraction checks these pipelines and derives aggregate staging, but cannot derive standalone SRAM latency. `scratchpad_contract` labels this limitation; observer comparisons exclude the assumed field.

## Source-derived overlap

`atlas_derive_overlap.py` and `atlas_overlap_rules.py` interpret reviewed source predicates, isolated event ages and deferred releases to compute suffix-safe pair gaps and composition classes. Native runs validate frozen rules rather than fit minima. Supported configuration changes regenerate matching rules; production does not silently replace changed configuration overlap with isolated-only scheduling.

Structural checks require 32-cycle MXU sequencers, MXU0/MXU1 compute tracker depths 3/2, two VPU slots, 32×32 geometry and 64 logical MREGs. Coverage includes complete MXU cross-command register/bank relations, selected legal VPU assignments and immutable aligned scalar LW/SW contexts. Unknown relations retain isolated lifetimes; the complete VPU operand-equivalence partition and mixed VPU/XLU overlap remain uncovered.

The bounded native gate samples operation/mechanism boundaries independently of benchmarks: minima and later suffix points, one-short controls, zero/nonzero operands, exact issue ages, complete drain and serial-reference final state. Chains exercise slot/tracker limits, nonadjacent aliases and retired-prefix port history. Zero-gap controls are encoding rejections. A passing one-short control records conservatism without changing the derived minimum.

Every admission still requires the exact relation/gap; larger live sets require every live pair, supported composition and dynamic capacity. Port/register holds, retired-command history, weight/accumulator ownership, alias proofs and epoch boundaries remain enforced. Waits invalidate earlier eligibility. DMA history/grants/backpressure exclude it from fixed overlap windows. Bounded samples do not prove arbitrary operand/data cases, mixed chains or RTL equivalence.

## MXU weight-use lifetime

`mxu_weight_timing` (`atlas.mxu_weight_timing.v1`) derives the weight masks from `hardware/mxu.py:_MatrixExecutionUnit._weight_uses`, with reviewed structure checks for the row/column mapping, sampling and push-write edge order. The inner engine reads every weight at ages 1 through 32. Systolic PE(i,j) reads at ages `1+i+j` through `32+i+j`; the weight-buffer row is the output-column axis `i`. Because a push replaces a whole row, that row's union of uses is `1+i` through `63+i`. The facts carry the affine PE window and all 32 projected row windows for each engine, bound into the derived semantic payload. Unsupported masks, mappings or sampling order stop derivation; missing or inconsistent windows stop production loading.

The provider represents each row's first and last use instead of the former age-zero-to-busy-end envelope. The existing ordered-resource dependency calculation then derives push-to-matmul/matmul-to-push gaps of 1/63 for MXU0 and 32/32 for MXU1 with the pinned model's row writes at ages 1 through 32. This preserves strict dependency ordering and the model's issue predicates even though model sampling precedes writes on an edge. Both matmul modes and both weight slots use the same derived windows. A push to another slot creates no weight data dependency.

Weight dependencies come directly from the ordered-resource calculation, independently of overlap admission. Engine occupancy, MREG ports and releases, slot restrictions, complete overlap epochs, capacities and all other admission checks remain enforced by their existing paths. In particular, precise weight windows alone do not allow overlapping instructions without separately admitted engine/resource use. Source-derived overlap rules retain their separate admission checks.

Fresh fact generation validates every PE's first and last use against isolated native model execution, checks boundary mutations, and runs native streamed weight-push boundaries with rejecting one-cycle-early controls. Validation observes or challenges derived windows; it does not supply them. The retained validation report records the cases and outcomes, while elapsed timing remains a separate receipt.

## Compile and select

```sh
python -B tools/lower_atlas_program.py input.mlir \
  --output-dir selected --atlas-bin-dir build/bin \
  --provider model --npu-model /path/to/npu_model \
  --model-facts /path/to/new-facts
```

The loader reads fixed filenames and locked digests; separate pair/composed files are unnecessary. Omit `--model-facts` to regenerate automatically. The compiler pass accepts `--select-atlas-model-evidence="facts=/path/to/new-facts"`. Use repeated `--model-region FILE` options for disjoint DRAM inputs with `address`, `initial_hex` and optional `expected_hex`. Outputs must be empty; path checks reject traversal, symlink ancestry and prohibited names before traversal.

Derived costs guide scheduling, allocation, delay slots, waits and lowering. Cold DMA/order estimates are proposal heuristics assuming empty queues, pure read/store history and no VMEM grant denial. Verified emitted candidates execute on the unmodified native model; source/numerical/configuration mismatches, runtime errors, conflicts and supplied-output mismatches are fatal. Scores cover first issue through drained ECALL; the winner's emitted words are the scored words.

DMA completion requires matching waits. The eight-slot command ring, `kAtlasDMACommandSlots`, is a separate RTL contract absent from model-derived architecture. The model does not enforce it. Launch n reuses n−8's slot; the compiler moves a pending n−8 wait before n and rejects pending-slot overwrite. Builtin retains existing behavior.

Default `full` search bounds initial/order/group attempts at 32/96/204, using deterministic batches of 12 and groups up to four nodes; `selection` allows 24 initial attempts without order search/PACK alternatives. Refused attempts consume budget. Native scoring chooses winners. `--guarded-pack` rewrites eligible PACKs to MXU operations. Remaining PACK scale codes are restricted to observed E8M0 codes 122..128: an empirical numerical target gate, not derived timing or universal numerical proof.

## Scoring targets

Default `curve` uses the checked model's curve backend. `fifo43` selects `standalone-atlascore-fifo43` in `tools/model-scoring-targets.json`: fixed latency 43, one cycle/beat, acceptance on absolute backend cycles 0–3 of each 11. This explicitly synthetic harness target is neither model-default nor extracted RTL timing. A versioned adapter maps its complete periodic descriptor to the model's `accept_fn` hook.

`selection.json` records every used native profile's effective configuration, source/numerical/base identity, synthetic provenance and canonical SHA-256 excluding `sha256` itself. Callback receipts contain the adapter/descriptor; curve receipts contain checked curve data. Workers share an invocation snapshot. Exact curve ties use FIFO43, retaining its receipt; double ties retain the earlier candidate. Explicit FIFO43 minimizes that profile alone. Thus default selection retains a synthetic secondary objective.

## Updating for NPU Model changes

For supported parameter changes, edit the selected configuration (for example, change `dma_memory_params["frequency_ghz"]` from `0.5` to `0.75`), regenerate into a new directory, and compile/score with that checkout and bundle. Operations, curve costs and overlap regenerate/revalidate together. Unused backend fields cannot change timing. Unsupported geometry, structures or refactors require reviewed extractor/consumer updates rather than pin changes.

Rerun relevant compiler/evidence tests, representative searches, numerical checks and DMA ring checks; preserve program/input/source/configuration identities and regressions. Historical results do not qualify this integrated bundle. Search limits and conservative policies remain engineering choices, not proof of optimality or physical timing. Curve bandwidth lacks address sensitivity, so close model rankings can reverse on a SoC; validate the intended target.
