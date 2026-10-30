# `npu_model` timing

A timing provider supplies the rules used by scheduling, waits, delay slots,
and lowering. The default `builtin` provider uses handwritten rules ported
from [`npu_model`](https://github.com/ucb-ee194-tapeout/npu_model). The `model` provider reads facts extracted from a selected
`npu_model` checkout. A candidate is one compiled program; selection runs
candidates on `npu_model` and keeps the one with the fewest cycles. A profile
chooses the memory behavior used to score those programs.

## How the pieces fit

```
npu_model checkout ──► tools/atlas_model_facts.py ──► model-timing.json
  (source + config)      reads each unit's timing         │
                         code, checks it natively         ▼
                                              lib/AtlasModelTables.cpp
                                              (load, check, TimingTables)
                                                          │
                                                          ▼
source.mlir ──► compiler passes ◄── `model` timing provider
                  │  (virtual scheduling, stream scheduling,
                  │   DMA waits, delays, timing verification)
                  ▼
           candidate programs ──► tools/atlas_model_scorer.py ──► cycles
                  ▲                (runs each on npu_model,           │
                  │                 checks expected bytes)            │
                  └────────── tools/atlas_model_selection.py ◄────────┘
                               (keeps the fastest; program.hex)
```

1. Extraction. `tools/atlas_model_facts.py` reads a `npu_model` checkout.
   The `tools/atlas_derive_*.py` modules interpret the ISA, architecture,
   frontend, VPU, MXU, LSU and XLU, DMA, and curve memory code.
   `tools/model-pins.json` records the source digests that interpretation was
   reviewed against, so an unreviewed change refuses instead of producing
   stale facts. Each of the 49 timing blocks is then run alone in
   `npu_model` and compared with the extracted events before
   `model-timing.json` is written.
2. Loading. `lib/AtlasModelTables.cpp` reads the facts, checks them against
   the compiler's target geometry and operand metadata, and builds
   `TimingTables`.
3. Compilation. With `provider=model`, the existing scheduling, DMA wait,
   delay, and timing-verification passes do the work; the shared rules in
   `lib/AtlasTiming.cpp` (footprints, dependences, reservations) read the
   tables where the `builtin` provider uses its handwritten values.
4. Selection. `tools/atlas_model_selection.py` compiles a fixed set of
   candidate schedules, `tools/atlas_model_scorer.py` runs each one on
   `npu_model` with the supplied expected regions, and the fastest exact
   candidate is written out. `tools/lower_atlas_program.py --provider model`
   is the entry point.

## Regenerating the facts

Use Python 3.14 with the model's dependencies available:

```sh
python -B tools/atlas_model_facts.py --npu-model PATH --out NEW_DIR
```

The output directory must be new. The command writes `model-timing.json`,
containing operation events and the parameters consumed by the compiler.
Regenerate it when changing the model checkout or supported configuration.
The tested checkout is `npu_model` revision
`74982e7c68bd28ba6505b4df37290f198102ccce`.

One guard table, `tools/model-pins.json`, records digests for 17 modules and
seven engine shapes. Central contracts ignore comments and leading
docstrings. Local renames and method reordering require source acceptance.
VPU slot and latency-table normalization and MXU timing and tracker
normalization remain supported. Extraction and loading check configuration;
geometry must equal the compiler target geometry.

After inspecting a source change and updating its interpretation where needed,
re-pin with the single command
`python -B tools/atlas_model_facts.py --npu-model PATH --accept-source`.
It prints each changed pin and updates the tested revision. This explicit
mode is separate from `--out`; regeneration never calls it or changes pins.

| Guarded `npu_model` input | Facts the compiler consumes |
| --- | --- |
| ISA, Core, scalar unit, and fetch unit | Operand roles, issue ages, delays |
| Architecture and bank mapping | Register spans, sizes, and VMEM layout |
| VPU planners and issue logic | Event ages, releases, groups, and capacity |
| MXU planners and weight helpers | Events, ports, capacities, row windows |
| LSU and XLU paths | Memory events, releases, and load response spacing |
| DMA, backend, configuration, and curve | Startups, issue spacing, costs |

Regeneration runs one isolated native instruction for each of the 49 timing
blocks. It compares every emitted event stream and the first age with no
instruction in flight. A mismatch refuses regeneration. Reservation releases,
MXU weight windows, DMA estimates, and concurrent issue rely on extraction
and compiler checks. Source files recorded in the facts are checked again
before output is written. A changed contract refuses output.

## Testing

Regenerate facts from the checkout used for testing, then configure CTest with
both paths set:

```sh
export ATLAS_DERIVE_MODEL_ROOT=/path/to/npu_model
python -B tools/atlas_model_facts.py --npu-model "$ATLAS_DERIVE_MODEL_ROOT" \
  --out /path/to/new-facts
export ATLAS_TEST_MODEL_FACTS=/path/to/new-facts
cmake -S . -B build
ctest --test-dir build --output-on-failure
```

`ATLAS_DERIVE_MODEL_ROOT` enables extraction, scorer, and spill execution tests.
`ATLAS_TEST_MODEL_FACTS` enables the native model-table and timing-provider
checks and Python suites marked `requires_model_evidence`, including model
selection and pipeline execution. Without these inputs, those checks skip;
the portable tests still run. CMake forwards variables set at configuration
time to both native and Python tests. Direct Python test runs read the same
environment variables.

## What comes from `npu_model` and what the compiler chooses

Scalar reads and writes at age zero follow the guarded issue-edge methods.
The loader checks scalar operand roles against compiler metadata. VPU operand
roles and two-input classification are compiler mappings. Physical scheduling
reads `TimingTables` directly and uses shared dependence and reservation rules;
the provider checks scope, operands, and the completed schedule. Provider
construction precomputes machine timing for virtual costs and refuses missing
operations by name.

The compiler chooses these policies and structural mappings:

- Native execution uses `rtl` numerics and disables randomized initialization.
- Search budgets, seeds, candidate pools, group size, and batch width.
- Unit capacity one outside MXU compute; VPU issue slots have their own
  extracted admission check. MXU compute capacities come from the facts.
- MXU source/result port windows span zero through the sequencer age minus
  one. MXU1 weight overwrite and both engines' weight use after push use
  the extracted sequencer duration.
- Probe fields use the loaded BF16 register span: non-MXU destination and
  sources occupy spans zero, one, two, then three, four, five. MXU probes
  start at zero; the second uses one span for matrix operands, zero for
  scale operands, and one for other fields. Scalar destination uses three
  spans. ISA metadata selects the live fields.
- DMA estimates use startup plus the beat count times the sum of the rounded
  up maximum curve latency/per-beat cost and issue interval. The maximum
  includes the configured minimum latency. Estimates serialize with a
  one-cycle gap; the extracted retire-to-wait age places the proposed wait.
- PACK relayout copies one row per iteration for every provider. Selection
  waits before reuse of an eight-slot DMA command-ring entry for every
  provider; this wait is an RTL rule. PACK cost omits general scalar setup and
  assumes the destination increment fills a delay slot when filling is enabled;
  physical scheduling proves that an actual filler is available.
- Native scoring stops at 200,000 cycles. Same-unit interval search stops
  at 100,000 cycles, and model program scope is at most 32,768 instructions.
- The synthetic `fifo43` target uses latency 43, one cycle per beat, and four
  accepted cycles in each eleven-cycle period.

DMA costs conservatively bound the selected curve for ordering. Matching
`dma.wait` operations establish completion; an estimate cannot replace them.

## Compiling with the model provider

This command selects a program and checks its supplied expected bytes:

```sh
python -B tools/lower_atlas_program.py source.mlir --provider model \
  --npu-model PATH --model-facts NEW_DIR --profile curve \
  --search-budget full --model-region expected.json --output-dir OUT
```

Each region file contains `address`, `initial_hex`, and optional `expected_hex`.
Use `expected_hex` for exact numerical checking; repeat `--model-region` for
outputs, inputs, and guards. An expected-byte mismatch stops the whole search.
The command checks the supplied regions, so supply all regions needed for the
program's numerical contract.

Omitting `--model-facts` regenerates facts under the output directory. Before
running a candidate, the scorer checks model sources, configuration sources,
and configuration values against the facts. This also applies to supplied
facts. The `curve` profile also refuses different curve bytes and binds the
checked bytes to the backend. The compiler checks the facts file again on
reload.

The PACK relayout pass replaces eligible scalar copies with VPU PACK and
UNPACK followed by an MXU push and readout. The roundtrip cost uses the shared
SELI, PACK, and UNPACK timing rules from the provider's precomputed machine
operation timings. The second virtual schedule keeps the selected provider
and random seed, and uses the same automatic lookahead selection as the first.
Explicit search orders stay applied after PACK expansion, and graph output
keeps the prepared graph used by group search. The schedule stage records its
options in the intermediate IR, and the spill stage restores them for this
second pass, including when the stages run as separate commands.

The default profile is `curve`; `--profile curve|fifo43` changes scoring memory
behavior. `--guarded-pack` converts proven-finite PACK inputs through MXU0 for
exact results. `--workers` controls parallel work. `--no-bank-candidates`
skips the eight MREG rank vectors in full mode.

The output contains `selection.json`, `program.hex`, `program.json`,
`allocation.json`, and `final.mlir`, plus `commands.txt`, `log.txt`, and each
attempted candidate's files. The selected words are exactly the scored words.
Choose a fresh output directory to prevent stale program artifacts.

## How selection works

1. Prepare the source once, then try the default candidate and fixed random
   seeds. Try source-wait placement from the current winner, then MREG rank
   vectors. Candidate numbers are assigned before parallel compilation.
2. Compile and verify every candidate. Execute legal words natively through
   ECALL, require drained engines and DMA, and check supplied DRAM regions.
3. Keep the lowest cycle count, breaking ties by the earliest candidate
   number. Duplicate words reuse their score within the same selection.
4. Move groups of up to four dependent operations from the winning order.
   Evaluate batches of twelve, including default/source wait siblings for
   the curve profile, and restart from an improved winner.

`--search-budget full` allows 18 initial attempts and 134 group attempts.
`--max-candidates` changes the initial limit; `--search-group-evaluations`
changes the group limit. Compiler refusals consume attempts; native scoring
faults stop the search. Exhausting legal neighbors may end it before the limit.
`--search-budget single` tries exactly the default candidate, with no MREG
or group search. Fixed seeds and index-based ties make candidate order and
winner independent of worker count and machine load.

## Updating for `npu_model` changes

Supported source changes are the `VPU_OP_LATENCIES` table and integer
`ISSUE_SLOTS` from 1 through 3 in `npu_model/hardware/vpu.py`; and the
`MXU_OP_LATENCIES` table, `compute_first`, tick row-window endpoints,
release/completion expressions, and integer `inflight_depth` from 1 through 64
in `npu_model/hardware/mxu.py`. These still must satisfy the extractor's
latency, row geometry, release, and native-event checks. Positive curve data
changes regenerate through the selected `CurveMemoryBackend`, subject to its
validation and the extractor's unloaded-mode, pacing, and geometry checks.

Other timing-engine literal or structure changes refuse. This includes LSU
timing, XLU `TRANSPOSE_CYCLES`, DMA timing, bank-conflict logic, and MXU port
literals; these files or expressions stay pinned. ISA changes, unsupported
configuration, and geometry differing from the target also refuse. For example:
`unsupported semantics: hardware/core.py: module contract changed`.

Configuration literals in `npu_model/hardware/config.py` and
`npu_model/configs/hardware/default.py` are checked by their resulting values.
Outstanding-DMA depth, maximum transfer size, and curve-backend parameters may
change within extraction, backend, and loader bounds; target geometry,
dispatch, and backend mode remain fixed.

Inspect the named file, then update the matching `tools/atlas_derive_*.py`
interpretation and the C++ reader when their representation changes. Inspect
the source diff, run the explicit `--accept-source` command above, regenerate,
run full CTest, and rerun the selection command above with exact expected
regions before using new facts. See
[the extractor](../tools/atlas_model_facts.py) and
[the loader](../lib/AtlasModelTables.cpp) for implementation details.

## Limits

`npu_model` is a software model, so its cycles are not RTL measurements.
Isolated native event checks do not test arbitrary concurrent programs.
Curve estimates ignore address effects, and close rankings can reverse on
the SoC. Exactness depends on the chosen numerical policy and input regions.

Timed IR from main may retain `atlas.timing_provider = "npu-model-rtl-match-v1"`.
The compiler normalizes that identity to `builtin` when reading it. Explicit
`provider=builtin` also accepts it; newly timed artifacts use `builtin`.
