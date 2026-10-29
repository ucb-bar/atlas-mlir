# NPU Model timing

A provider supplies the timing rules used by scheduling and verification.
The default provider uses handwritten rules ported from npu_model. The
`model` provider uses the same rule functions with timing tables extracted
from an npu_model checkout. Scheduling, waits, delay slots, and lowering
use those tables. Model selection compiles candidates, runs them on
npu_model, and keeps the fastest program. A candidate is one combination
of scheduling and allocation choices. The default provider is unchanged.

## Regenerating the facts

Facts are the timing values extracted from the model source and its default
configuration. Generate them into a new directory with one command:

```sh
python -B tools/atlas_model_facts.py --npu-model /path/to/npu_model \
  --out /path/to/new-facts
```

Regeneration takes about ten seconds on the tested checkout.
`model-timing.json` contains the operation events, release ages, frontend
values, engine resources, weight windows, architecture, and DMA estimates.
`facts.lock.json` selects the timing file by its content hash.
`validation-report.json` records the native VPU row-sweep check.
The check compares actual port events and drain with the extracted values
and rejects a wrong fact. `tools/model-timing.lock.json` pins the tested
model revision. Regenerate when changing revisions instead of editing the
facts by hand. See `tools/atlas_model_facts.py` for the extraction entry point
and `lib/AtlasModelTables.cpp` for the compiler reader.

## What comes from npu_model and what the compiler chooses

| Origin | Values |
| --- | --- |
| Model source and configuration | Operation events, release windows, and frontend timing |
| Model source and configuration | Geometry, engine capacities, ports, and VPU groups |
| Model source and configuration | MXU weight endpoints and sequencer windows; DMA curve costs |
| Compiler policy | Search budgets, seeds, and one-cycle DMA estimate serialization |
| Compiler policy | Two-row PACK grouping, eight-slot DMA command ring wait, and synthetic fifo43 scoring |

The model provider reads every timing value used by its shared rules from
these extracted tables. Recognized changes to ages or capacities flow into
the compiler when facts are regenerated. The original calibrated DMA
estimator belongs to the default provider. Model estimates use the extracted
startups and conservative per-beat curve bound. Both endpoints of each
MXU weight-use row are retained, including the final systolic use.

DMA estimates guide ordering and wait placement. A matching wait still
establishes completion, and the command ring wait remains a separate RTL
rule. The compiler checks supported geometry and event shapes before using
the tables. Missing events, invalid lifetimes, and unsupported geometry
produce a diagnostic instead of using another provider's timing values.

## Compiling with the model provider

Compile and select a program using the generated facts:

```sh
python -B tools/lower_atlas_program.py input.mlir \
  --output-dir selected --atlas-bin-dir build/bin \
  --provider model --npu-model /path/to/npu_model \
  --model-facts /path/to/new-facts --profile curve \
  --model-region /path/to/expected-region.json
```

Omitting `--model-facts` regenerates facts into the output directory. A
profile is the memory configuration used to score candidates. `curve` uses
the configured memory curves. `fifo43` uses fixed latency and a periodic
readiness pattern for comparison. Both profiles use the same extracted
compiler timing tables.

The main selection options are:

- `--profile curve|fifo43` chooses the scoring configuration.
- `--search-budget` chooses a bounded search preset.
- `--max-candidates` limits the initial candidate pool.
- `--guarded-pack` enables exact PACK for supported finite inputs.
- `--model-region` supplies an expected memory region for numerical checks.

The output directory contains `selection.json`, `program.hex`,
`program.json`, `allocation.json`, and the selected candidate's IR.
Candidate directories retain the compiled programs and their diagnostics.
The selection file records attempts, cycle counts, and the winning index.

## How selection works

1. Build candidates in a fixed order from fixed seeds and number them.
2. Compile each candidate and run the normal compiler verifiers.
3. Run each candidate on npu_model through drained ECALL and check the
   supplied output regions.
4. Keep the lowest cycle count, breaking ties by the lowest candidate index.
5. Search dependency-group moves from the current winner while the configured
   budget allows it.

Compiler refusals count toward the budget. Native execution faults and
numerical mismatches stop selection. Scores for identical instruction words
are reused within one search. Selection is deterministic for a fixed worker
count and configuration, and the winner's words are exactly the scored words.

## Updating for NPU Model changes

Supported latency, curve, startup, and capacity changes regenerate from the
model. The extractors refuse helper structures they cannot interpret and
configuration or geometry outside the supported set. For example, a changed
MXU row schedule can report "unsupported MXU sequencer geometry (requires
32 cycles)". A diagnostic identifies the helper or condition to inspect.
Update the corresponding `tools/atlas_derive_*.py` extractor and, when its
supported shape changes, the compiler reader. Then regenerate twice, run
CTest and the native tests, compare consumed timing values, and rerun
representative selections with exact numerical checks.

## Limits

npu_model is not RTL. Isolated events and extracted engine resources feed
shared dependence and reservation rules; native scoring checks each actual
candidate. Structural guards can refuse a harmless source refactor. The VPU
spot check is bounded and does not cover every concurrent instruction stream.
Curve estimates ignore address effects, so close rankings can reverse on
hardware.
