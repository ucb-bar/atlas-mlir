# NPU Model timing

A timing provider supplies the rules used by scheduling, waits, delay slots,
and lowering. The default `builtin` provider uses handwritten rules ported
from npu_model. The `model` provider reads facts extracted from a selected
npu_model checkout. A candidate is one compiled program; selection runs
candidates on npu_model and keeps the one with the fewest cycles. A profile
chooses the memory behavior used to score those programs.

## Regenerating the facts

Use Python 3.14 with the model's dependencies available:

```sh
python -B tools/atlas_model_facts.py --npu-model PATH --out NEW_DIR
```

The output directory must be new. The command writes one file,
`model-timing.json`, containing operation events and the parameters consumed
by the compiler. Regenerate it when changing the model checkout or supported
configuration. The tested checkout is npu_model revision
`74982e7c68bd28ba6505b4df37290f198102ccce`.

One guard table, `tools/model-pins.json`, records one digest for each of 17
modules and whole-engine shapes for seven other files. Central contracts
ignore comments and leading docstrings. Local renames and method reordering
require source acceptance. Engine slot, tracker, and latency parameters retain
their supported normalization. Extraction and the compiler loader check
configuration and geometry limits.

After reviewing source changes, update the pins with
`python -B tools/atlas_model_facts.py --npu-model PATH --accept-source`.
The command prints changed pins; regeneration never updates them.

| Guarded npu_model input | Facts the compiler consumes |
| --- | --- |
| ISA, Core, scalar unit, and fetch unit | Operand roles, issue ages, delays |
| Architecture and bank mapping | Register spans, sizes, and VMEM layout |
| VPU planners and issue logic | Event ages, releases, groups, and capacity |
| MXU planners and weight helpers | Events, ports, capacities, row windows |
| LSU and XLU paths | Memory events, releases, and load response spacing |
| DMA, backend, configuration, and curve | Startups, issue spacing, costs |

Regeneration also runs one isolated native instruction for each of the 49
operation timing blocks. It compares every emitted event stream and the
first age at which the unit has no instruction in flight. A mismatch refuses
regeneration. Reservation releases, MXU weight windows, DMA estimates, and
concurrent issue rely on their guarded extraction and compiler checks.

## What comes from npu_model and what the compiler chooses

Timing facts retain their source or configuration derivation. Scalar reads
and writes at age zero follow the guarded issue-edge methods. The loader
checks scalar operand roles against compiler metadata and architecture
values against target limits. Cost-probe operands use the loaded BF16 span.
Missing events or unsupported values fail rather than borrow default rules.

The compiler chooses these policies:

- Search budgets, seeds, candidate pools, group size, and batch width.
- One cycle of serialization between DMA estimates.
- Two logical rows per PACK relayout iteration for model lowering.
- A wait before reusing an eight-slot DMA command-ring entry, an RTL rule.
- The synthetic `fifo43` target: latency 43, one cycle per beat, and four
  accepted cycles in each eleven-cycle period.

DMA costs conservatively bound the selected curve for ordering. Matching
`dma.wait` operations establish completion; an estimate cannot replace them.

## Compiling with the model provider

```sh
python -B tools/lower_atlas_program.py source.mlir --provider model \
  --npu-model PATH --model-facts NEW_DIR --profile curve \
  --search-budget full --output-dir OUT
```

Omitting `--model-facts` regenerates facts under the output directory.
`--profile curve|fifo43` changes scoring memory behavior. `--guarded-pack`
converts proven-finite PACK inputs through MXU0 for exact results.
`--model-region FILE` supplies a DRAM address, initial bytes, and optional
expected bytes; repeat it to check outputs, inputs, and guards during scoring.

The output contains `selection.json`, `program.hex`, `program.json`,
`allocation.json`, and `final.mlir`, plus `commands.txt`, `log.txt`, and each
attempted candidate's files. The selected words are exactly the scored words.
The compiler hashes the selected facts file and checks it again on reload.
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
changes the group limit. Refusals consume attempts; native scoring faults
stop the search. Exhausting legal neighbors may end it before the limit.
`--search-budget single` tries exactly the default candidate, with no MREG
or group search. Both modes use fixed seeds and index-based ties, so worker
count and machine load do not change the candidate order or winner.

## Updating for NPU Model changes

Supported numerical changes regenerate through the same extractors, such as
latency literals and positive curve costs. Unsupported helper structure,
ISA changes, or geometry outside compiler limits refuse with a diagnostic.
For example: `unsupported semantics: hardware/core.py: module contract changed`.

When refusal fires, inspect the named file. Update the matching
`tools/atlas_derive_*.py` interpretation and the C++ reader if their supported
representation changes. Review the source diff, run `--accept-source` to
update the pins, regenerate, run full CTest, and
rerun selections with exact numerical checks before using the new facts.
See [the extractor](../tools/atlas_model_facts.py) and
[the loader](../lib/AtlasModelTables.cpp) for the implementation.

## Limits

npu_model is a software model, so its cycles are not RTL measurements.
Isolated native event checks do not test arbitrary concurrent programs.
Curve estimates ignore address effects, and close rankings can reverse on
the SoC. Exactness depends on the chosen numerical policy and input regions.
