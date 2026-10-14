# rtl_extract: timing from CIRCT HW IR

`rtl_extract` executes HW IR control logic cycle by cycle to derive operation events, release ages and issue spacing. Module names, ports, instructions and stimulus belong in specs. See the [provider overview](../../docs/rtl-timing/README.md) for compiler guarantees, assumptions, limits and artifact reproduction, and the [evidence map](../../docs/rtl-timing/verification.md) for the checks behind them.

## Build and run

The exporter is a standalone CIRCT build:

```sh
P=/path/to/circt-install
cmake -S tools/rtl_extract/export -B build/rtl-extract/exporter \
  -DCIRCT_DIR="$P/lib/cmake/circt" -DMLIR_DIR="$P/lib/cmake/mlir" -DLLVM_DIR="$P/lib/cmake/llvm"
cmake --build build/rtl-extract/exporter -j8
```

Use debug HW IR with named values and retained on-chip memories; see [Reproduce](../../docs/rtl-timing/README.md#reproduce). From the repository root:

```sh
PYTHONPATH=tools python3 -m rtl_extract --hw-ir ee290.debug.hw.mlir \
  --exporter build/rtl-extract/exporter/hw_ir_export --engines xlu \
  --output build/rtl-extract/out/xlu.json
```

Omit `--engines` to extract every engine. Missing control dependencies or unsupported operations produce unresolved blocks; the CLI exits 1 if any block is unresolved. `python3` needs PyYAML; use the prepared `VPY` interpreter for tests.

## Implementation and specs

The [exporter](export/CMakeLists.txt) produces JSON; [ir.py](ir.py) loads requested modules and their instance closure. [control.py](control.py) slices control cones at registers and executes exact finite-width operations with unknown propagation. [derive.py](derive.py) resolves bundle ports, memory latency and command codes; [runner.py](runner.py) executes recipes; [summaries.py](summaries.py) reduces traces; [facts.py](facts.py) writes timing blocks.

[spec.py](spec.py) validates YAML keys and loads specs through `read_spec(path)`. The six engine specs, engine pair specs and MXU engine chains remain YAML files. AtlasCore system specs, core pair enumerations and core chains are generated on load by [gen_core.py](specs/gen_core.py); their logical YAML names still work with the sweep CLI. To inspect one:

```sh
python3 tools/rtl_extract/specs/gen_core.py pairs/core_vpu.yaml \
  --output build/rtl-extract/specs/pairs/core_vpu.yaml
```

Generator choices are listed by `--help`. Generated expectations are regression pins, not inputs to extraction.

## Timing output

The document is `{schema: "merlin.op_timing.v1", hw_ir: {path, sha256}, op_timing: [...]}`. Each block names `<engine>.<op>` or `<engine>.<op>/<variant>` and carries module, engine, operation, optional variant, `source: control_simulation`, evidence, assumptions, events and `first_free_age`. `next_issue_age` appears for engines declaring it. The JSON has no status field: unresolved blocks have null timing values and an explanation in `evidence`. The document pins the input IR, not the exporter or specs.

Events summarize first/last ages, count, spacing, row order and optional split streams. Response latencies appear in `assumptions.scratchpad_read_latency`; RTL-derived latencies also carry `scratchpad_read_derivation`. Unknown values are never replaced by estimates.

## Single-circuit schema

See [spec.py](spec.py) for validation and [vlsu.yaml](specs/vlsu.yaml) for an engine example. A run resets and flushes before age 0. Control-cone registers must then be known except declared `unreset` registers; control signals must be known every age, and fields whenever their event is valid.

- Required: `engine`, `module`, `events`. `inputs` maps approved control inputs to idle integer values. `command: PREFIX` approves `PREFIX_valid` and `PREFIX_bits_*` input ports, defaulting them to zero.
- `events.GROUP` uses `valid: SIGNAL` or `bundle: PREFIX`. Bundles derive valid, approve an input ready port and include fields supported by the control cone; explicit `fields: {NAME: SIGNAL}` overrides this expansion. `row` reports row contiguity; `split_by` separates streams.
- `response: {input, memory?, latency?}` pulses the response input after each event. `memory` requires `system`: exactly one engine and memory instance must be siblings with unique request/response wiring. The memory's measured response latency must not precede its consistent `seq.firmem` read latency. Explicit `latency` overrides the measured value and is recorded as `used`.
- `decode: {input, instruction: {instance, port}, words: {NAME: WORD}}` requires `system`. Each word drives the RTL decoder; the resulting codes must be known, distinct and different from the all-zero word's code. Commands reference them as `$NAME`.
- `operations.OP.commands` maps an age or inclusive `"A..B"` range to input overlays. Optional `next_issue: {signal, bit?}` identifies an issue guard. `operation_table: {codes?, template}` expands a template's `$code`; without `codes`, it uses `decode.words`.
- `busy` is a signal or list of signals combined by OR. It must form a contiguous interval and drain; `first_free_age` is the first free age after the command.
- `probes: {NAME: SELECTOR}` exposes hierarchical values; `cuts: {INPUT: SELECTOR}` substitutes approved inputs. Selectors use `{instance, result}` or `{path, value}`; values match `name`, then `sv.namehint`.
- `unreset` selects registers by `{path, name}` or `{path, match}` (regex). A selector must match registers in the control cone that remain unknown after reset/flush. A coupled partner declares its own selectors.
- `responders.NAME` declares request/response handshakes, positive `latency` and an optional repeating `accept` pattern. Requests are answered in order; accepted request and response handshakes become events. Request fields and optional tags are checked for known values.
- `invariants: {exclude: [...]}` enables RTL assertion checking. Firing or unknown conditions fail. Each exclusion needs a reason and must match an assertion. The exporter supplies assertions with `--assertions`; their clocks must trace to the top clock.
- `variants.NAME` deep-merges an overlay, optionally selecting `only: [OPS]`; identity, derivation and regression keys cannot be overridden. `checks: [{equal: [PATHS], reason?}]` marks compared records unresolved on a mismatch. `expect` is read by tests, never by extraction.
- Defaults: `clock: clock`, `reset: reset`, `reset_cycles: 16`, `flush_cycles: 16`, `limit: 320`, `tail: 2`.

## Coupled circuits

`recipe: coupled` adds `partner: {module, inputs, cuts?, probes?, unreset?}` and `links: {INPUT: OUTPUT}`. Partner inputs name entries in the top-level `inputs`. [coupled.py](coupled.py) starts linked inputs unknown each cycle and evaluates both circuits until links settle before committing state. Unknown links stay unknown; links that do not settle fail. Every observed signal must belong to exactly one circuit. [scalar_lsu.yaml](specs/scalar_lsu.yaml) is the ScalarCore/LSU example.

## Pair and chain sweeps

[pairs.py](pairs.py) loads `pairs/NAME.yaml`: `engine`, named `pairs` with `first`, `second`, positive `gaps`, optional operand overlays (`first_inputs`, `second_inputs`), `guards`, compiler aliases (`as`) and operand `relation`. File-level `equivalent` combines event groups; `ignore` omits level events. `invariants` enables assertions; `shared` declares allowed shared control inputs or instance paths.

A trial issues the second operation at the selected gap. Acceptance requires known control, drain, clear issue guards and the exact event multiset of both isolated runs, with the second shifted by the gap. The default guard is the second operation's `next_issue`. Output schema `rtl_extract.op_pairs.v1` retains per-gap results and rejection reasons, `min_gap`, `stable_gap` (the smallest gap whose remaining swept gaps all pass), assertion counts and, for disjoint event groups, shared-control reachability.

[sequences.py](sequences.py) loads `sequences/NAME.yaml`: named chains with an `ops` cycle (`op`, optional `inputs`, `guards`, `as`), `lengths` and `gaps`. Position `i` issues at `i * gap`; the same acceptance rules apply to the full chain. `rtl_extract.op_sequences.v1` records include inputs and per-length gaps. `--jobs N` parallelizes isolated runs and trials. Both CLIs take `--hw-ir`, `--exporter` and `--output`; use `--pairs` or `--sequences` to select names. Artifact commands and compiler gap-coverage semantics live in the [overview](../../docs/rtl-timing/README.md).

[pairfacts.py](pairfacts.py) joins assertion-checked sweeps with resolved timing blocks into `merlin.op_pairs.v1`. It pins timing and source documents by SHA-256, checks the hardware IR agrees, validates records against current specs and rejects duplicate `(first, second, relation)` records. Pair records carry both operand overlays, resolved aliases and merged relations; sequence steps carry overlays and resolved aliases, including empty overlays. Missing metadata or a mismatch with the current spec requires regeneration. Each source records canonical digests of its sweep and engine specs. Pair facts retain the swept range and stable gap. Capacity is the maximum observed live count with a witness, using `next_issue_age` where declared or the last event inclusive otherwise; all blocks of an engine must use the same window convention. Per-gap detail stays in sweeps.

## Extend and check

Add an engine spec with approved inputs, events and commands; encode alternatives in variants and equalities in `checks`. New control operations need exact semantics plus a mutation or rejection test in [test_rtl_extract_synthetic.py](../../test/test_rtl_extract_synthetic.py). Add hardware regressions under `EngineCase` in [test_rtl_extract_hw.py](../../test/test_rtl_extract_hw.py). Pair/chain behavior, pair facts and generated-spec pins are checked by [test_rtl_extract_pairs.py](../../test/test_rtl_extract_pairs.py).

`CompilerCrossCheck` in the hardware test compares mapped RTL quantities with compiler footprints through the CMake target `rtl-timing-facts-probe` in the build's `bin`. [rtl-timing-facts-map.yaml](../../test/rtl-timing-facts-map.yaml) declares equality or conservative bounds and unmodeled records/groups; every record and nonempty group needs coverage. `ATLAS_OOT_BIN_DIR` selects the binary directory; `ATLAS_TIMING_PROBE` overrides it. `ATLAS_OP_TIMING` selects facts; otherwise `ATLAS_HW_IR` and `ATLAS_HW_EXPORTER` derive them.

```sh
PYTHONPATH=tools:test "$VPY" -m unittest test_rtl_extract_hw.CompilerCrossCheck
```

`tools/regenerate-rtl-timing.py` rebuilds the complete artifact inventory with cached stages, reference checks and `env.sh`; see [Reproduce](../../docs/rtl-timing/README.md#reproduce).
