# rtl_extract: timing from CIRCT HW IR

`rtl_extract` simulates HW IR control logic to derive events, release ages and issue spacing. Specs supply module names, ports, instructions and stimulus. The [overview](../../docs/rtl-timing/README.md) describes guarantees and reproduction; the [evidence map](../../docs/rtl-timing/verification.md) links checks.

## Build and run

Build the standalone exporter with a CIRCT installation:

```sh
P=/path/to/circt-install
cmake -S tools/rtl_extract/export -B build/rtl-extract/exporter \
  -DCIRCT_DIR="$P/lib/cmake/circt" -DMLIR_DIR="$P/lib/cmake/mlir" -DLLVM_DIR="$P/lib/cmake/llvm"
cmake --build build/rtl-extract/exporter -j4
```

Use debug HW IR with named values and retained memories. From the repository root:

```sh
PYTHONPATH=tools python3 -m rtl_extract --hw-ir ee290.debug.hw.mlir \
  --exporter build/rtl-extract/exporter/hw_ir_export --engines xlu \
  --output build/rtl-extract/out/xlu.json
```

Omit `--engines` for all engines. Unsupported operations or missing control dependencies produce unresolved blocks and exit 1. Python needs PyYAML; tests use the prepared `VPY` interpreter.

For artifact regeneration, use [regenerate-rtl-timing.py](../regenerate-rtl-timing.py) and its [manifest](specs/regenerate.json); the [reproduction guide](../../docs/rtl-timing/README.md#reproduce) covers arguments, cached stages, reference checks and `env.sh`.

## Specs and output

[spec.py](spec.py) loads flattened engine YAML through `read_spec(path)`. [gen_core.py](specs/gen_core.py) generates MXU engine specs and their pairs/sequences, plus AtlasCore enumerations on load under logical YAML names; generated files are not committed. Inspect one with:

```sh
python3 tools/rtl_extract/specs/gen_core.py pairs/core_vpu.yaml
```

`--help` lists choices; `--output PATH` saves YAML. Expectations pin regressions and never drive extraction.

Output is `{schema: "merlin.op_timing.v1", hw_ir: {path, sha256}, op_timing: [...]}`. Blocks name `<engine>.<op>[/<variant>]` and carry identity, `source: control_simulation`, evidence, assumptions, events, `first_free_age` and optional `next_issue_age`. Unresolved timing values are null, with explanations in `evidence`; JSON has no status field. The document pins IR only.

Events report first/last ages, count, spacing, row order and optional split streams. Assumptions record `scratchpad_read_latency` and measured `scratchpad_read_derivation`. Unknowns remain unknown.

## Single-circuit schema

See [spec.py](spec.py) and [vlsu.yaml](specs/vlsu.yaml). Reset/flush precedes age 0. Control registers must become known except declared `unreset` registers; control signals must always be known, fields whenever valid.

- Required: `engine`, `module`, `events`. `inputs` approves control inputs with integer idle values. `command: PREFIX` approves `PREFIX_valid`/`PREFIX_bits_*`, defaulting to zero.
- `events.GROUP` selects `valid: SIGNAL` or `bundle: PREFIX`. Bundles expand valid, ready and fields; explicit `fields: {NAME: SIGNAL}` overrides fields. `row` checks contiguity; `split_by` separates streams.
- `response: {input, memory?, latency?}` schedules responses. `memory` requires sibling engine/memory instances under `system` with unique wiring and measured latency consistent with `seq.firmem`. Explicit `latency` overrides measurement, recorded as `used`.
- `decode: {input, instruction: {instance, port}, words: {NAME: WORD}}` requires `system`; RTL codes must be known, distinct and differ from zero-word decoding. Commands use `$NAME`.
- `operations.OP.commands` overlays inputs at an age or inclusive `"A..B"` range. `next_issue: {signal, bit?}` declares an issue guard. `operation_table: {codes?, template}` expands `$code`, using `decode.words` without `codes`.
- `busy` ORs selected signals, requiring contiguous activity and drain. `first_free_age` records the first free age after the command.
- `probes: {NAME: SELECTOR}` exposes hierarchy; `cuts: {INPUT: SELECTOR}` substitutes inputs. Selectors use `{instance, result}` or `{path, value}`, matching `name` then `sv.namehint`.
- `unreset` uses `{path, name}` or `{path, match}` regex selectors for control registers remaining unknown after reset/flush. Coupled partners declare separate selectors.
- `responders.NAME` defines handshakes, positive `latency` and repeating `accept`; responses preserve order. Request fields/tags must be known.
- `invariants: {exclude: [...]}` checks assertions; firing/unknown conditions fail. Exclusions require reasons and matches. Exporter `--assertions` includes assertions whose clocks trace to the top clock.
- `variants.NAME` deep-merges overlays with optional `only: [OPS]`; fixed keys cannot change. `checks: [{equal: [PATHS], reason?}]` marks mismatches unresolved. Tests alone read `expect`.
- Defaults: `clock: clock`, `reset: reset`, `reset_cycles: 16`, `flush_cycles: 16`, `limit: 320`, `tail: 2`.

## Coupled circuits

`recipe: coupled` adds `partner: {module, inputs, cuts?, probes?, unreset?}` and `links: {INPUT: OUTPUT}`. Partner inputs reference top-level `inputs`. [coupled.py](coupled.py) settles both circuits before committing state; unknown links remain unknown, unsettled links fail. Observed signals belong to exactly one circuit. See [scalar_lsu.yaml](specs/scalar_lsu.yaml).

## Pair and chain sweeps

[pairs.py](pairs.py) loads `pairs/NAME.yaml`: named `first`/`second` operations, positive `gaps`, operand overlays, `guards`, aliases and `relation`. `equivalent` merges event groups; `ignore` omits level events; `invariants` enables assertions; `shared` permits shared controls/instances.

Acceptance requires known control, drain, clear issue guards and both isolated event multisets, shifted by the issue gap. The default guard is the second operation's `next_issue`. `rtl_extract.op_pairs.v1` stores per-gap results, rejections, `min_gap`, `stable_gap` (all remaining swept gaps pass), assertion counts and disjoint-group shared-control reachability. Records require `first_inputs`, `second_inputs`, resolved `as` and merged `relation`, including empty overlays.

[sequences.py](sequences.py) loads repeating `ops` with `op`, `inputs`, `guards`, `as`, plus `lengths`/`gaps`. Position `i` issues at `i * gap` under the same acceptance rules. `rtl_extract.op_sequences.v1` records require step `inputs`/resolved `as`. Both CLIs accept `--hw-ir`, `--exporter`, `--output`, `--jobs N` and selection via `--pairs`/`--sequences`.

[pairfacts.py](pairfacts.py) joins assertion-checked sweeps and resolved timing into `merlin.op_pairs.v1`, validates metadata against current specs and rejects duplicate `(first, second, relation)` records. It pins matching IR, timing/source documents and canonical sweep/engine spec digests in `sources`. Missing/mismatched metadata requires regeneration. Facts retain resolved operations, relations, swept ranges and stable gaps. Capacity records maximum observed live count with a witness, using `next_issue_age` or inclusive last-event windows consistently per engine.

## Validation

New control operations need exact semantics and mutation/rejection [synthetic tests](../../test/test_rtl_extract_synthetic.py). [Hardware tests](../../test/test_rtl_extract_hw.py) cover regressions and `CompilerCrossCheck`; [pair tests](../../test/test_rtl_extract_pairs.py) cover sweeps, facts and generator pins. The [facts map](../../test/rtl-timing-facts-map.yaml) requires equality, conservative bounds or explicit unmodeled declarations for every record/nonempty group.

`CompilerCrossCheck` uses `rtl-timing-facts-probe` from `ATLAS_OOT_BIN_DIR` (override `ATLAS_TIMING_PROBE`) and `ATLAS_OP_TIMING`, or derives facts from `ATLAS_HW_IR`/`ATLAS_HW_EXPORTER`.
