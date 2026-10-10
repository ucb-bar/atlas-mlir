# rtl_extract: engine timing from CIRCT HW IR

Computes per-operation engine timing (event ages, release, next issue) by executing the control logic of CIRCT HW IR cycle by cycle. Python stays target-agnostic; every module, port, opcode and stimulus lives in `targets/<target>/<engine>.yaml`. The output is a `merlin.op_timing.v1` document that drops into Merlin's RTL facts as the `op_timing` section (see Output). The [overview](../../docs/rtl-timing/README.md) covers the pipeline, scope and results; [verification](../../docs/rtl-timing/verification.md) the checks.

## Layout

| Path | Role |
| --- | --- |
| `export/` | `hw_ir_export`: C++ CIRCT exporter (HW IR → JSON). Standalone CMake; the compiler build never depends on CIRCT. |
| `ir.py` | `load_document(hw_ir, modules, exporter)`: exports the modules plus their instance closure. The only code that knows about the exporter. |
| `control.py` | `ControlCircuit`: register-boundary slicing and exact finite-width execution with unknown (`None`) propagation. Fails closed on unapproved inputs, ops outside the whitelist, enabled/async registers, and foreign clocks/resets. |
| `spec.py` | `load_spec`, `load_target`, `variant`: YAML loading and validation; unknown or missing keys fail. |
| `derive.py` | Derives bundle ports, memory response latency and decoded command codes from the IR before a circuit is built. |
| `runner.py` | `extract(document, spec)`: recipe registry (`RECIPES`), the generic `single` recipe, summaries, spec checks. |
| `coupled.py` | `coupled` recipe: two circuits clocked together; same-cycle links resolved by fixed-point iteration over unknowns. |
| `summaries.py` | Stream summaries (first/last age, count, step, row contiguity, split streams), first-free and next-issue ages. |
| `facts.py` | Record shape, `block` (record to `op_timing` block) and `document` (the output). Unresolved blocks carry `null` values and the reason as `evidence`. |

## Output

`tools/extract-rtl-timing.py` writes `{schema: "merlin.op_timing.v1", hw_ir: {path, sha256}, op_timing: [block, ...]}`. The document names only the IR it read; the target, spec set and exporter are not recorded.

`op_timing` is a named-block list keyed by `name` (`<engine>.<op>` or `<engine>.<op>/<variant>`), the shape Merlin merges per block beside its per-module `timing` section. A block has `name`, `module`, `engine`, `operation`, `variant` (only for variants), `source` (`control_simulation`), `evidence`, `assumptions`, `events`, `first_free_age` and, for engines that declare one, `next_issue_age`. `null` means unknown, never a guess: an unresolved block has `events`, `first_free_age` and `next_issue_age` all `null` and its reason in `evidence`. There is no `status` field; a block is resolved when `events` is not `null`, and the CLI exits 1 otherwise.

## Build and run

```sh
P=/path/to/circt-install   # needs lib/cmake/{circt,mlir,llvm}
cmake -S tools/rtl_extract/export -B build/rtl-extract/exporter -G "Unix Makefiles" \
  -DCIRCT_DIR=$P/lib/cmake/circt -DMLIR_DIR=$P/lib/cmake/mlir -DLLVM_DIR=$P/lib/cmake/llvm
cmake --build build/rtl-extract/exporter -j8
python3 tools/extract-rtl-timing.py --hw-ir ee290.debug.hw.mlir --exporter build/rtl-extract/exporter/hw_ir_export \
  --target atlas --engines xlu --output build/rtl-extract/out/xlu.json
python3 -m unittest discover -s test -p 'test_rtl_extract_*.py'   # hardware tests need ATLAS_HW_IR, ATLAS_HW_EXPORTER
```

Run on a debug lowering of the selected FIRRTL (`-O=debug --preserve-values=named`, no `--repl-seq-mem`; see the [overview](../../docs/rtl-timing/README.md#reproduce)). The default lowering removes dead ports and wire names, and memory replacement turns on-chip SRAMs into external blackboxes; the extractor then reports `unresolved` rather than substituting proxies. The CLI exits 1 if any record is unresolved.

Responses are fed back after `events.<g>.response.latency` ages and reported as `assumptions.scratchpad_read_latency`. This is the on-chip VMEM/MREG read latency, fixed by the RTL: the vector and scalar LSU are always granted ahead of DMA and TileLink. External memory and DMA completion are not modeled; they are variable and governed by `dma.wait`, not by a cycle count.

## Spec schema (`single` recipe)

- Required: `engine` (file stem), `module` (top), `events`. `inputs` lists approved control inputs that are not derived, and idle-value overrides; derived inputs idle at 0.
- `command: <prefix>`: approves `<prefix>_valid` and every `<prefix>_bits_*` input port of `module` (idle 0 unless listed in `inputs`).
- `events.<group>`: `valid` (signal) or `bundle: <prefix>` (valid `<prefix>_valid`, an input `<prefix>_ready` is approved, fields are the `<prefix>_bits_*` outputs named by suffix that depend only on approved inputs through supported operations; payload bits drop out); optional explicit `fields` (`{name: signal}`, overrides the expansion), `row` (field checked for 0..n-1 order), `split_by` (field that splits the group into `streams`), `response: {input, memory?, latency?}` (the input, idle 0, pulses `latency` ages after each valid).
- `system`: module instantiating the engine, its memories and the instruction decoder; required by `memory` and `decode`, and loaded with the engine.
- `response.memory: <module>`: derives `latency`. The single `system` instance of that module must be a sibling of the single engine instance, with the event valid wired to exactly one memory input and the response input driven by exactly one memory output. That memory is simulated alone (other inputs idle 0): the ages from a one-cycle request pulse to response valid are the latency. Every `seq.firmem` `readLatency` in it must agree and not exceed the measurement; an extra register makes the measurement win. An explicit `latency` (e.g. in a variant) overrides it. Records report `assumptions.scratchpad_read_derivation.<group>` (memory instance, ports, `measured`, `memory_read_latency`, `used`).
- `decode: {input, instruction: {instance, port}, words: {name: word}}`: command codes come from the RTL. Each declared ISA word (names and encodings are vocabulary, matching `lib/AtlasEncoding.cpp`) drives the decoder input `instruction` (instance path in `system`; its driver must be a named value or an instance result), and the code is the value reaching `input` (a port of the engine instance or a cut) in the same cycle. Codes must be known, distinct, and differ from the all-zero word's. Commands refer to codes as `"$name"`.
- `operations.<op>.commands`: `{age | "a..b": {input: value}}` overlaid on the idle inputs for those ages; optional `next_issue: {signal, bit?}`. `operation_table: {codes?: {op: code}, template: {...}}` generates operations, replacing `"$code"`; without `codes` it generates one operation per `decode.words` name.
- `busy`: signal; gives `first_free_age` (first age after the first command with busy low) and requires a contiguous busy interval.
- `probes: {name: selector}` exposes values inside the hierarchy as observable signals; `cuts: {input: selector}` replaces a value by an input (declare its idle value in `inputs`). Selectors: `{instance: "a/b", result: port}` or `{path: "a/b", value: name}` (matches `name`, then `sv.namehint`).
- `unreset`: optional list of `{path, name}` register selectors (`path` is the instance path, `""` for the top; a coupled `partner` takes its own `unreset`) that may stay unknown after reset/flush. Every other register in the control cone must be known; an entry that matches no register, lies outside the cone, or is already known after reset/flush fails the run. Control outputs and valid event fields must still be known at every age.
- Defaults: `clock: clock`, `reset: reset`, `reset_cycles: 16`, `flush_cycles: 16`, `limit: 320`, `tail: 2`.
- `variants.<name>`: deep-merged overlay of the spec (not `engine`, `module`, `checks`, `expect`, `operation_table`, `system`, `command`, `decode`); optional `only: [ops]`. Records are named `<engine>.<op>` and `<engine>.<op>/<variant>`.
- `checks: [{equal: ["op.events.write", "op/variant.events.write"], reason?}]`: a failure marks the named records unresolved.
- `expect`: regression values per record, read only by `test/test_rtl_extract_hw.py`. Never feed them into extraction.

A run resets, flushes, then requires every register in the cone of the control signals (`busy`, event `valid`s, `next_issue` signals) to be known, except those listed in `unreset`; fields must be known whenever their event is valid. Ages are absolute; age 0 is the first cycle after the flush.

## `coupled` recipe

`recipe: coupled` simulates `module` (primary, with `cuts`/`probes`) and `partner: {module, inputs: [names], cuts?, probes?}` in lockstep; `partner.inputs` names the top-level `inputs` that belong to the partner. `links: {input: output}` drives an input port of one circuit with the same-cycle output (port or probe) of the other. Each cycle starts the linked inputs unknown and re-evaluates both circuits until the links stop changing, so evaluation order never matters and a combinational loop between the circuits leaves values unknown (unresolved). Every observed signal must be an output of exactly one circuit; the rest of the schema is that of `single`. Example: `targets/atlas/scalar_lsu.yaml`.

## How to add an engine

1. Write `targets/atlas/<engine>.yaml` with the debug IR open: list the approved inputs, events, responses and commands. Add `expect:` only from values you have checked independently, labelled as regressions.
2. Run the CLI for the engine. `Timing depends on unapproved input` means a payload reaches control (or an input is missing from `inputs`); `Unsupported timing operation` means the cone hit an op outside the whitelist; extend `ControlCircuit.allowed` only with exact semantics and a synthetic test.
3. Engine-specific invariants (families sharing a write age, overwrite vs accumulate equality, overlap neutrality) go in `checks:` or in variants that encode the alternative stimulus, not in Python.
4. If one circuit is not enough, add a generic recipe: register `RECIPES[name] = (build, run)` in `runner.py` and its extra spec keys in `spec.RECIPE_KEYS[name]`; extend `runner.modules(spec)` if it simulates more than `module`. `run` must return the per-age trace shape of `run_single` (`age`, `busy`, `signals`, `events`) so summaries and facts stay shared. Wiring between circuits belongs in YAML.
5. Add a class deriving `EngineCase` in `test/test_rtl_extract_hw.py`, ideally with one mutation (an extra register or a payload-dependent control path) that must move or reject the result.
6. If the compiler models the operation, map it in `test/rtl-timing-facts-map.yaml` so the cross-check below covers it; otherwise list it there as not modeled.

## Compiler cross-check

`test/test_rtl_extract_compiler.py` compares the compiler's built-in footprints (`lib/AtlasTiming.cpp`, through `test/rtl-timing-facts-probe.cpp`) with the computed facts and fails on any difference. `test/rtl-timing-facts-map.yaml` maps each record to a compiler mnemonic and states the conventions (inclusive hold ends, so `first_free_age` is the hold end plus one). Every computed record must be compared or listed as not modeled. The compiler's behavior is unchanged; this only detects drift between its constants and the RTL.

```sh
L=/path/to/llvm   # LLVM source include/ and build include/ directories
c++ -std=c++17 -O2 -I include -I $L/llvm/include -I $L/build/include \
  test/rtl-timing-facts-probe.cpp lib/AtlasTiming.cpp -o build/rtl-extract/probe/rtl-timing-facts-probe
ATLAS_TIMING_PROBE=build/rtl-extract/probe/rtl-timing-facts-probe ATLAS_OP_TIMING=build/rtl-extract/out/all.json \
  python3 -m unittest discover -s test -p 'test_rtl_extract_compiler.py'
```
