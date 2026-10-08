# Capturing selected EE290 build inputs

These tools record new build executions for the public `EE290SimConfig` target. They complement the [existing receipt checker](build-provenance.md): checking an old source list cannot establish which bytes built an old simulator. Every output directory below must be fresh and under this checkout's ignored `build/rtl-timing/` directory.

There are separate source-elaboration and simulator-build steps. A successful step records its own inputs, command, logs and outputs. Connecting their products to the selected CIRCT hardware and executed program requires explicit correspondence checks. Neither step qualifies instruction timing.

## Selected source elaboration

[`elaborate-ee290-source.py`](../../tools/elaborate-ee290-source.py) compiles a selected source overlay, then elaborates it ahead of a pinned Chipyard dependency JAR. It invokes Java and Scala directly. The current recipe admits 69 Atlas main Scala files, 11 Atlas Chipyard integration files and these three Chipyard files:

- `generators/chipyard/src/main/scala/config/EE290Configs.scala`
- `generators/chipyard/src/main/scala/config/AbstractConfig.scala`
- `generators/chipyard/src/main/scala/iobinders/IOBinders.scala`

Prepare an ignored JSON inventory with schema `atlas.ee290_source_build_inputs.v0`. Its `sources` array contains `{ "relative_path": "generators/.../File.scala", "identity": { "path": "/absolute/source/File.scala", "sha256": "...", "bytes": 123 } }` records. Its `tools` object supplies identities for `java`, `scala_compiler`, `scala_library`, `scala_reflect`, `chisel_plugin` and `chipyard_jar`. Select compatible Scala/Chisel versions. Pin `espresso` and `dtc` as additional tool roles when required by the selected elaboration; omitted helpers can otherwise fall back to system executables, which are outside the captured helper boundary. Optional `jre_files` can identify selected Java runtime files without implying a complete system-runtime inventory.

Set `source_config` to the actual local `package:Config` constructor and `target_config` to `EE290SimConfig`. Actual local constructor names and paths belong in this ignored inventory, not in upstream examples or compiler target names. `top_module` is `chipyard.harness.TestHarness`; `timeout_seconds` bounds each phase. Descriptive `metadata` can record revisions, local changes and the explicitly selected source scope.

```bash
python3 tools/elaborate-ee290-source.py \
  --inputs /path/to/selected-source-inputs.json \
  --expected-inputs-sha256 INPUTS_SHA256 \
  --output build/rtl-timing/source-build-001
```

The tool copies selected sources, JARs and helper executables, compiles `compile/overlay.jar`, and records FIRRTL, Chisel annotations and JVM class origins under `elaborate/`. The class-origin check requires the selected Atlas, configuration and binder classes to come from the overlay. Generated lambdas must identify an already verified overlay owner. Unexpected file-backed class origins reject the result.

Java receives an explicit environment with output-owned temporary and home directories. The remaining Chipyard classes and resources are a pinned opaque binary dependency. This is a bounded selected-source build; it does not reconstruct that JAR's source history or every Java/native runtime dependency. The successful receipt is `source_elaboration_captured`, with full-Chipyard-source, simulator-link and scheduling-qualification flags false.

## Simulator preparation and compilation

[`build-ee290-simulator.py`](../../tools/build-ee290-simulator.py) starts from an independently selected successful witness receipt and its recorded VCS command. It parses an admitted command recipe without executing saved shell text. Preparation checks source/file-list identities, resolves literal SystemVerilog includes, discovers actual C++ headers, copies selected inputs, then checks dependency discovery again after relocation.

```bash
python3 tools/build-ee290-simulator.py \
  --witness-report /path/to/passing-witness/report.json \
  --expected-witness-sha256 WITNESS_SHA256 \
  --vcs /path/to/vcs/bin/vcs --vcs-version SELECTED_VERSION \
  --cxx /path/to/g++ --cc /path/to/gcc --cxx-version SELECTED_VERSION \
  --cxx-runtime /path/to/libstdc++.so.6 \
  --jobs 8 --timeout-seconds 120 \
  --output build/rtl-timing/simulator-plan-001
```

Preparation runs C++ dependency checks and does not invoke licensed VCS. The receipt records the prepared plan's hash. Tool versions are caller declarations accompanied by executable identities. Review the source set, definitions, library selection, dependency checks and plan before running it.

The explicit C++ runtime matters at both link and execution time. A library built with a newer toolchain can require `GLIBCXX` symbols absent from the link driver's default library; setting `LD_LIBRARY_PATH` alone does not change its implicit link-library selection. The builder snapshots the selected `libstdc++` under its real filename, linker lookup name and runtime SONAME alias, and explicitly links it from the captured library directory.

From an environment with working VCS licensing:

```bash
python3 tools/build-ee290-simulator.py --run \
  --plan build/rtl-timing/simulator-plan-001/plan.json \
  --expected-plan-sha256 PLAN_SHA256 \
  --timeout-seconds 7200 \
  --output build/rtl-timing/simulator-build-001
```

The command creates a separate simulator and build directory. It bounds compilation to at most eight jobs, preserves the selected preprocessing mode, and enables debug access for a later observation attempt. `DEBUG` remains undefined so the test driver does not automatically dump the entire system. Debug access does not itself prove that a particular signal is visible or that a trace was captured.

`phase.json` identifies the new `simv`, runtime archive and generated build files. A successful build must be followed by numerical witnesses using that exact simulator; old passing runs do not transfer to a new executable. New traces likewise require an explicit link to their capture execution. Relevant vendor tools and headers are identified, while the vendor implementation's complete subprocess closure and unrelated system runtime remain outside this recipe's claims.

## Interpretation and validation

The shared capture library records selected input stability before and after each command, explicit environment and tool identities, logs, timeout/exit status and declared output identities. A `phase_completed` record is evidence of the selected command execution, not an automatic assertion that its input selection is complete or that all preceding build links are established. Preserve failed receipts and use new directories for retries.

Focused regressions exercise mismatched identities, restricted paths, failed commands, recipe/selection rejection, dependency relocation, class-origin checks and library alias handling:

```bash
python3 -m unittest discover -s test -p test_ee290_build_capture.py -v
python3 -m unittest discover -s test -p test_ee290_source_elaboration.py -v
python3 -m unittest discover -s test -p test_ee290_simulator_build.py -v
```

These synthetic checks do not replace actual source elaboration, VCS compilation, numerical execution or integrated timing observations. Keep generated sources, CIRCT artifacts, plans, binaries, traces and local configuration labels in ignored output.
