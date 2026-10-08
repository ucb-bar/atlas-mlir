# EE290SimConfig VCS observation capture

[`capture-ee290-vcs-observation.py`](../../tools/capture-ee290-vcs-observation.py) prepares a native VPD observation followed by an explicit VCD conversion using an already captured simulator build and its successful numerical witness. It reuses the exact saved host ELF, program, and emitted-word include, retains both selected receipts, and records the subsequent invocation, executed UCLI input, logs, and produced trace. Preparation does not start VCS. No compiler acceptance, export, provider interface, or qualification policy is changed.

Select each receipt by an independently checked SHA-256. The simulator bytes must match the captured build's simulator output, and the witness's runtime archives must match that build's captured archive outputs. Their current bytes are checked before and after execution. The active capture producer, decoder, and shared path guard must match their prepared snapshots. The selected simulator stays at its original path so its accompanying archive remains available; the ELF and program inputs are copied into the new ignored capture directory. This records the inputs used by observation, without promoting the original build or witness to aggregate accepted evidence.

```sh
python3 tools/capture-ee290-vcs-observation.py prepare \
  --witness-report "$WITNESS_REPORT" \
  --expected-witness-sha256 "$WITNESS_SHA256" \
  --build-phase "$CAPTURED_BUILD_PHASE" \
  --expected-build-phase-sha256 "$BUILD_PHASE_SHA256" \
  --vpd2vcd "$VPD2VCD_EXECUTABLE" \
  --atlas-scope "$ATLAS_SCOPE" \
  --output "$NEW_IGNORED_CAPTURE_DIRECTORY"
```

The output contains `plan.json`, `probe.ucli`, `capture.ucli`, a boundary-observation plan prepared by the existing observer, selected receipt and producer snapshots, and the exact ELF/program/word inputs. It pins the supplied converter executable and saves its `vpd2vcd -full64 native.vpd converted.vcd` argument vector. The native VPD and converted VCD remain separate retained artifacts. Only the new `capture.ucli` is executed for trace capture; the observer's `boundary-plan/capture-candidate.ucli` remains an unexecuted historical-format candidate. The plan saves both simulator argument vectors and working directories. It retains the successful witness's bounded cycle count, deterministic seed, and `-no_save`, then adds `-ucli -do` with the saved input script. The default simulation timeout is 900 seconds; `--timeout-seconds` changes that bounded timeout. The tool accepts the existing witness's explicit invocation shape rather than arbitrary simulator commands.

Run the saved plan from the environment where the selected VCS runtime and license work:

```sh
PLAN_SHA256=$(sha256sum "$NEW_IGNORED_CAPTURE_DIRECTORY/plan.json" | cut -d' ' -f1)
python3 tools/capture-ee290-vcs-observation.py run \
  --plan "$NEW_IGNORED_CAPTURE_DIRECTORY/plan.json" \
  --expected-plan-sha256 "$PLAN_SHA256"
```

The first launch is a 60-second UCLI help probe that never issues a `run` command. It requires a completed sentinel, the relevant dump flags, and no recognized command/license error before starting the trace launch. `--probe-only` stops after that launch. Every invocation consumes its prepared directory: prepare a new directory for a retry or for capture after a probe-only run. Timeout, interruption, and other exceptions terminate the owned simulator process group; a parent exit with remaining descendants is rejected and that group is terminated. Runtime and license configuration are inherited from the launching shell; the receipt records selected runtime variables and whether license configuration exists, without saving license strings.

The trace launch preserves the numerical panel checks and requires the UCLI setup sentinel. It records its output even when numerical execution or decoding fails. A successful trace must declare all 356 fixed signals with their selected widths, and the existing [`observe-ee290-vls.py`](../../tools/observe-ee290-vls.py) must validate three complete VLOAD/VSTORE panels with release, marker publication, and ECALL/halt endpoints. The output `report.json` links the saved simulator invocation and script to `run/trace.vpd` and `simulation.log`, then links the separately recorded converter invocation and `conversion.log` to `run/trace.vcd` and `boundary-observation/report.json`. Conversion has a 300-second bound and fails closed on an error, timeout, surviving descendant, or missing output; the native trace remains retained. An incomplete trace, missing signal, UCLI error (including numeric vendor diagnostics), timeout, or numerical mismatch cannot produce `numerical_and_boundary_observation_passed`.

## UCLI syntax and observed runtime result

The prepared script uses `set atlas_vcd_fid [dump -file ... -type VPD]`, then explicit signal commands `dump -add ... -depth 0 -fid $atlas_vcd_fid`. It wraps setup in Tcl `catch`, prints an explicit failure or success sentinel, and follows `run` with `dump -flush $atlas_vcd_fid` and `quit`. The file ID is necessary: an add command without `-fid` can select VCS's configured VPD/FSDB default instead of the explicitly opened capture file. In UCLI, depth zero means all hierarchy levels when applied to a scope; this script adds individual signals.

The initial licensed attempt confirmed that this simulator's native backend rejects VCD at time zero with `UCLI-DUMP-UNSUPP-FORMAT`, even though the UCLI Tcl front end recognizes that type. It reported VPD and EVCD as supported formats. The repaired script selects VPD explicitly and invokes `vpd2vcd -full64 input.vpd output.vcd`. Synthetic capture/conversion tests and a real Tcl-interpreter regression cover the observed format rejection. Exact emitted sentinel lines distinguish successful setup from VCS echoing the script's unexecuted commands.

The repaired workflow has now completed licensed baseline and scheduled captures against the selected simulator. Both arms passed three numerical/guard panels, exposed all 356 selected signals, retained native and converted traces, and passed the boundary decoder. The observed VLOAD-to-VSTORE gaps were 257 and 35 cycles, respectively, in every panel. In the scheduled program, the marker write occurred on store release and terminal acceptance followed one cycle later. These are finite observations of the bank-0 line-32 to line-96 copy using MREG4; they do not establish arbitrary operands or overlapping traffic. Independent capture-link, physical SRAM and competing-request validation are separate from this producer's successful report.

## Observed projection and scope

The fixed decoded projection contains 39 scalar/frontend, LSU, and CSR signals from `ScalarCore`, `LSU`, and `CSRFile_1`. It covers scalar issue, LSU command acceptance and operands, VMEM/MREG request-response data, destination writes, busy/release, marker publication, and ECALL/halt. Its sampling convention is the existing decoder's values immediately before each rising clock timestamp.

The additional raw projection retains five VMEM request valids for host, scalar, and DMA competition; fourteen MREG request valids for MXU0, MXU1, VPU, and XLU competition; seven read/write SRAM ports for each of six VMEM banks; and eight SRAM ports for each of thirty-two MREG banks. The declaration check establishes visibility only. The existing decoder does not interpret these supplementary values, validate physical SRAM arbitration, or establish quiescence against competing requests. A trace containing wrapper enables alone cannot exclude suppressed competitors. Independent analysis of the supplemental requests, grants, and SRAM behavior remains necessary before accepting those assumptions.

The separate [system memory checker](system-memory-observation.md) now performs that finite supplemental analysis over each continuous observed entry-through-halt envelope. Both recorded captures pass its competing-request and physical SRAM checks, including the baseline's idle gaps. Its result preserves the distinction between observed finite behavior and general operand/environment qualification.

Every result retains `scheduling_qualified = false`. The existing boundary report also retains `physical_sram_arbitration_verified = false` and `capture_execution_link_verified = false`; the new capture receipt records production separately and does not change that report's semantics. Neither a successful numerical run nor a complete capture enables compiler rules. Whole-kernel performance and broader instruction/concurrency domains remain outside this workflow.

Synthetic orchestration and rejection checks run without VCS:

```sh
python3 -m unittest discover -s test -p 'test_ee290_vcs_observation_capture.py' -v
```
