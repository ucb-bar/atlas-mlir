# Bounded EE290 VLS boundary observations

[`observe-ee290-vls.py`](../../tools/observe-ee290-vls.py) prepares a fixed signal selection and decodes ordinary VCD for the selected `EE290SimConfig` witness. Full EE290 system VCD capture and selected UCLI syntax remain unvalidated; selected AtlasCore Verilator replays described below have passed. The boundary decoder always emits `scheduling_qualified = false`, `physical_sram_arbitration_verified = false` and `capture_execution_link_verified = false`. A passing decoder result cannot enable qualified compiler rules.

The first slice observes the scalar/frontend and LSU boundaries: scalar `s1_fire` and `is_lsu_launch`; LSU command operands and idle-state capture; VMEM and MREG request/response ports; busy/release; DBG0 CSR write and synchronous publication; and ECALL/halt. The fixed mapping uses generated module-relative names from `ScalarCore`, `LSU` and `CSRFile_1`. There is no user-authored remapping or arbitrary qualification flag. Preparing a plan verifies the selected successful witness's manifest, hardware IR, simulator, program and those three generated-source identities, checks the manifest/HW crosslink, and checks that source files contain the selected declarations. These structural checks do not automatically prove the mapping's semantic interpretation or the complete simulator build chain; see [build provenance](build-provenance.md).

## Prepare and capture

Select a successful numerical witness for the exact simulator to be observed. A simulator rebuilt for `-debug_access+all` needs its own passing witness; changing the simulator path in an older receipt is insufficient. Obtain the AtlasCore instance scope from that build's selected hierarchy metadata and a bounded visibility probe. The exact scope is local metadata, not inferred from the configuration name.

```sh
python3 tools/observe-ee290-vls.py prepare \
  --witness-report "$WITNESS" \
  --expected-witness-sha256 "$WITNESS_SHA256" \
  --atlas-scope "$ATLAS_SCOPE" \
  --output build/rtl-timing/observation-plan
```

The new ignored directory contains `plan.json`, a fixed `signals.txt`, input/producer snapshots, and `capture-candidate.ucli`. The candidate uses `dump -file ... -type VCD`, individual `dump -add ... -depth 0` commands, `run` and `quit`. **These command flags are unvalidated.** The plan records `capture_command_validated = false` and `capture_candidate.syntax_validated = false`; it does not execute a command. The selected simulator's current UCLI help or a separately recorded bounded licensed probe must confirm syntax and visibility before an actual capture. An older Virsim `dump` manual is insufficient to establish current UCLI syntax. Ordinary waveform plusargs alone do not establish dumping in a non-DEBUG TestDriver.

After that check, the existing [witness runner](../../tools/run-ee290-vls-witness.py) can retain numerical checks while forwarding separately selected simulator arguments, for example `--sim-arg=-ucli --sim-arg=-do --sim-arg="$UCLI_SCRIPT"`. This example does not certify those flags. Preserve the exact executed script, simulator, program, numerical receipt, stdout/stderr and produced trace identities. The present decoder records the selected trace bytes but does not yet validate an execution receipt linking their production to that simulator invocation.

## Decode the finite trace

```sh
python3 tools/observe-ee290-vls.py decode \
  --plan build/rtl-timing/observation-plan/plan.json \
  --expected-plan-sha256 "$PLAN_SHA256" \
  --trace "$VCD" --expected-trace-sha256 "$VCD_SHA256" \
  --expected-panels 3 --max-edges 3000000 \
  --output build/rtl-timing/observation-result
```

The output directory must be new and ignored. A successful report has schema `atlas.ee290_vls_boundary_observation.v0` and state `bounded_boundary_events_passed`; it records command operands, measured request/response/destination edges, release edges, marker publication and terminal endpoints. The prepared plan, decoder and path-guard dependency are retained. The original VCD remains a selected hashed input, without another large copy.

At each rising clock timestamp, the decoder samples values settled at the preceding VCD timestamp. It groups all changes at one timestamp before moving to the next and does not infer delta-cycle ordering from their textual order. This excludes sequential post-edge updates from the command consumed at that edge. Inputs must have settled before the edge; the present projection cannot resolve a setup race at the same timestamp. Missing signals, incompatible widths, unknown required control/active payload values, backwards timestamps, ambiguous clock changes, dump suppression and incomplete panels fail closed. Unknown inactive payloads are permitted. The parser supports ordinary scalar/binary values and whole-vector declarations, with explicit timescale and an edge-count bound.

For each panel, the validator admits one aligned 32-row VLOAD followed by one disjoint VSTORE from the same MREG. Command opcodes must be 1/2, the VMEM line must be a multiple of 32 in `0..49120`, and both engine states and busy outputs must be idle at acceptance. All VLS command gaps must be at least 35 clock edges. Source requests must occur at ages `1..32`, boundary responses at `2..33`, destination writes at `3..34`, and the corresponding busy interval at `1..34`, followed by an observed idle release sample at age 35. Addresses/rows and response-to-destination payload transport must match on every row; each VSTORE MREG response must also equal the corresponding prior VLOAD MREG write. Missing and unsolicited boundary events reject.

DBG0 must initially be clear for a panel. The supported marker is a scalar-fired CSR write with address `0xC10`, op 1 and data 1, after both VLS operations drain. Publication must appear in `reg_dbg0` at the next observed edge. ECALL must follow that publication with DBG0 equal to 1 and the combinational halt signal asserted; illegal-instruction and EBREAK events reject. Quiescent initial host-loading and post-ECALL readback/marker-clear intervals may remain halted, with no scalar firing or LSU/CSR events. Halt deassertion is recorded as such; it is not automatically proof of the host START transaction.

## Qualification boundary

One-cycle response timing is checked at the LSU memory interfaces. These ports do not prove a physical SRAM request won arbitration, what competing engine/host accesses occurred, or every blackbox behavior. The numerical witness remains responsible for output and guard checks. Physical grants and memory ports, competing traffic, execution-to-trace linkage, admission boundaries beyond these finite operand panels and the remaining [conditional scope](selected-evidence.md) must be discharged before qualification. Copying a qualification flag into a plan or receipt cannot change this decoder's status.

Run the focused synthetic regressions with `python3 test/test_ee290_vls_observation.py`. They cover complete streams, pre-edge sampling, initial/between-panel host quiescence, missing/wrong events and payloads, issue/acceptance mismatches, unsupported addresses, early marker/halt, unknown values, malformed VCD, restricted paths, receipt hash mismatches and a changed fixed signal projection. They establish decoder behavior rather than measured hardware timing.

## Selected AtlasCore replay with behavioral SRAMs

[`replay-ee290-atlascore.py`](../../tools/replay-ee290-atlascore.py) supplies a license-free component experiment using the selected AtlasCore generated RTL and the emitted words from a pinned successful EE290 witness. It verifies the complete reviewed 123-definition HW semantic closure, snapshots 119 generated module files, the two additional inline queue RAM files (`ram_128x256`, `ram_32x264`) and the selected memory implementation file: 122 source files total. The latter implements the four external memory definitions in the HW closure. The implicit RAM files have also been independently matched to the selected FIRRTL-to-Verilog derivation. No replacement SRAM behavior or modified Atlas RTL is introduced.

The [C++ host harness](../../test/ee290-atlascore-replay.cpp) drives AtlasCore's IMEM, CSR and VMEM TileLink ports directly. It loads/readbacks the exact emitted instruction words; initializes the same 1,536-word window and three input patterns; checks settled ECALL status 5, DBG0 1 and illegal PC 0; and checks 256 output plus 1,280 preserved words per panel. Host VMEM transactions use full 32-byte lines rather than CPU MMIO word accesses. Handshake/opcode checks are bounded; the harness is not a full TileLink protocol checker. The DMA reply is idle and unexpected DMA requests reject. Verilator's default two-state initialization applies to unused idle payload fields, and valid transactions set all their request fields explicitly.

```sh
python3 tools/replay-ee290-atlascore.py \
  --witness-report "$BASELINE_WITNESS" \
  --expected-witness-sha256 "$BASELINE_WITNESS_SHA256" \
  --verilator "$SELECTED_VERILATOR" \
  --verilator-root "$SELECTED_VERILATOR_RUNTIME" \
  --cxx "$SELECTED_CXX" --make "$SELECTED_MAKE" --ar "$SELECTED_AR" \
  --jobs 4 --timeout-seconds 2400 --max-cycles 200000 \
  --output build/rtl-timing/atlascore-baseline
```

The driver records separate captured compilation and execution phases through the [phase recorder](../../tools/ee290_build_capture.py), selected input stability, tool identities, exact argv/environment, logs/exits and VCD output. Verilator runs only its fresh generated Makefile, with no Chipyard make/SBT traversal. Trace depth 3 exposes frontend/LSU and bank-wrapper SRAM ports. The installed Verilator/C++ runtime support remains an opaque dependency scope rather than a hermetic build closure. Reports use `atlas.selected_atlascore_replay.v0` and explicitly set `ee290_system_execution_verified = false`, `build_linkage_complete = false` and `scheduling_qualified = false`.

For a second pinned witness, add `--reuse-compile build/rtl-timing/atlascore-baseline/report.json --expected-reuse-sha256 "$BASELINE_REPLAY_SHA256"` and select a new output directory. Reuse requires a successful captured compile, unchanged RTL/harness/tool bytes, every unchanged compiler flag and the exact environment. Only byte-identical snapshot input locations and the fresh output path are normalized. The executed model must match a captured compile output. New program words are then captured as separate execution inputs; changing the harness, flags, tools or RTL requires another compile.

The conservative baseline and scheduled streams have both passed real AtlasCore execution, numerical/guard checks and boundary decoding with the same compiled model. The baseline's measured VLOAD-to-VSTORE command gap was 257; the scheduled gap was exactly 35. In all three panels per arm, requests occurred at ages `1..32`, responses at `2..33`, writes at `3..34`, and release at 35. Marker publication and ECALL/halt followed the drained store. These are finite observed operands (VMEM bank 0, source line 32, destination line 96, MREG 4), not qualification of all possible banks/operands or the complete EE290 system.

[`check-atlascore-sram-observation.py`](../../tools/check-atlascore-sram-observation.py) independently verifies the compiled-model-to-executed-tool identity, captured program/trace argv and outputs, settled numerical records, and recomputed boundary events. It derives a fixed physical projection from those measured operands and uses a private instance of the same pre-timestamp VCD sampler. It checks all six VMEM and 32 MREG physical bank enable patterns during command windows; selected SRAM addresses, mode and full byte masks; SRAM response data; MREG high-half rows; and physical write payloads against the LSU boundaries. The baseline and scheduled arm have both passed these checks. The report state is `selected_sram_port_events_passed`, with `selected_model_port_events_verified = true` and `scheduling_qualified = false`.

```sh
python3 tools/check-atlascore-sram-observation.py \
  --replay-report build/rtl-timing/atlascore-baseline/report.json \
  --expected-report-sha256 "$BASELINE_REPLAY_SHA256" \
  --output build/rtl-timing/atlascore-baseline-sram
```

The bank-wrapper wires directly connect the selected behavioral SRAM models. Matching enable patterns do not exclude a losing/suppressed same-bank request, qualify a physical macro/PVT corner, or establish full EE290 CPU/system arbitration. The component host, two-state simulator and bounded operand panels are a narrower experiment than the integrated witness. Keep that boundary explicit when consuming these observations.

Run `python3 test/test_atlascore_replay.py` and `python3 test/test_atlascore_sram_observation.py` for compile-reuse rejection checks and synthetic physical-port mutations. Together with the boundary suite, these checks cover changed hashes/inputs, failed or unstable captures, changed compilation flags/environment, changed/uncaptured executables, competing granted bank events, wrong SRAM addresses/masks/data and private projection isolation. Real replay receipts remain the evidence of actual model execution.
