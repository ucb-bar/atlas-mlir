# Finite EE290SimConfig integrated observation validation

[`check-ee290-vcs-observation.py`](../../tools/check-ee290-vcs-observation.py) validates a selected capture pair without launching a simulator or rebuilding RTL. Select the pair, both plans, and both arm receipts independently by SHA-256. A saved success flag alone is insufficient: the checker reconstructs the admitted command chain, verifies the linked artifact bytes, and recomputes the VCD boundary observations.

```sh
python3 tools/check-ee290-vcs-observation.py \
  --pair "$PAIR/report.json" --expected-pair-sha256 "$PAIR_SHA" \
  --baseline-plan "$PAIR/baseline/plan.json" --expected-baseline-plan-sha256 "$BASELINE_PLAN_SHA" \
  --baseline-capture "$PAIR/baseline/report.json" --expected-baseline-capture-sha256 "$BASELINE_CAPTURE_SHA" \
  --scheduled-plan "$PAIR/scheduled/plan.json" --expected-scheduled-plan-sha256 "$SCHEDULED_PLAN_SHA" \
  --scheduled-capture "$PAIR/scheduled/report.json" --expected-scheduled-capture-sha256 "$SCHEDULED_CAPTURE_SHA" \
  --bounded-applicability "$BOUNDED_PACKET" --expected-bounded-applicability-sha256 "$BOUNDED_SHA" \
  --atlas-emit "$ATLAS_EMIT" --expected-atlas-emit-sha256 "$EMITTER_SHA" \
  --include-system-memory \
  --output "$NEW_IGNORED_VALIDATION_DIRECTORY"
```

The output must be a fresh ignored directory beneath this compiler's `build/rtl-timing`. It retains current validator snapshots, fresh resolver exports and their command/stdout/stderr identities when requested, and the resulting `report.json`. A final recheck of all selected inputs precedes publication, so a changed input cannot leave a success packet. Original capture artifacts and historical receipts are read only.

The pair must contain exactly baseline and scheduled arms. Each arm's independently selected plan and receipt must match the pair's declarations. Its selected numerical witness, captured build phase, simulator, archive libraries, source/file-list inventory, and runtime environment must agree with the retained snapshots and actual commands. Historical producer and converter identities are checked against their executed snapshots; changing a producer's current source path does not substitute new bytes into an old run. Saved phase execution snapshots and stable before/after inputs are checked through the existing bounded-evidence utilities.

The program binding compares the compiler witness, selected MLIR, emitted words, generated host include, and copied ELF. The complete little-endian word array must occur uniquely in the selected ELF. **This is content corroboration with the selected build/source receipt, not a machine-code proof of host behavior.** Host initialization, IMEM installation, and DBG0 ownership/completion policies remain assumptions of the selected fixture source until separately accepted. The checker verifies the recorded emission/host compilation command shapes and successful numerical logs; it does not execute the host or reinterpret the whole RISC-V binary.

The UCLI script must match the fixed native-VPD projection, and the actual probe and simulation commands must consume its selected paths and ELF. Only emitted sentinel lines count; echoed Tcl source cannot fabricate a setup pass. Each arm must retain a native VPD, an explicitly identified converter with its executed-byte snapshot, the exact `-full64 native.vpd converted.vcd` command, a successful conversion log, and the converted VCD. Missing output, failed commands, timeouts, surviving owned descendants, runtime mismatches, or altered trace identities reject the chain.

The checker validates all 356 selected declarations, then recomputes the existing observer's pre-edge scalar/LSU/CSR events for three complete panels. Saved command operands, request/response/write sequences, release edges, marker publication, and terminal acceptance must match that recomputation. Other event streams or incomplete endpoints remain unsupported.

## Optional source, resolver, and memory checks

Selecting a completed, conditional bounded-applicability packet recomputes its selected-source/FIRRTL/HW closure and Verilog content correspondence, rather than copying its status. The referenced LSU replay must also have completed with matching compiler comparisons and no qualification promotion. Compared Atlas module files must occur unambiguously in each integrated build's source inventory. This bridges the selected component evidence to the captured system inputs by content. The unrecorded Verilog-generation execution edge and external behavioral-memory implementations remain explicit limits; this check leaves `source_to_simulation_complete = false`.

Supplying a pinned `atlas-emit` requires that source/HW bridge. The checker invokes a fresh `--rtl-timing-json` export on the exact captured program snapshot, verifies its conditional evidence and final words, and uses the existing shared-resolver event binder. It compares concrete operands, issue gaps, source/destination ages, one-cycle responses, inclusive path holds and next-edge release, marker publication, and terminal acceptance. Each panel's origin is its first VLS issue minus that instruction's exported logical issue cycle; variable host launch time does not change the comparison. The conditional export retains its existing qualification.

`--include-system-memory` calls [`check-ee290-system-memory-observation.py`](../../tools/check-ee290-system-memory-observation.py) to independently analyze the supplemental SRAM/competition projection. Its boundary observations must agree with the aggregate check. The resulting window and physical-memory observations retain that helper's explicit scope, sampling, clock, and off-edge limitations. The aggregate checker does not duplicate its decoder or broaden those limits.

These extensions are optional. The report's `coverage` fields explicitly identify whether the source/HW content bridge, fresh resolver event binding, and system-memory windows were checked. Without the corresponding selections, capture records and boundary events do not imply those checks passed.

## Accepted finite evidence and limits

For the actual selected copy pair, all three baseline panels measure a VLOAD-to-VSTORE issue gap of 257 cycles and all three scheduled panels measure 35 cycles. Both trace/export comparisons preserve the following issue-relative convention:

| Event | Observed age |
| --- | --- |
| Source requests | 1–32 |
| Source read responses | 2–33 |
| Destination writes | 3–34 |
| Path release | 35 |

The scheduled marker is published at the store release edge, followed by terminal acceptance at the next edge. These are finite observations for the captured program words and operands, not qualification of every bank, address, MREG, arithmetic variant, instruction family, or concurrency case. Numerical CSR counts and process wall time are not whole-kernel timing measurements.

Every packet keeps `scheduling_qualified`, `system_qualified`, `operand_domain_qualified`, and `source_to_simulation_complete` false. The checker does not change the compiler selector, scheduler, provider, or export qualification, and it does not enable unsupported operations. Extending the accepted operand/environment domain requires a separate reviewed applicability policy and consumer checks.

Synthetic rejection checks run without a VCS license:

```sh
python3 -m unittest discover -s test -p 'test_ee290_vcs_observation.py' -v
```
