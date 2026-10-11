# Verification of RTL timing facts

Four layers check the facts against the compiler and against simulated hardware. None qualifies operations or configurations outside the extracted scope.

## Extractor tests

`test/test_rtl_extract_*.py` cover the interpreter, the coupled recipe and spec validation on synthetic circuits, and (with `ATLAS_HW_IR` and `ATLAS_HW_EXPORTER` set) reproduce every value in the [results table](README.md#results). Mutations such as an extra register stage or a payload-dependent control path must move or reject the result.

## Compiler cross-check

`test/test_rtl_extract_compiler.py` compares each extracted record with the compiler's built-in footprints (`lib/AtlasTiming.cpp`, via `test/rtl-timing-facts-probe.cpp`) and fails on any difference. `test/rtl-timing-facts-map.yaml` maps records to compiler mnemonics and states the conventions; every record is compared or listed as not modeled. This detects drift between the compiler's constants and the RTL. Commands are in the [extractor reference](../../tools/rtl_extract/README.md#compiler-cross-check).

## Selected-mode tests

`test/test_rtl_timing.py` and, for MXU, `test_rtl_mxu_timing.py` run scheduling, delay insertion and `--verify-atlas-rtl-timing` with the facts selected, and check that wrong digests, a missing selection, unsupported domains, early or missing terminals and unsafe streams are rejected, and that admission numbers come from the facts rather than constants. Final verification rechecks timing from the actual instruction stream, also in `atlas-emit` and the LLVM handoff, so rewrites after scheduling cannot silently invalidate it.

## Simulation

**Compiler-scheduled programs on Verilator.** `tools/replay-ee290-compute.py --facts F.json --bin <dir with atlas-opt, atlas-emit> --work DIR (--model VAtlasCore | --rtl <dir of .sv/.v>)` schedules each case (`--cases xlu,vmul,dma_xlu`, default `vmul,dma_xlu`; `--consumer delay|schedule|both`) with the facts selected and `dma=wait`, runs it on AtlasCore, and prints `PASS <case>` or exits nonzero. `tools/check-ee290-compute-export.py` binds the decoded VCD events to `atlas-emit --rtl-timing-json`; DMA completion is anchored to the matching wait, never to a latency. `test/test_ee290_compute_replay.py` runs it when `ATLAS_EE290_COMPUTE_MODEL`, `ATLAS_OOT_BIN_DIR` and `ATLAS_OP_TIMING` are set.

**Issue #11 VCS witness.** `tools/run-ee290-vls-witness.py --program test/examples/ee290_vls_copy.mlir --facts F --atlas-opt PATH --atlas-emit PATH --simulator simv --host-cc <riscv gcc> --output DIR [--schedule delay|schedule]` selects the facts, schedules or inserts delays, verifies, emits, builds the host program and runs `simv +loadmem`; it exits 0 on pass, 1 on a failed or license-unavailable run and 2 on a tool error. It checks outputs and guard memory, not event timing. Earlier Verilator and VCS waveform captures of the VLS copy kernel matched the compiler's predicted events cycle for cycle.

**DMA.** Simulation cannot bound external completion, so the `dma=wait` policy rests on structure: one pending transfer, VMEM exclusive until the matching wait, and verification that every launched transfer is waited before reuse, publication or halt.
