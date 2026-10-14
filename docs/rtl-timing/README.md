# RTL timing for Atlas

The `rtl-op-timing-v1` provider uses timing computed from selected Atlas RTL for scheduling, delay insertion and final verification. A CIRCT HW IR exporter and control simulator produce `merlin.op_timing.v1` facts; the compiler supplies instruction structure and resolves timing on concrete operands. Optional `merlin.op_pairs.v1` facts admit overlap backed by assertion-checked sweeps.

![Chisel and SystemVerilog converge in CIRCT core dialects](figures/chisel_and_sv_circt_convergence.svg)

See the [extractor reference](../../tools/rtl_extract/README.md) for simulation and schemas, and the [verification map](verification.md) for tests and execution evidence.

## The provider

Selection pins facts by SHA-256 and retains `atlas.rtl_evidence`. Pair facts must match the hardware IR digest, pin the selected operation facts, name resolved blocks and report checked assertions. Subsequent loads recheck selection; resolved exports include both digests. [AtlasRTLEvidence.cpp](../../lib/AtlasRTLEvidence.cpp) implements these checks; [test_rtl_timing.py](../../test/test_rtl_timing.py) exercises malformed and mismatched evidence.

Timing passes resolve `provider=<id>`, then retained `atlas.timing_provider`, then `rtl-op-timing-v1` over selected evidence, otherwise `npu-model-rtl-match-v1`. The model is refused over selected evidence. Without selection, the default provider emits exactly what it did before RTL selection existed; the one exception is a block falling through into ECALL, whose drain now ends on a NOP (DELAY then NOP), accepting a previously rejected program. [AtlasStream.cpp](../../lib/AtlasStream.cpp) implements the guard; [test_delay_insertion.py](../../test/test_delay_insertion.py) checks halt padding.

The common verifier checks dependencies, reservations and drained boundaries. Emission and LLVM handoff check selected streams even without timing metadata. `atlas-emit --allow-untimed` permits inspection without a timing claim; `atlas-emit --rtl-timing-json` requires a safe timed selection and exports footprints, block-relative issues, successors and wait-relative epochs. [test_rtl_export.py](../../test/test_rtl_export.py) checks these boundaries.

## Guarantees and assumptions

[AtlasRTLSelection.cpp](../../lib/AtlasRTLSelection.cpp) enforces these rules; [verification](verification.md) maps each to evidence.

- Every reachable block starts with idle engines and no pending DMA, and drains before its successor or ECALL. Branch delay slots contain ADDI or LUI; stalls before redirects or halt end on a NOP. Programs are bounded by 32,768 words and 1,000,000 issue cycles.
- Footprints use exact scalar values at each concrete block entry, following one branch edge when operands are known, otherwise both. Scheduling and verification respect every distinct footprint group. After 4096 block entries, values merge across paths; moving or unknown addresses may be rejected. Operand alignment, range and bank checks apply in every context.
- VLOAD and VSTORE reserve separate paths and overlap subject to VMEM bank and MREG port/release rules. VPU and XLU exclude each other; VLS and either MXU can overlap them under memory rules. Each MXU reserves its own unit. VSTORE protects its source until it drains against writers other than VLOAD. Matmul retains its weight slot through its last accumulator write.
- Without pair facts, VPU commands wait for `next_issue_age`, and commands on one MXU wait for its busy end. With them, every overlapping pair must match its ordered operations, covered gap and operand relation: disjoint or shared physical MREG banks and, where applicable, same or other accumulator and weight slot. Other resource rules still apply. Selected facts cap each VPU/MXU at two live commands; missing coverage falls back to exclusive admission.
- VPU live windows end at `next_issue_age`; MXU windows include their last event and MREG read responses. Hold ends are inclusive: occupation through age 34 permits exclusive issue at 35.

These guarantees assume selected hardware, deasserted reset, quiescent entry, immutable instructions, supplied operand data and no concurrent host access to VMEM or DMA base changes. Competing requesters and alternative memories need review. Selected VMEM/MREG response latency, measured from IR, is one cycle. Extraction assumes data-independent engine control, rejects unknown control outputs and permits only explicitly declared unreset registers. This is timing evidence, not numerical qualification.

### DMA

DMA requires `dma=wait`. Launch captures scalar operands and the global DRAM base; both may then be reused. VMEM/DRAM effects remain live until a matching channel wait. Completion latency is unknown; issue coordinates across waits are lower bounds.

One transfer may be pending per channel. The eight slots use launch count: launch `n` requires launch `n-8` to have been waited. Every transfer needs a wait within its block; markers and halt require completion. Pending VMEM windows must be disjoint from transfers and VLS tiles; scalar reads may share a store window. DRAM spans may overlap only for two reads. Empty, stale and wrong-channel waits fail. These compiler protections cover unguarded RTL channel reuse and address overlap; [test_rtl_engines.py](../../test/test_rtl_engines.py) checks them.

### Gap coverage

Gap coverage uses `stable_gap`, except for a pair with a DMA launch: a rejected gap whose only failure is a moved DMA or DRAM event does not bind. The bound is one past the last other failing gap, or 1 if none; a firing assertion always binds. Two launches bind their order. A VLS compiler bound can therefore be below `stable_gap`. [test_rtl_gap_coverage.py](../../test/test_rtl_gap_coverage.py) compares compiler gaps with this bound, rejects unmapped records and requires unsupported operations to be provider rejections.

## Scheduler

`--schedule-atlas-stream="rtl-best-of=true rtl-dma-estimate=true"` enables ordering estimates under `rtl-op-timing-v1`; both default true and other providers ignore them. The options compare source/list orders and use model DMA estimates in list priorities. They change only the orders tried; the common verifier checks every result and entry context. Estimates do not assign DMA completion latency. [test_rtl_scheduler.py](../../test/test_rtl_scheduler.py) covers these policies.

## Reproduce

Run from the repository root with the prepared environment. Inputs stay read-only; outputs and cache receipts live under the selected build root.

```sh
source build/env.sh
R=${ATLAS_HW_IR%/build/rtl-extract/ir-debug/*}
C=${R%/generators/*}
X=$PWD/build/regen
"$VPY" tools/regenerate-rtl-timing.py --build-root "$X" --python "$VPY" \
  --firrtl "$R/build/rtl-timing/ee290-portable-helper-check/inputs/input.fir" \
  --annotations "$R/build/rtl-timing/ee290-portable-helper-check/inputs/original.anno.json" \
  --lowering-options "$R/build/rtl-timing/ee290-portable-helper-check/inputs/lowering-options.txt" \
  --circt-prefix "$C/.conda-env/riscv-tools" \
  --rtl "$R/build/rtl-timing/atlascore-replay-002/inputs" --bin "$ATLAS_OOT_BIN_DIR" \
  --verilator "$C/.conda-env/bin/verilator_bin" \
  --verilator-root "$C/.conda-env/share/verilator" --jobs 32 \
  --check "$R/build/rtl-extract/out" --reference-hw-ir "$ATLAS_HW_IR"
source "$X/env.sh"
```

The [manifest](../../tools/rtl_extract/specs/regenerate.json) selects thirteen pair sweeps, four sequence sweeps and combined pair facts. The command also lowers and verifies debug HW IR, builds the exporter and Verilator model, extracts operation facts, cross-checks the CMake `rtl-timing-facts-probe` from `--bin`, and verifies 80/111 unique replay passes without/with pairs. Build the probe with the compiler before running. MXU chains validate spacing but stay outside the selected two-command capacity evidence.

Stage keys include commands, tool versions and source/input digests, including generated-spec code. Output digests and replay inventories must also match before a stage skips. `run.json`, `runs/` and `logs/` retain results. An unchanged rerun skips every stage; sweeps can take hours on a first run. At most 32 workers run concurrently.

`--stages lower,exporter,facts,compiler_facts,probe,model,pairs-core_dma` selects a fast subset; `sweeps` selects every sweep. Selection does not run missing prerequisites. `--check-only` compares existing outputs. Checks compare the selected timing artifacts, every gap and every record field, verifying hardware/dependency SHA pins before normalizing source paths and the two FIRRTL hierarchy-output destinations. Old sweeps lack operand metadata and intentionally differ; old pair facts must be regenerated.

Generated specs load through [spec.py](../../tools/rtl_extract/spec.py); `tools/rtl_extract/specs/gen_core.py NAME.yaml --output PATH` exports one for inspection. Expectations are pinned regressions, not extraction inputs. `tools/replay-ee290-compute.py` remains the standalone model/replay entry point.

Compiler selection and resolved export (substitute an untimed physical program):

```sh
"$ATLAS_OOT_BIN_DIR/atlas-opt" prog.mlir \
  --select-atlas-rtl-evidence="op-timing=$X/out/compiler-facts.json op-timing-sha256=$(sha256sum "$X/out/compiler-facts.json" | cut -d' ' -f1) dma=wait" \
  --schedule-atlas-stream --verify-atlas-timing -o "$X/final.mlir"
"$ATLAS_EMIT" --rtl-timing-json "$X/final.mlir" > "$X/resolved.json"
```

Add `op-pairs=$X/out/op-pairs.json op-pairs-sha256=<digest>` for pair selection; use `--insert-atlas-delays` for source-order timing. Virtual programs first run `--verify-atlas-virtual-stream --schedule-atlas-virtual --lower-atlas-virtual-to-machine`; selection survives lowering. Virtual scheduling uses provider latencies and same-unit intervals on canonical operands.

## Not covered

- Runtime-dependent addresses, moving addresses beyond 4096 block-entry contexts, or engine/DMA work crossing blocks.
- Unbounded same-unit overlap: selection caps each unit at two live commands, although some uniformly spaced MXU chains accept three.
- Exact weight-read timing: matmul conservatively owns its weight slot through its last accumulator write.
- DMA completion latency, concurrent host traffic, or memories outside selected assumptions.
- Virtual scheduling models same-unit intervals, not physical cross-engine exclusion; the physical verifier checks the lowered stream.
- A VLOAD write into an MREG bank still being read is not asserted by RTL; compiler release rules protect it.
