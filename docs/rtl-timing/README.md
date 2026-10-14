# RTL timing for Atlas

The `rtl-op-timing-v1` provider schedules, inserts delays and verifies concrete operands using `merlin.op_timing.v1` facts extracted from selected Atlas RTL. Optional `merlin.op_pairs.v1` facts admit assertion-checked overlap.

![Chisel and SystemVerilog converge in CIRCT core dialects](figures/chisel_and_sv_circt_convergence.svg)

See the [extractor reference](../../tools/rtl_extract/README.md) for simulation and schemas, and the [verification map](verification.md) for evidence.

## The provider

Selection retains SHA-256-pinned `atlas.rtl_evidence`. Pair facts must match the hardware digest, pin operation facts, resolve blocks and report checked assertions. Loads recheck selection; resolved exports include both fact digests. [AtlasRTLEvidence.cpp](../../lib/AtlasRTLEvidence.cpp) implements these checks.

Timing passes choose explicit `provider=<id>`, then retained `atlas.timing_provider`, then `rtl-op-timing-v1` over selected evidence, otherwise `npu-model-rtl-match-v1`. The model is refused over selected evidence. Unselected output is unchanged except fallthrough into ECALL: drainage ends on DELAY then NOP, accepting previously rejected programs.

The common verifier checks dependencies, reservations and drained boundaries. Emission and LLVM handoff recheck selected streams without timing metadata. `atlas-emit --allow-untimed` permits inspection; `--rtl-timing-json` requires safe timed selection and exports footprints, block-relative issues, successors and wait-relative epochs.

## Guarantees and assumptions

[AtlasRTLSelection.cpp](../../lib/AtlasRTLSelection.cpp) enforces these rules; [verification](verification.md) maps their tests.

- Reachable blocks enter with idle engines and no pending DMA, and drain before successors or ECALL. Branch slots contain ADDI or LUI; redirect/halt stalls end on NOP. Limits are 32,768 words and 1,000,000 issue cycles.
- Footprints use exact scalars at concrete block entries: known branches select one edge, unknown branches both. Scheduling and verification check every distinct footprint group. After 4096 entries, path values merge; moving or unknown addresses may fail. Alignment, range and bank checks apply in every context.
- VLOAD/VSTORE reserve separate paths under VMEM-bank and MREG-port/release rules. VPU excludes XLU; VLS and either MXU may overlap them under memory rules. MXUs reserve separate units. VSTORE protects its source against writers except VLOAD until drainage. Matmul owns its weight slot through its last accumulator write.
- Without pair facts, VPU waits for `next_issue_age`, each MXU for its busy end. Overlap requires every live pair's ordered operations, gap and operand relation to match: shared/disjoint MREG banks and applicable accumulator/weight slots. Other resource rules remain; each VPU/MXU admits at most two live commands. Missing coverage uses exclusive admission.
- VPU live windows end at `next_issue_age`; MXU windows include the last event and MREG responses. Holds are inclusive: occupation through age 34 permits exclusive issue at 35.

Assumptions: selected hardware, deasserted reset, quiescent entry, immutable instructions, supplied data, no concurrent host VMEM access or DMA-base changes, and selected one-cycle VMEM/MREG responses. Competing requesters or alternative memories require review. Extraction assumes data-independent control, rejects unknown control outputs and allows only declared unreset registers. Numerical correctness requires separate qualification.

### DMA

DMA requires `dma=wait`. Launch captures scalars and the global DRAM base, allowing their reuse. VMEM/DRAM effects persist until the matching channel wait. Completion latency is unknown; coordinates across waits are lower bounds.

One transfer may be pending per channel. Eight slots count launches: launch `n` requires launch `n-8` waited. Transfers need block-local waits; markers and halt require completion. Pending VMEM windows must avoid transfers and VLS tiles; scalar reads may share store windows. DRAM overlap permits only read/read. Empty, stale and wrong-channel waits fail. These checks protect unguarded RTL channel reuse and address overlap.

### Gap coverage

Coverage uses `stable_gap`, except DMA-launch pairs ignore failures caused solely by moved DMA/DRAM events. The bound is one past the last other failure, or 1; assertions always bind. Two launches bind their order. Thus a VLS bound may be below `stable_gap`. [test_rtl_gap_coverage.py](../../test/test_rtl_gap_coverage.py) checks these bounds using recorded overlays, rejects unmapped records and requires provider rejection of unsupported operations.

## Scheduler

`--schedule-atlas-stream="rtl-best-of=true rtl-dma-estimate=true"` compares source/list orders and uses model DMA estimates for list priorities. Both options default true; other providers ignore them. Estimates affect ordering only. The common verifier checks every result and entry context; DMA admission still requires waits.

## Reproduce

From the repository root with the prepared environment:

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
  --verilator-root "$C/.conda-env/share/verilator" --jobs 12 \
  --check "$R/build/rtl-extract/out" --reference-hw-ir "$ATLAS_HW_IR"
source "$X/env.sh"
```

The [manifest](../../tools/rtl_extract/specs/regenerate.json) drives debug lowering, exporter, operation facts, compiler cross-check, thirteen pair/four sequence sweeps, pair facts, Verilator model and 80/111 unique replay passes without/with pairs. Build the compiler's `rtl-timing-facts-probe` target first. Inputs remain read-only; outputs, logs and receipts stay under `--build-root`.

Cache keys pin commands, versions and inputs, including generated-spec code; skipping also requires matching outputs and replay inventories. An unchanged rerun skips every stage. First-run sweeps can take hours. `--stages lower,exporter,facts,compiler_facts,probe,model,pairs-core_dma` selects a subset without supplying missing prerequisites; `sweeps` selects all sweeps. `--check-only` compares existing artifacts. Checks verify dependency/hardware SHA pins and every record field/gap before normalizing source paths and two FIRRTL hierarchy-output destinations. Legacy sweeps missing operand metadata require regeneration.

Generated specs load through [spec.py](../../tools/rtl_extract/spec.py); `tools/rtl_extract/specs/gen_core.py NAME.yaml --output PATH` exports one. Expectations pin regressions, not extraction inputs. [replay-ee290-compute.py](../../tools/replay-ee290-compute.py) runs model/replay alone.

For an untimed physical `prog.mlir`:

```sh
"$ATLAS_OOT_BIN_DIR/atlas-opt" prog.mlir \
  --select-atlas-rtl-evidence="op-timing=$X/out/compiler-facts.json op-timing-sha256=$(sha256sum "$X/out/compiler-facts.json" | cut -d' ' -f1) dma=wait" \
  --schedule-atlas-stream --verify-atlas-timing -o "$X/final.mlir"
"$ATLAS_EMIT" --rtl-timing-json "$X/final.mlir" > "$X/resolved.json"
```

Add `op-pairs=$X/out/op-pairs.json op-pairs-sha256=<digest>` for pairs; `--insert-atlas-delays` preserves source order. Virtual programs first use `--verify-atlas-virtual-stream --schedule-atlas-virtual --lower-atlas-virtual-to-machine`; selection survives lowering. Virtual costs use provider latencies and same-unit intervals on canonical operands.

## Not covered

- Runtime-dependent addresses, moving addresses beyond 4096 entry contexts, or engine/DMA work crossing blocks.
- Unbounded same-unit overlap: selection caps units at two commands despite some accepted three-command MXU chains.
- Exact weight-read timing: matmul owns its weight slot through the last accumulator write.
- DMA completion latency, concurrent host traffic or alternative memories. Measured EE290Sim latency curves ([npu_model](https://github.com/ucb-ee194-tapeout/npu_model/pull/66)) could later recalibrate the scheduler's DMA ordering estimate; this affects ordering only, never admission.
- Virtual cross-engine exclusion; physical verification checks the lowered stream.
- RTL assertions for VLOAD writes into still-read MREG banks; compiler release rules protect them.
