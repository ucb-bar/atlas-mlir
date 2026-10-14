# DMA utilization: scheduling-only experiments

The final experiment preserves the original source and machine loops. Loop unrolling belongs to the higher level and is excluded, following Jeremy's correction. Earlier unrolling measurements are superseded. The selected patch combines O1, O2, O3, and instruction scheduling within the original O4 pack loop. O5 is retained as a rejected experimental candidate; O6's structural changes are outside this scope.

## Final measurements

These are npu_model measurements of the same single-tile sources, with zero runtime errors and ECALL termination in every run.

| Program | Baseline timing | Baseline cycles / DMA utilization | Scheduling-only cycles / DMA utilization |
|---|---|---:|---:|
| Combined attention + MLP | insert-atlas-delays | 8,817 / 40.9% | 6,574 / 54.9% |
| Combined attention + MLP | schedule-atlas-stream | 8,458 / 42.7% | 6,361 / 56.7% |
| Attention | insert-atlas-delays | 4,465 / 57.7% | 4,052 / 63.6% |
| Attention | schedule-atlas-stream | 4,365 / 59.0% | 3,949 / 65.3% |

The new default is stream scheduling. Compared with the original default, combined cycles fall 27.9%, from 8,817 to 6,361. DMA busy time remains 3,609 cycles for combined and 2,577 for attention. Timed IMEM uses 249 words for combined and 127 for attention, below 2,048.

## Agent results

| Approach | Result | Isolated measurement | Final disposition |
|---|---|---|---|
| O1: DMA across pack | Exact pending-transfer identities propagate over CFG edges; bounded execution proves helper VMEM footprints against captured DMA windows. Input/output barriers remain. A follow-up fixes overrejection of completed transfers within source loops. | Combined 7,946 cycles with delays; 7,612 with stream. Attention stream remains 4,365. | Included. |
| O2: DMA queue and costs | Separate frontend command issue from the serial DMA engine. Await dependencies stall issue at transfer completion. Calibrate pack frontend/result timing. | Combined unchanged alone; overlap example 4,918 → 4,782; none of 19 executable examples slows. | Included; pack cost updated for final instruction order. |
| O3: stream timing default | Default the driver to the existing machine stream scheduler; retain explicit insert-atlas-delays selection. | Combined 8,458; attention 4,365. | Included. |
| O4: instruction scheduling in pack | Rotate x16/x17/x15 through loads and stores within each original one-row iteration. | Combined 7,473 / 7,210; attention 4,017 / 3,949 (delay / stream). | Included with no unrolling. |
| O5: expose legacy MXU phases | Expand legacy matmul before scheduling into weight/reset/readout operations. | Combined 8,848 / 8,553, slower than baseline in both timing modes. Also regresses with the measurement-only O1 prototype. | Experimental patch only; excluded from selected changes. |
| O6: overlap tiles | The two-query program/residency experiment changes higher-level program structure. | Retained as a benchmark fixture. | Structural changes excluded; existing-fixture DMA overlap remains scheduling evidence. |

## Original loop structure retained

Every pack still executes 32 iterations, copying eight words per iteration with eight LW and eight SW instructions. The row counter advances by one; both source pointers advance by 16 bytes and the destination by 32. The same branch and delay-slot boundaries remain. The extra temporary x15 was already reserved, so scalar capacity remains nine. The final selected patch contains no unroll factor or two-query program construction.

Combined untimed words remain 195 and attention remains 98, exactly matching baseline. The scope verification (`out/dma/experiments/scope-verification.json`, a local artifact) records instruction counts, loop bounds and pointer strides from emitted IR. The existing buffer checker proves the same word interleave; no buffer or timing proof was relaxed.

The revised pack takes about 1,160 cycles with delay insertion, versus about 1,608 before. Stream instruction scheduling reduces it further to about 1,105. Profile totals include generated helper delays, which the named pack rows alone omit.

## Verification

The final composed checkout passes all 11 CTest targets: 473 Python tests, with 104 optional environment integrations skipped, plus the C++ verification suites. Tests include overlapping and unknown VMEM addresses, nonterminating helper counters, pending DMA launches/waits inside the helper, illegal backedges, legal outer source loops, boundary input/output barriers and random virtual schedules. The O1 agent also reviewed the shared CFG, footprint and timing proof and fixed the source-loop regression before final composition.

All six final simulator runs (the two single-tile programs and the optional two-query benchmark, both timing modes) halt on ECALL with zero runtime errors. The final patch passes git diff --check and git apply --check against the user's current checkout at 106c9bb.

Additional unchanged-workload benchmark: the existing two-query fixture measures 10,822 cycles with delays and 10,314 with stream (5,157 per tile, 50.0% DMA). Its source, weight residency and tile count are supplied by the higher-level fixture and are not transformations in the selected patch. This comparison exercises scheduling of a larger workload; it is not an unrolling or residency optimization claim.

## Independent verification

The selected patch was re-applied to a clean checkout of `106c9bb`, rebuilt,
and checked separately from the agents' runs:

- **Measurements:** all cycle counts above, the O5 numbers and the two-query
  benchmark reproduce exactly with `tools/simulate_atlas_program.py`.
- **Tests:** 11/11 CTest targets pass with `ATLAS_LLVM_BIN` and an
  `--image-base=0` lld wrapper as `ATLAS_LLD`. The Python suite runs 473
  tests (463 at baseline) with the same 104 environment skips as baseline.
- **Same results:** on three random sets of FP8 inputs, baseline and patched
  programs, in both timing modes, write byte-identical outputs for the
  combined tile and the attention tile.
- **No unrolling:** static `lw`/`sw`/branch counts (24/24/3 combined),
  untimed words and the 32-row bound are unchanged, and the LSU executes the
  same 1,549 instructions as baseline.
- **No regressions:** none of the 19 lowerable `test/examples/virtual_*.mlir`
  programs gets slower in either timing mode.

**Open question for hardware:** DMA now writes VMEM while the pack's scalar
and vector accesses use other banks. npu_model and the generated checks
accept this; RTL evidence that AtlasCore does is still needed before an FPGA
run.

## Artifacts

The selected changes (O1 + O2 + O3 + revised O4) are committed on
`feat/combined-attention-mlp-dma-utilization`. The experiment artifacts stay
local in `out/dma/experiments/`, which is gitignored. That directory holds
`scheduling-only.patch`, the per-item `o1.patch`–`o6.patch` and their reports
(O5 rejected, O6 benchmark-only), the stage artifacts and profiles under
`scheduling-only/`, and the CTest logs. The agents' implementation came from
branch `experiment/dma-scheduling` (`bb3a6fd`) in a temporary checkout.

To reproduce:

```sh
ninja -C build
~/.venvs/atlas-mlir/bin/python tools/lower_atlas_program.py \
  test/examples/virtual_attention_mlp_tile.mlir --output-dir out/dma/reproduce-scheduling
~/.venvs/atlas-mlir/bin/python tools/simulate_atlas_program.py \
  out/dma/reproduce-scheduling --profile
```

Model timing remains unvalidated against RTL. The timing simulator initializes DRAM with zero values; buffer contracts and evaluator tests provide separate structural/value checks. No FPGA run was performed.
