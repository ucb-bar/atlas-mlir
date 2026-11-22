# Quantized model efficiency worklist

Continue the dispatch and traffic reduction work on `perf/quantized-model-efficiency`
in Atlas MLIR and Merlin. Starting revisions: Atlas `0ef85dbc154868c3f72b155adfc9170ee52b4494`
and Merlin `4a039544d309afbec4066cc8c99dbee881e7a6f1`. Execution uses the separately
cloned `Jeremy-Ryan-Mills/npu_model_rs`; record its exact revision and binary hash
with measurements. Historical performance numbers are context, not this branch's baseline.

## Action items

1. **Establish reproducible measurement.** Build the compiler and Rust simulator;
   retain input bytes, compiler options, revisions, image hashes, exact outputs,
   dispatch counts, device cycles, executed instructions, DMA bytes and engine
   busy counters. Start with independent kernels, then representative complete
   layers. Separate device cycles from host launch and packing time.
2. **Reduce dispatches through composition.** Extend reusable, address-parameterized
   device stages and compatible region composition. Identify remaining host-only
   layout/packing boundaries and eliminate those provably expressible as device
   views or direct consumer reads. Respect final physical instruction memory,
   execution bounds, status propagation, buffer ownership and publication rules.
3. **Avoid intermediate transfers.** Keep private values in registers or VMEM
   across producer/consumer boundaries. Fold exact casts, layouts and static
   quantization into their consumers or producers where source semantics permit.
   Measure actual DMA bytes, including padding and status writes.
4. **Reuse operands.** Improve bounded activation/weight panel selection, repeated
   input caching and reuse across batches. Account for cache fills, VMEM pressure,
   spill traffic and tails; use the measured total rather than read savings alone.
5. **Improve scheduling.** Expose independent DMA and tensor work, prefetch the
   next useful panel, delay waits until actual consumers, prioritize critical
   readouts and evaluate double buffering. Preserve dependencies, DMA channel
   ownership, register hazards and finite scratch capacity.
6. **Reduce instruction overhead.** Hoist invariant address/scalar work, share
   dispatch bodies, balance bounded slices and remove redundant validation only
   under explicit ownership and numerical proofs. Distinguish fewer launches
   from fewer cycles; do not silently raise budgets to claim an optimization.
7. **Promote demonstrated wins.** Add independent shape, tail and refusal cases;
   compare exact outputs and protected memory, rerun relevant compiler checks,
   then retain before/after receipts and update this list. Generic algorithms
   belong in Merlin; Atlas schedules and legality facts belong here.

## Progress

- Created isolated worktrees for both repositories; existing checkouts are preserved.
- Built Rust simulator revision `0186e3781284f4a9ec9d148ac99ec24e05aaf0b8` and added
  issued DMA beat/byte counters on its local `perf/compiler-metrics` branch.
  Fifteen CPU regression tests and Clippy pass; the CUDA test remains optional.
- Built Atlas with the LLVM installation's matching C/C++ compiler. The baseline
  CMake needed LLVM's space-separated definitions split before `add_definitions`.
  The initial system-compiler build crashed during builtin MLIR initialization
  and is not used for measurements.
- Added explicit `--quantized-operand-reuse interleaved-panel-cache`: reuse one
  loaded MXU weight across two or four activation rows, preserving every ordered
  BF16 partial readout. Older policies and the dispatch-reduction preset retain
  their existing schedules. Larger panels retain their original traversal.
- Added Merlin's dense-order proof for identity gathers and singleton-axis
  transposes under `--application-buffer-views`. Alias legality, public buffer
  ownership and distinct input-port checks remain in force. The focused proof,
  generated-C, replay, layout and region tests pass (72 tests).
- Executed the generated C application through Rust-backed MMIO: a three-stage
  2 KiB layout chain becomes two launches with views enabled. After the native
  layout improvement below, matched device cycles are 8,862 versus 5,908;
  reads and writes each fall from 6,144 to 4,096 bytes. Host uploads and downloads
  each fall by the same amount. This is a synthetic layout application, not a model.
- Native layouts now ignore singleton strides when selecting dense DMA and XLU
  transpose paths. The public singleton-only transpose falls from 18,711 to 2,954
  cycles with identical bytes. The layout/application suite passes 37 tests and
  74 subtests; one Python-model integration test remains separately gated.
- Explicit next-weight prefetch was tested on optimized 128 × 128 × 768 panels.
  The scheduler already produces identical machine code (236,013 cycles), so
  that source change was not retained. Artifacts: `out/efficiency/prefetch-*`.
- Added bounded row-exponent reuse to inline interleaved panels when exponent
  caching is selected. Four or fewer LHS pages persist across columns; a RHS
  row page is loaded once per column. Exact outputs, tails, batch identity,
  multi-panel K, shared weights and late invalid exponents pass in both pipelines
  (nine focused tests). Legacy records without panel roles retain their schedule.
- Repaired composition of accumulator-retention stages with stages that contain
  no MXU operations. The latter can inherit the inert hint; a contraction that
  omitted the hint still refuses a mismatch. The three-stage authored MLP
  (32 × 32 → 288 → 32, ReLU and explicit FP8 packing) now executes in one image.
  Both pipelines match independent reference values for all stored intermediates
  and both outputs on two input panels. This is a diagnostic graph, not a
  TorchAO model or a model-quality measurement.
- Removed an unnecessary power-of-two restriction from contiguous layout DMA
  chunk selection. In optimized copies, 192 bytes takes 502 versus 1,031 cycles;
  320 bytes takes 689 versus 1,703; and 8,480 bytes takes 13,482 versus 77,137.
  Bytes moved and the single device launch are unchanged. The layout suite
  passes 48 tests and 74 subtests, including nontrivial affine outer dimensions.
- Extended long dense spans to a bounded full-tile loop plus one tail transfer,
  so prime beat counts also stay coalesced. An 8,224-byte copy falls from 74,809
  to 12,953 cycles; 8,480 bytes improves further to 13,313. Dense copies and
  proven dense-order affine reads include only their existing private transport
  padding in the last beat. Large odd-byte payloads and nested affine rows pass
  exact-output/guard checks; the expanded layout suite passes 58 tests and 74
  subtests. The externally visible element count is unchanged.
- Remaining integration work: complete-layer measurements, residency interaction,
  and more useful overlap in the optimized schedule. No complete-layer or board
  speedup is claimed from the kernel experiments below.

## Initial matched panel measurements

Rust CPU execution, RTL numerics, measured-curve memory, identical input bytes,
128,000,000-instruction software budget and 128 KiB physical image limit. Each
case remains one dispatch. Complete output and protected memory match exactly;
DRAM read/write byte totals are unchanged. Savings are in on-chip movement,
weight pushes and instruction overhead.

| M × N × K | Pipeline | Original cycles | Interleaved cycles | Reduction |
| --- | --- | ---: | ---: | ---: |
| 64 × 64 × 256 | mandatory | 58,341 | 56,060 | 3.91% |
| 97 × 65 × 257 | mandatory | 133,633 | 129,117 | 3.38% |
| 64 × 96 × 2049 | mandatory | 367,215 | 344,712 | 6.13% |
| 128 × 128 × 768 | mandatory | 291,245 | 278,822 | 4.27% |
| 64 × 64 × 256 | optimized | 50,736 | 50,726 | 0.02% |
| 128 × 128 × 768 | optimized | 241,661 | 236,013 | 2.34% |

Local source, machine IR, input/output hashes and measurements are retained in
`out/efficiency/panel-*`; `out/efficiency/env.sh` selects the isolated source trees
and tools. `test/test_interleaved_panels.py` executes independent shapes, tails,
batches, late exponent rejection, both pipelines and native/C export replay with
`ATLAS_RUST_NPU_MODEL` and `ATLAS_OOT_BIN_DIR` explicitly selected.

## Follow-up measurements

With both weight sharing and row-exponent caching, the optimized 128 × 128 × 768
case takes **218,865 cycles** versus the original panel policy's **241,661**
(9.43% fewer). Reads fall from **231,424 to 206,848 bytes** (10.62% fewer), writes
stay at 34,816 bytes, and the image shrinks from 15,852 to 14,812 bytes. Complete
memory bytes match. Reproduce with the selected environment:

```sh
"$EFFICIENCY_PYTHON" tools/profile_quantized_panels.py \
  --m 128 --n 128 --k 768 --pipeline optimized \
  --out out/efficiency/a-new-panel-run
```

The command saves source requests, schedules, virtual and machine IR, full input
and output bytes, instruction proofs, simulator reports and source/tool hashes.
It uses a fresh output directory and the explicitly selected Rust binary.

The connected MLP baseline is 68,289 cycles (mandatory) or 67,777 (optimized),
with 55,296 read bytes and 29,696 write bytes. Applying the existing explicit
`resident-a` policy to the complete mandatory graph reduces it to 61,774 cycles
and 47,104 read bytes, with unchanged outputs and writes. This measures an
existing reuse policy on the newly composable graph; it is not a new kernel
algorithm. Its optimized-pipeline refusal remains in force.

Merlin's full structure gate currently reports the pre-existing root-level
`merlin/tests/test_application_deployment_paging.py`; the no-regex gate reports
four existing call sites in `application_deployment.py` and `baremetal_harness.py`.
Neither failure is introduced by these changes. Package docs were regenerated.

## Measurement acceptance

Use matched source and model revisions, memory timing and numeric modes. Verify
all outputs and rejection/publication behavior. Retain unsuccessful experiments
and report regressions. Simulator measurements describe this model and configuration;
they do not establish board performance. Current focus is reducing actual work
under the existing constraints, with larger software budgets treated separately.
