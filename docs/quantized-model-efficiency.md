# Quantized model efficiency worklist

Continue the dispatch and traffic reduction work on `perf/quantized-model-efficiency`
in Atlas MLIR and Merlin. Starting revisions: Atlas `0ef85dbc154868c3f72b155adfc9170ee52b4494`
and Merlin `4a039544d309afbec4066cc8c99dbee881e7a6f1`. Execution uses the separately
cloned `Jeremy-Ryan-Mills/npu_model_rs`; record its exact revision and binary hash
with measurements. Historical performance numbers are context, not this branch's baseline.

The current branches have been rebased onto the November 22 integration snapshot:
Atlas `fafd5d5244cfcfc1dc38442d36014669311b8d3d`, Merlin
`b9077b094b29307a0de37e65aed317eef1629b47`. Atlas's package dependency pins that exact
Merlin WIP tip from `reed-nicolas/merlin`, as requested. To exercise the additional
Merlin affine-view optimization, use the matching `perf/quantized-model-efficiency`
checkout as an editable override. The original worktrees are preserved, and both
repositories retain `backup/quantized-efficiency-before-nov22-20261122`.
Merlin `origin/main` was fetched; Atlas has no remote `main` (its default branch
is `handwritten-implementation`). No WIP branch was rewritten.

## November 22 measurements

[Machine-readable evidence](measurements/quantized-efficiency-nov22.json) records
the exact source/input/output hashes, compiler binaries, options, and revisions.
The unchanged lifted quantized-linear capture was independently rebuilt on the
WIP base and on this branch. The WIP build needed only the existing CMake fix
that splits LLVM definitions correctly for the selected conda toolchain.

| Measurement | WIP base | This branch, batching enabled |
|---|---:|---:|
| Device launches | 33 | 2 |
| Device cycles | 66,684 | 41,528 |
| Device read bytes | 10,240 | 4,352 |
| Device write bytes | 6,144 | 3,232 |
| Host upload bytes | 8,192 | 4,352 |
| Host download bytes | 4,100 | 3,204 |

The public output SHA256 is identical:
`912221867948cc98cec7a08073e070a211ee268cd7d5c31011c846afc685a3ce`.
On this branch alone, turning batching off gives 33 launches and 51,536 cycles;
host transfer callbacks fall from 36 writes/66 reads to 5 writes/4 reads when
batching is enabled. These are simulator measurements for this capture, not
full-model, board, or end-to-end latency measurements.

The authored 128 × 128 × 768 panel comparison also replays on the new base:
`batch-panel-cache` takes 231,629 cycles and 198,784 read bytes;
`interleaved-panel-cache` takes 208,761 cycles and 197,248 read bytes.
Both use one launch, write 32,800 bytes, and produce identical protected memory.

The shared C runtime now reads only occupied BF16 register halves for partial
output tiles, including final row/column tails. Full tiles retain bulk reads.
Examples of total host download bytes, including status:

| M × N × K; batch | Previous prefix read | Occupied-half read |
|---|---:|---:|
| 35 × 37 × 65; 2 | 16,388 | 6,636 |
| 3 × 35 × 65; 2 | 4,484 | 528 |
| 3 × 5 × 65; 2 | 196 | 152 |
| 35 × 3 × 65; 2 | 4,100 | 2,140 |
| 1 × 17 × 33; 1 | 1,060 | 38 |
| 32 × 32 × 32; 1 | 2,052 | 2,052 |

Generated-C/Rust tests check every output and transfer count against independent
logical-word indexing in both pipelines. Streamed and full-frame host packing
preserve public outputs when the last read fails. Device work is unchanged by
this host-only change. Partial tiles can require two callbacks; host latency has
not been measured. Hoisted tensor/batch exponent loads were also ported to the
128-byte payload after the rebase exposed one remaining 2-KiB load path.

Machine scheduling now visits only graph-ready instructions, retaining original
index order for priority ties. The 225 × 35 × 3073 panel probe emits identical
machine IR before and after. Counted machine-scheduling work falls from
240,189,729 to 15,640,033 units; total compile work falls from 440,479,204 to
215,929,508. Single-run compilation profiles were 43.26 and 28.29 seconds.
Wall time varies: the existing 30-second test still timed out under load.
No work allowance or test timeout was increased. Exact work-boundary refusal
and the scheduler/slot/DMA/timing suites pass.

Dependency-edge merging now uses a hash lookup while retaining the original
ordered edge vector. Two local comparisons after the ready-set change measured
29.16 → 27.39 seconds and, with run order reversed, 28.94 → 26.74 seconds.
Both emitted identical machine IR and identical work-budget reports. These are
local compilation timings, not device-cycle savings. The affected scheduler,
delay, DMA, work-budget, and Rust panel suites pass 61 tests and 46 subtests.
The previously timing-sensitive large-panel Rust test also passes with the final
compiler and its unchanged 30-second per-compilation timeout.

An authored DMA-only probe identifies the next device traffic opportunity. Each
case includes the same 2,048-byte input load, then writes a physical BF16 tile
prefix or its two occupied halves. All logical bytes and protected memory are
checked. This is **not integrated into contraction code generation**.

| Tile rows × columns | Prefix write bytes / cycles | Serial halves bytes / cycles | Queued halves bytes / cycles |
|---|---:|---:|---:|
| 1 × 17 | 1,056 / 2,319 | 64 / 1,654 | 64 / 1,621 |
| 3 × 32 | 1,120 / 2,363 | 192 / 1,748 | 192 / 1,715 |
| 16 × 17 | 1,536 / 2,644 | 1,024 / 2,330 | 1,024 / 2,298 |
| 24 × 32 | 1,792 / 2,818 | 1,536 / 2,676 | 1,536 / 2,644 |
| 31 × 32 | 2,016 / 2,969 | 1,984 / 2,979 | 1,984 / 2,948 |
| 32 × 32 | 2,048 / 2,991 | 2,048 / 3,023 | 2,048 / 2,991 |

Splitting adds eight executed instructions in this probe. Serial splitting can
lose cycles despite moving fewer bytes; full tiles should retain bulk transfers.
Queued results assume both DMA channels are available. Repeat with
`python tools/profile_partial_output_dma.py --out /path/to/new/probe` under the
same tool/model environment. The selected model's timing is part of this result;
contraction integration, application latency, and board measurements remain open.

## November 22 validation and reproduction

The rebased native tools build successfully. The selected WIP deployment,
quantized-loop, panel, residency, and independent reference suites finish with
**422 passed, 5 skipped, 284 subtests passed, and 8 failures**. The eight failures
(including three failed subtests) reproduce on the WIP base with the same
reference environment: the external `functional_runtime` module is absent,
two RISC-V compilation fixtures fail, archive byte reproducibility fails, and
resident ELF fixtures omit the required `inputs` field. These gates remain
unresolved. The large-panel test passed in this run, but the earlier timeout
under load remains a compile-latency concern.

Focused transport, independent Rust-output, late rejection, scheduler, and work
budget suites pass; their individual counts and log hashes are in the measurement
record. The Rust model uses RTL numerics with explicit `register_rows` packing.
Python numerical qualification uses the pinned `npu_model` revision
`b97b2f5a97e432141c4ef8504bec396651d77a8f` and the real built Merlin host-math
supplier. The simulator results do not constitute independent board qualification.

Merlin's focused runtime/placement suite passes **113 tests and 4 subtests**.
Its artifact-layout check passes. The full structure report matches the WIP
baseline, including existing package-documentation and dependency/size-ratchet
issues. The test-budget check reports **278 collected cases against a ceiling
of 150 on both revisions**. The affine-view coverage was integrated into existing
placement cases; no test allowance was raised.

To repeat the captured application comparison, select the matching Atlas native
tools, the Rust executable, and the Merlin efficiency checkout in the environment
(`ATLAS_OOT_BIN_DIR`, `ATLAS_RUST_NPU_MODEL`, `MERLIN_NATIVE_ROOT`, and Python's
module path), then run:

```sh
python tools/profile_quantized_capture.py \
  --compile-inputs /path/to/unchanged/compile-inputs \
  --input /path/to/recorded/input.bin \
  --pipeline optimized --out /path/to/new/comparison
```

The evaluator refuses mismatched input extents, compiles and verifies both
applications before reading input bytes, checks public outputs and guard bytes,
and records tool/source/input/output hashes in `comparison.json`. The receipt
identifies the source and input hashes required for the table above. Local logs
and full compiled artifacts remain under `out/efficiency/`; the compact evidence
and remaining action items are versioned here.

## Next work after this checkpoint

1. Finish qualifying the WIP deployment gates with the external functional
   evaluator and platform dependencies. Preserve the pinned Python model,
   explicit numerical declarations, refusal paths, and source/publication checks.
2. Reduce the remaining all-pairs dependency-graph work in large native blocks;
   prove equivalent hazards and scheduling ties before changing its traversal.
   The ready-instruction change removes redundant candidate scans, but does not
   remove dependency construction or verification costs.
3. Measure host callback latency for partial-tile reads, including whether
   coalescing nearby occupied halves beats the byte minimum on the real transport.
   Then evaluate splitting the matching device output DMA into occupied register
   halves. For a 1 × 17 BF16 tile, the present 1,056-byte prefix could become two
   32-byte transfers, with the same frame offsets and status store. The authored
   probe above measures this transport opportunity and its near-full-tile
   regression. A source-offset store needs explicit lowering, replay, and memory
   contract support before application selection. Account for available DMA
   channels and retain bulk stores where splitting loses. Row/column tails need
   independent application checks, including rejection and preservation of
   unwritten gaps.
4. Extend the existing quantizer-to-contraction handoff proof beyond 32 × 32.
   Larger codes and exponent pages require explicit physical-layout and private
   ownership proofs before application selection can use the combined image.
5. Evaluate double-buffered panel loads with measured DMA/MXU overlap and total
   cycles. Extend the WIP resident layer-loop implementation. Layout-copy
   composition is being developed elsewhere and is outside this work.

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
- Added explicit Rust `pack_layout="register_rows"` for this compiler's pinned
  RTL. The Rust port's default `tile_rows` targets a newer Python model layout;
  using it for raw quantizer PACK changes outputs. Rust commit `a231973` preserves
  that default and adds the selected layout, including read/write timing. All
  recorded RTL vector traces now run, including four previously excluded PACK
  and UNPACK cases (16 CPU tests pass). The application harness selects and
  checks `register_rows` explicitly. The panel comparison and independent MLP
  intermediate checks pass again with this setting; panel cycles and traffic
  are unchanged. New panel artifacts: `out/efficiency/official-panel-register-128-128-768`.
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
- Added opt-in `--dynamic-quantizer-group-batching`. Complete aligned BF16 token
  groups share a counted body while retaining separate exponents. Candidates
  contain at most 256 groups and must pass the selected machine instruction
  proof; resource refusals halve the candidate, then reuse the admitted size.
  Compiler work exhaustion propagates without retries. Defaults and presets
  are unchanged. The quantizer, option, independent host-reference, generated-C,
  and saved-schedule replay checks pass 250 tests and 32 subtests (one separately
  gated Python-model test). A 256 × 1024 source uses eight admitted 32-group
  bodies under the original 1,000,000-instruction limit.
- Hoisted a complete single-tile dynamic input above reduction and encoding so
  the same SSA tensor feeds both phases. This removes its second DRAM read
  without introducing a VMEM scratch buffer. Independent bitwise references,
  special values, multiple groups, public guards, actual read counters and
  generated-C replay pass (27 focused tests; 59 related tests and 32 subtests).
  Multi-tile traversal is unchanged. Static and folded-input quantizers retain
  their source and publication checks.
- Removed reduction/encoding loop control when the group is statically one
  tile, and exponent-broadcast loop control when exactly one exponent is stored.
  Scalar correction loops for exceptional values remain counted. Static,
  token, block, head-block, head and multi-tile tensor sources match independent
  references in both pipelines (100 tests and 32 subtests across the affected
  suites; two Python-model-only tests remain separately gated).
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

On the existing lifted 32 × 32 quantized-linear capture, token grouping reduces
actual application launches from **33 to 2**, device cycles from **70,416 to
60,438** (14.17%), and device writes from **6,144 to 5,248 bytes**. Reads remain
10,240 bytes and host uploads remain 8,192; host downloads fall from 4,100 to
3,204 bytes. Both runs use the same input bytes, mandatory pipeline, 1 MiB
staging window, pinned RTL packing, and 1,000,000-instruction limit. Public
outputs match exactly. This is a captured small linear fixture, not a complete
language model. The earlier `captured-linear-baseline`/`captured-linear-batched`
experiments used the wrong raw PACK layout and are superseded by
`out/efficiency/captured-linear-batching-final`.

Reproduce a matched application comparison with an existing lifted capture:

```sh
"$EFFICIENCY_PYTHON" tools/profile_quantized_capture.py \
  --compile-inputs /path/to/compile-inputs --input /path/to/bf16-input.bin \
  --out out/efficiency/a-new-capture-run
```

The evaluator requires one runtime BF16 input, keeps captured parameters and
source policy, compiles both applications before reading execution samples,
replays exports, and retains simulator configuration and source/tool hashes.
With single-tile input reuse, the same batched captured fixture improves further
from **60,438 to 51,322 cycles** and **10,240 to 8,192 read bytes**. Dispatches
remain two and writes remain 5,248 bytes. Relative to the original 33-launch
run, the combined changes remove 27.12% of device cycles and 20% of device reads.
The unbatched run also benefits (61,392 cycles). Matching output hashes and
receipts are in `out/efficiency/captured-linear-input-reuse`.

Removing the one-iteration loops then reduces the batched fixture to **49,882
cycles** and **12,624 executed instructions**, versus 51,322 and 14,608 after
input reuse alone. Dispatch and byte counts remain unchanged; output hashes
still match. This is 29.16% fewer cycles than the original 70,416-cycle run.
Artifacts: `out/efficiency/captured-linear-single-iteration`.

An experiment extended one-prefix reuse to non-power-of-two groups. With mixed
finite/special inputs, 3 × 96 optimized quantization fell from 12,231 to 6,898
cycles, but 3 × 768 regressed from 26,132 to 31,352: one exceptional value now
forces a larger scalar finite-only scan. This broader change is **not retained**.
Results and machine sources: `out/efficiency/quantizer-prefix-experiment`.
The larger authored 32 × 5632 × 2048 projection takes 9,376,678 cycles with both
panel policies with scalar LHS exponents: one M tile uses the existing row-cache
fallback and its scalar page is already invocation-cached. Outputs match; no
gain is claimed for that configuration. Artifacts:
`out/efficiency/panel-32-5632-2048`.

At 128 × 5632 × 2048, the existing interleaved panel improvements reduce cycles
from **15,101,929 to 13,405,993** (11.23%) and reads from **14,682,112 to
13,600,768 bytes** (7.37%). Writes remain 2,885,632 bytes and both runs use one
dispatch. Complete memory matches. Artifacts: `out/efficiency/panel-128-5632-2048`.

The profiler now accepts `--lhs-exponents row` for per-token scales. With this
source, the 32 × 5632 × 2048 row-cache fallback can retain its LHS exponent page
across columns. It reduces cycles from **9,736,339 to 9,473,598** (2.70%) and
reads from **12,320,768 to 11,962,368 bytes**, with the same output, writes and
dispatch count. This cache is recorded explicitly under the selected
`interleaved-panel-cache` policy; old records without the new row role retain
their prior traversal. Each new M row and batch refreshes its page, while scale
domain checks and scalar-refusal behavior remain active. The affected row,
panel, page-cache and interleaving suites pass 38 tests, including native
rebinding and C export. Artifacts:
`out/efficiency/panel-token-exponents-32-5632-2048`.

Two baseline compiler limits were encountered independently of row-page reuse:
the batched 65 × 35 × 65 QK path exhausts non-spillable working registers, and
the optimized 3 × 35 × 6145 wide-cache path reaches its 256-spill work limit.
Neither limit was raised. Baseline sources/refusals are retained in
`out/efficiency/row-cache-existing-register-refusal` and
`out/efficiency/row-cache-existing-wide-spill-refusal`. Long-prefix numerical
checks currently use the admitted mandatory pipeline.

Exponent DMA now loads only the first **128 bytes** of each 2 KiB page: those
32 signed words contain every scale used by restoration. Page addresses,
allocation sizes, host uploads and resident asset formats are unchanged; the
partial BF16 load zero-fills the remaining scratch lanes. Both ordinary and
panel-prefetch paths use the bounded prefix. The affected suites pass **51
tests** (five separately gated Python-model tests), including poisoned unused
padding, scalar/row/batch scales, invalid scales, both pipelines, native
rebinding and C replay.

With this change, the captured batched linear fixture takes **47,301 cycles**
and **4,352 read bytes**, retaining two dispatches, 5,248 write bytes and the
same public output hash. Against the original corrected baseline, that is
32.83% fewer cycles and 57.5% fewer device reads; host uploads are still 8,192
bytes. Artifacts: `out/efficiency/captured-linear-prefix-scales`.

The row-scaled 32 × 5632 × 2048 projection improves further from **9,473,598 to
9,257,750 cycles**, with reads falling from 11,962,368 to **11,622,528 bytes**.
The scalar-LHS 128 × 5632 × 2048 interleaved case improves from **13,405,993 to
13,156,874 cycles** and from 13,600,768 to **13,260,928 read bytes**. Writes,
dispatch counts and complete memory bytes are unchanged in both comparisons.
Artifacts: `out/efficiency/panel-prefix-scales-32-5632-2048` and
`out/efficiency/panel-prefix-scales-128-5632-2048`.

Completion now writes one **32-byte DMA beat**, instead of a complete 2 KiB
status page. The page reservation and subsequent buffer alignment are unchanged;
the host still checks the first u32 before exposing output. The remaining 2,016
private padding bytes are intentionally untouched. Repeated success after stale
failure status, late invalid scales, input/output guards, both pipelines,
residency, epilogues, canonical rebinding and generated C pass **52 tests and
39 subtests** (two Python-model tests separately gated).

The captured batched fixture consequently takes **45,965 cycles** and writes
**3,232 bytes**, down from 47,301 cycles and 5,248 bytes. Reads remain 4,352 bytes,
launches remain two, and public output hashes match the original corrected run.
Artifacts: `out/efficiency/captured-linear-status-prefix`. This comparison uses
public outputs and protected memory; unused status padding is no longer zeroed.

Constant scaling masks are now omitted when the RHS exponent is scalar or
per-batch: those factors already broadcast over the complete tile. Row-scale
setup clears eight adjacent words per counted iteration, reducing 256 loop
iterations to 32. An unroll sweep of 1/2/4/8/16/32 retained eight as the smaller
image tradeoff: the mandatory 32 × 32 × 32 case falls from **10,915 to 8,899
cycles**, while its image grows from 3,008 to 3,176 bytes. Larger unrolls offer
diminishing savings. The affected suites pass **43 tests** (five separately
gated Python-model tests), including scalar/batch mask omission, invalid scales,
complete-memory equality, both pipelines and native/C replay. Sweep artifacts:
`out/efficiency/mask-setup-experiment`.

The captured batched fixture now takes **43,949 cycles** and **10,624 executed
instructions**, retaining two dispatches, 4,352 read bytes, 3,232 write bytes and
the original public output hash. That is 37.59% fewer device cycles than the
corrected 70,416-cycle baseline. Artifacts:
`out/efficiency/captured-linear-mask-setup`.

An explicit physical handoff experiment connects the capture's 32 × 32 dynamic
quantizer to its contraction in one image. Both pipelines pass the existing
native checks and original 1,000,000-instruction proof. Quantizer intermediates
match the independent host reference; complete memory matches two separate
device programs on captured, varied, exceptional and refusal inputs. The
optimized finite case changes **41,561 → 40,655 cycles** and **two → one
dispatch**. Mandatory changes **43,949 → 44,025 cycles**, and the subnormal
refusal path also regresses. DMA bytes are unchanged. This is **an experiment,
not an application fusion provider**: it uses an explicitly authored frame and
does not yet prove generic selection, packing or removal of the intermediate
host code-domain check. No host-transfer timing claim is made. Artifacts:
`out/efficiency/quantized-handoff-experiment` and its local probe script.

Single-M-tile and single-N-tile contractions now read only their active FP8
rows. A one-row activation uses **32 bytes per K tile**, instead of 1,024;
partial DMA supplies positive zero to the unused rows. Packed addresses,
strides, allocations and host transfers stay unchanged. The counted emitter
uses independent activation/weight transfer sizes, including activation-cache
fills. Multi-row-tile tails retain their original full transfers.

Matched optimized measurements, with complete memory equality:

| Shape | Policy | Cycles before → after | Read bytes before → after |
| --- | --- | ---: | ---: |
| 3 × 65 × 257 | ordered | 56,939 → 38,559 | 56,064 → 31,008 |
| 3 × 65 × 257 | interleaved/row cache | 42,496 → 36,721 | 37,376 → 29,024 |
| 1 × 5632 × 2048 | ordered | 17,798,048 → 10,001,073 | 23,091,328 → 11,917,440 |
| 1 × 5632 × 2048 | interleaved/row cache | 9,163,281 → 9,122,116 | 11,622,528 → 11,559,040 |

All remain one dispatch with unchanged writes. The incremental large-case
benefit is smaller with activation caching already enabled. Partial-load zero
fill also adds instructions: the large cached image grows from 11,592 to 13,508
bytes, still within its checked limit. Artifacts:
`out/efficiency/operand-prefix-experiment`.
Checks cover both pipelines, activation and weight prefixes, cache fills,
batches, late scale refusals and poisoned unused rows. Native/C rebinding and
the existing interleaved/page-prefix regressions pass. The broader installed
CLI check initially lacked the Atlas entry point in the test interpreter;
installing the provider into isolated `out/efficiency/provider-site` resolves
that setup issue (six looped-CLI tests and 21 subtests pass; the additional
reuse/interleaving/prefix suite passes 29 tests).

Partial DMA now reuses its reserved zero tensor until a physical write or
control-flow boundary invalidates it. It also reuses an already cleared staging
suffix for nonshrinking prefixes of the same tile width within one source block.
Slot saves/restores preserve that fact because their checked allocations are
disjoint from staging; other source operations conservatively discard it.
Lowering, CFG expansion and tile records derive the decision from source and
checked placements. The final buffer proof still verifies every word's origin
and rejects an intervening scalar write or a nonzero prefill.

Matched interleaved/row-cache measurements against the preceding operand-prefix
compiler, with complete memory equality, unchanged DMA bytes and one dispatch:

| Shape | Pipeline | Cycles before → after | Image bytes before → after |
| --- | --- | ---: | ---: |
| 3 × 65 × 257 | mandatory | 39,796 → 39,181 | 3,520 → 3,324 |
| 3 × 65 × 257 | optimized | 36,721 → 36,484 | 3,560 → 3,348 |
| 128 × 128 × 768 | mandatory | 260,946 → 260,626 | 16,304 → 16,196 |
| 128 × 128 × 768 | optimized | 219,156 → 219,022 | 16,528 → 16,420 |
| 1 × 5632 × 2048 | mandatory | 9,532,728 → 9,528,146 | 12,072 → 10,560 |
| 1 × 5632 × 2048 | optimized | 9,122,116 → 9,119,905 | 13,508 → 11,736 |

Tensor reuse alone slightly regressed the optimized 128 × 128 × 768 case by 76
cycles; the combined staging-tail change removes that regression. The captured
batched fixture changes **43,949 → 43,844 cycles**, with 10,615 executed
instructions, the same two launches, traffic and public output hash. Artifacts:
`out/efficiency/zero-reuse-experiment`,
`out/efficiency/staging-zero-reuse-experiment` and
`out/efficiency/captured-linear-staging-zero-reuse`. The measured combined
`atlas-opt` SHA256 is
`4ef8f4d0a0bbe4757aab5b6aedc45f5f7e05727c36581552af48fc7446727243`.
The broader DMA, scheduling, CFG, tile, source-memory, buffer and quantized-prefix
suites pass **182 tests and 1,881 subtests** (three separately gated tests).
Four additional focused cases cover full BF16 writes and corrupted reused tails.

Scalar helper literals now survive until an actual scalar write invalidates
their destination. Repeated setup of an unchanged DMA address, size or scratch
base emits no instruction. Unknown operations and control-flow boundaries
discard all facts; source constants retain explicit SSA producers. Final
address, CFG, buffer and timing checks remain active. Two verifier mutation
fixtures now supply explicit mutation sites/stable bases instead of assuming
redundant setup instructions exist.

The captured batched application changes **43,844 → 43,238 cycles** (1.38%
fewer), with **10,615 → 10,512 executed instructions** and unchanged two
dispatches, DMA/host bytes and exact output. Relative to the original corrected
70,416-cycle capture, the retained changes now remove 38.60% of device cycles.
Artifacts: `out/efficiency/captured-linear-scalar-reuse-only`.

This is a code-size/captured-application win with small panel cycle tradeoffs:

| Shape | Pipeline | Cycles before → after | Image bytes before → after |
| --- | --- | ---: | ---: |
| 3 × 65 × 257 | mandatory | 39,181 → 39,191 | 3,324 → 3,196 |
| 3 × 65 × 257 | optimized | 36,484 → 36,484 | 3,348 → 3,240 |
| 128 × 128 × 768 | mandatory | 260,626 → 260,826 | 16,196 → 15,204 |
| 128 × 128 × 768 | optimized | 219,022 → 219,220 | 16,420 → 15,532 |
| 1 × 5632 × 2048 | mandatory | 9,528,146 → 9,525,747 | 10,560 → 9,772 |
| 1 × 5632 × 2048 | optimized | 9,119,905 → 9,119,912 | 11,736 → 10,536 |

Complete memory, traffic and single-dispatch counts match. The largest measured
regression is 0.091%; fewer instructions do not guarantee fewer device cycles.
Artifacts: `out/efficiency/scalar-reuse-experiment`; measured `atlas-opt` SHA256
`4ee4b31c3a03145a281ed0d1b0ff3fa49518be252db968aca41b16aef5307b6b`.
The affected native, DMA, CFG, buffer, spill, scalar, PACK, slot, work-budget
and quantized-prefix suites pass **191 tests and 2,400 subtests** (five optional
integration tests separately gated). Compiler policy limits are unchanged.

A further experiment used signed ADDI deltas for nearby known addresses.
It saved only one additional cycle in the captured application, while the
mandatory 1 × 5632 × 2048 case regressed to 9,528,387 cycles. That extension is
**not retained**, despite its smaller images. Candidate source and receipts:
`out/efficiency/scalar-delta-reuse-experiment` and
`out/efficiency/captured-linear-scalar-delta-reuse`.

Additional generated-C testing exposed a partial-load regression in the ordered,
batched K loop: after the first continuation tile, both operand pointers advanced
by the activation transfer length instead of the packed 1,024-byte tile stride.
A 3 × 35 × 65 batched QK example returned 64 instead of 65 before scaling.
The counted loop now carries separate activation/weight stride constants.
Both pipelines match full-transfer programs on left-only, right-only and two-sided
short operands, shared and per-batch weights, and late exponent refusals. The
existing unbatched captured benchmark uses a different K path and is unaffected.
The prefix/generated-C suite passes 20 cases; the broader counted, resident,
panel, interleaving and work-bound suite passes 42 tests and 42 subtests (six
optional Python-model integrations are separately gated).

Generated contraction C now uploads and initializes only each exponent page's
128-byte payload, retaining its existing 2,048-byte stride and allocation.
Resident exponents remain untouched; other operand uploads and permission spans
are unchanged. Quantizer C downloads only the logical exponent words, including
short final bodies, while retaining device DMA alignment. All required transfers
still complete before public output is copied.

The captured linear fixture's host uploads fall **8,192 → 4,352 bytes (46.875%)**.
The 33-launch version's host downloads fall **4,100 → 3,204 bytes**; the two-launch
version already downloads 3,204 bytes. Its **43,238 device cycles**, 4,352 DMA read
bytes, 3,232 DMA write bytes and exact public output remain unchanged. Host time
is not measured: multiple exponent pages now use multiple smaller callbacks.
Artifacts: `out/efficiency/captured-linear-host-padding-final`, with matched input,
output and simulator/compiler provenance.

The transport/residency/epilogue suite passes **40 tests and 32 subtests** (two
optional integrations gated), and the quantizer batching/application suite passes
**29 tests and 32 subtests** (one optional integration gated). Four additional
generated-C executions cover both pipelines, multi-page BF16 and batched f32 QK
outputs against independently calculated dot products, poisoned host/device page
padding, protected memory, unchanged inputs, and atomic refusal of a bad final
exponent. A last-upload failure also prevents launch, download and publication.

Counted contractions now record a shorter output-tile write prefix when M fits
one row tile or N fits one 16-column register half. The prefix covers every
logical BF16 lane in the two physical register halves; allocations and 2,048-byte
tile strides remain unchanged. Multi-row-tile tails retain full row coverage.
Only final restored outputs use this extent: intermediate panel stores remain
complete. Padding after the recorded prefix is untouched and must not be assumed
initialized by a later consumer. Missing metadata preserves the older full store;
source generation and C export validate the recorded extent against the shape.

Matched optimized measurements (mandatory improves on all four shapes too):

| Shape | Batches | Cycles before → after | DMA write bytes before → after |
| --- | ---: | ---: | ---: |
| 3 × 65 × 257 | 1 | 36,484 → 34,672 | 6,176 → 3,392 |
| 1 × 5632 × 2048 | 1 | 9,119,912 → 9,006,216 | 360,480 → 185,888 |
| 3 × 5 × 65 | 2 | 13,516 → 10,913 | 4,128 → 224 |
| 35 × 3 × 65 | 2 | 32,140 → 29,480 | 8,224 → 4,128 |

Reads and dispatch counts are unchanged. Every logical output and written prefix
matches the full-store program; untouched tails match poisoned initial memory.
The focused prefix/generated-C suite passes **21 tests**, including the 16/17
column boundary, full-size controls, both pipelines, replay and late refusal.
The broader contraction/residency/epilogue/restoration/panel suite passes **50
tests and 26 subtests** (eight optional integrations gated). Artifacts:
`out/efficiency/output-prefix-recorded`; the earlier isolated experiment is
`out/efficiency/output-prefix-experiment`. The 32 × 32 captured linear
fixture requires both complete halves, so this shape-specific change does not
improve its measured 43,238 cycles.

Generated C now also downloads just the recorded prefix of each output tile,
with the complete permission span and allocation retained. Real generated-C/Rust
runs verify output-plus-status downloads of **8,196 → 4,484 bytes** for batched
3 × 35 × 65 QK, **4,100 → 196 bytes** for batched 3 × 5 × 65 linear, and
**8,196 → 4,100 bytes** for batched 35 × 3 × 65 QK. Device programs and cycles are
unchanged by this host change. Host output padding remains untouched; the last
tile download failing leaves public output unchanged. Full-output records keep
their one bulk read. These smaller transfers add callbacks for multiple tiles;
host latency is not measured. The generated-C, transport, resident, narrowing,
epilogue and row-block suite passes **32 tests and 26 subtests** (two optional
integrations gated).

Next: extend the handoff proof to useful larger physical layouts before changing
application selection. Preserve exceptional scan
granularity while reducing quantizer reads, and examine residency and compiler
register pressure. Investigate the panel scheduling tradeoffs before broadening
scalar-literal reuse across control-flow boundaries.

Before the November 22 rebase, Merlin's full structure gate reported the root-level
`merlin/tests/test_application_deployment_paging.py`; the no-regex gate reported
four existing call sites in `application_deployment.py` and `baremetal_harness.py`.
Those are historical results from the old layout, not the current WIP gate
status. Package docs were regenerated for that earlier check.

## Measurement acceptance

Use matched source and model revisions, memory timing and numeric modes. Verify
all outputs and rejection/publication behavior. Retain unsuccessful experiments
and report regressions. Simulator measurements describe this model and configuration;
they do not establish board performance. Current focus is reducing actual work
under the existing constraints, with larger software budgets treated separately.
For this compiler, select `pack_layout="register_rows"` in Rust bundles. Arithmetic
mode `numerics="rtl"` alone does not select register layout. The current selected
Rust binary SHA256 is `1d141164229004be150bd89c436b5edd15453089ef369edf1506201af29f890a`.
