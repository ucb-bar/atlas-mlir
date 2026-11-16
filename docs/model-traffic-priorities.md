# Remaining model compilation optimizations

Reconciled on 2026-11-16 against Atlas PR #50 at `0748e5a` and Merlin PR #63
at `cadc2080b`. The next implementation branch in each repository is
`perf/model-traffic-reduction`. The old `perf/model-dispatch-reduction` branches
and their measurement artifacts remain unchanged.

## What is already covered

The current PR heads include the prior dispatch-reduction stack: exact cast
boundary folding, dense storage views, operand layout folding, operand and
exponent-page caches, panel readout, private VMEM staging, per-scratch-buffer
scheduling, and execution-budget-aware quantizer and softmax dispatches.
Merlin common-value elimination is also already available. None of these is a
new opportunity to claim or reimplement.

The updated heads additionally make compiler recipes explicit options, repair
I/O placement and replay, and improve compiler implementation performance.
Historical environment-only launch scripts must be translated to those options.
Use a fresh native snapshot with the selected I/O capacity and current Merlin
sources; do not bypass snapshot checks to reuse an older snapshot.

## Evidence and order

The retained exact single-layer measurements at `ac077e3` are:

| Layer | Device cycles | Dispatches | DRAM bytes | Serial same-work resource bound |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama, B=1, 64 tokens | 74,114,788 | 56 | 79,481,664 | 54,724,767 |
| SmolVLA vision, B=1, 1,024 tokens | 491,010,121 | 113 | 459,809,376 | 353,461,374 |

These are Rust model measurements with RTL numerics, not hardware timings or
full-model results. The resource bound preserves the current work and serial
launch boundaries; it is not a speedup prediction. A fresh PR-head control is
required to isolate each new optimization from upstream changes.

| Priority | Work | Evidence / expected benefit | Ownership |
| --- | --- | --- | --- |
| 1 — implemented | Admit valid zeros through the vector check for exact BF16 addition/subtraction | Saves 2.60M TinyLlama and 50.85M SmolVLA cycles in matched full-layer runs; details below. | Atlas: proof against the selected VPU primitives and native route selection. |
| 2 — implemented | Fuse private softmax output with static activation quantization | SmolVLA writes a 25,165,824-byte BF16 probability tensor that its quantizer reads back. Removing that pair of transfers removes about 34.1M cycles of bandwidth demand at 1.477 B/cycle; actual saved cycles require simulation. | Merlin: reusable private producer/consumer ownership proofs. Atlas: row schedule, original BF16 rounding, exact FP8 conversion and exponent publication. |
| 3 — traffic prerequisite implemented | Reduce exact bias/residual traffic, then consider producer epilogues | Single-pass validation/computation now saves 1.27M TinyLlama and 26.81M SmolVLA cycles. Projection-store fusion remains future work: it must preserve post-projection rounding, safe fallback/publication and operand reuse. | Merlin: existing private result and broadcast-layout proofs. Atlas: exact arithmetic, private device publication and producer-store fusion. |
| 4 | Pipeline tile DMA with vector compute and validation | Before priority 1, exact-float and folded-layout kernels accounted for about 7.26M TinyLlama and 106.75M SmolVLA cycles in which DMA was inactive. Remeasure after scalar work removal before selecting schedules. Prefetch the next independent tile while processing the current one. | Atlas: VMEM lifetimes, register pressure, DMA slots and instruction timing. |
| 5 | Compose compatible regions while keeping private intermediates on device | The current 56/113 launches include real data dependencies as well as image/resource limits. Composition must remove repeated publication/packing or enable overlap to reduce device cycles; merely concatenating launches chiefly changes host overhead. | Merlin: generic composition, ownership and lifetime contracts. Atlas: admitted schedules, memory placement and image bounds. |
| 6 | Extend independent-request batching and weight reuse | B=2 already improves TinyLlama throughput. Further batching amortizes weights but increases activation storage and attention work per batch. Measure throughput separately from single-request latency. | Merlin: independent-request graph semantics. Atlas: panel selection and residency. |

TinyLlama's seven projections contain about 44 MB of FP8 weights. At the
selected model bandwidth, fetching them once already costs about 29.8M cycles
of service. More scheduling alone cannot remove that traffic. Higher DMA busy
percentage and higher MXU busy percentage are not objectives when achieved by
doing extra work. Optimize end-to-end cycles, useful MAC throughput and traffic.

General MXU0 substitution, approximation of the exact float regions, and changing
reduction/rounding order are separate numerical-policy projects. They are not
exact rewrites under the current comparison contract.

## First implementation: zero-aware vector admission

The opt-in `exact_binary_vector_preflight` route first retains its cheap min/max
check. If the minimum magnitude is zero, it additionally checks each operand
using `t = min(abs(x), 2^-32) * 2^32` and the probe `t * (1 - t)`.
Within the finite bounded domain, the probe is zero exactly when the original
operand is zero or has magnitude at least `2^-32`. The existing upper-magnitude
check rejects values above `2^32`, infinities and NaNs.

The smallest nonzero BF16 input becomes `2^-101`, so an inadmissible small value
cannot disappear through underflow in the probe. The nearest representable
value below one is `255/256`, so subtraction cannot turn it into zero either.
The arithmetic operands are never replaced by probe values. Scalar fallback,
whole-input validation, atomic output publication and resource checks remain.

This is a runtime domain check derived from the operation's numeric contract.
It does not depend on model names, captured values, calibration or reference
outputs. It needs no new Merlin optimization or target-specific shared pass.

### Matched full-layer results

Fresh builds at the PR heads reproduce the retained baseline cycle counts
exactly. Both variants use the same Merlin snapshot, inputs, weights, native
tools, compiler options and Rust executable. Only the Atlas vector admission
implementation differs. All 131,072 TinyLlama and 786,432 SmolVLA output words
match the independent original-FX Atlas-arithmetic references.

| Fixture | Before cycles | After cycles | Saved | Dispatches |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama B=1, 64 tokens | 74,114,788 | 71,519,269 | 2,595,519 (3.50%) | 56 → 56 |
| SmolVLA vision B=1, 1,024 tokens | 491,010,121 | 440,156,657 | 50,853,464 (10.36%) | 113 → 113 |

Instructions fall from 7,786,086 to 5,107,077 and from 77,654,613 to 25,165,229,
respectively. Every launch retains its exact DRAM read/write byte counts.
Every unchanged program image retains identical cycles, instruction histogram
and engine-busy counters. Changed images are confined to exact-float additions
and their folded-input-layout wrappers.

The nonzero route pays a small branch overhead: TinyLlama's entirely admitted
residual add grows from 1,641,238 to 1,642,006 cycles (+768, or 0.047%). Its other
three changed additions save 1,204,064, 150,508 and 1,241,715 cycles. The complete
layer improvement includes this regression. SmolVLA has 42 slightly slower
launches, totaling 32,256 additional cycles; its reported net saving includes
all of them.

DMA busy fraction increases from 79.49% to 82.37% for TinyLlama and from 68.24%
to 76.12% for SmolVLA. These are engine-busy fractions, not payload bandwidth or
useful MXU utilization. The total time decreases without adding DMA traffic.
The recalculated serial same-work resource bounds are 53,811,754 and
311,310,882 cycles. Further traffic elimination can lower those bounds.

The runs retain Rust `be1b521015ff54ec75699c2d5d63e567ab0ae22a`, CPU, RTL
numerics, `register_rows`, 64 MiB I/O, 128,000,000 dynamic instructions and the
unchanged 128 KiB physical IMEM limit. Host packing/dispatch time is excluded.
SmolVLA is the independently calibrated vision block, not an action-expert or
full-policy run.

Directed validation passes six tests and ten subtests, including exhaustive
65,536-encoding admission analysis and Rust add/subtract runs with partial and
multiple tiles, signed zeros, tiny values next to zero, both input positions,
NaNs, infinities, immutable inputs and atomic late rejection. Float/layout
regressions pass 36 tests and 20 subtests; two Python-simulator tests are
deliberately skipped because execution uses Rust. A rehashed instruction-bound
refusal test also passes. All full applications pass fresh native publication,
memory, timing, allocation and resource verification.

Local reproduction and receipts are in the Atlas worktree's
`out/traffic-audit/`: `env.sh`, `compile_layer.py`, `run_layer.sh`,
`compare_layer.py`, `zero-preflight-comparison.json`, and the
`rust-{tinyllama,smolvla}-{baseline-v2,zero-domain}` directories. Each result
binds its source, application, simulator and exact output by SHA-256.
The clean baseline source is retained at `out/traffic-audit/atlas-baseline`.

## Acceptance gate

For every feature, cover multiple shapes, partial tiles, signed zeros, domain
boundaries, malformed inputs and resource refusals. Compare all output words
and protected memory using Rust. Compile and execute both complete retained
layers with matched options and source/model hashes. Report cycles, dispatches,
instructions and DRAM bytes, including regressions. Commit and push only after
the feature's independent checks and full-layer comparisons pass.

## Second implementation: softmax output quantization

Opt into `softmax_quantize` after `softmax_narrowing` and
`softmax_input_widening` in the adjacent-pattern list. A private BF16 probability
result feeding both projections of a static activation quantizer is packed in
its original row program. Dynamic scales, escaping probability tensors and an
explicit host quantizer route retain the separate regions.

The selected softmax computation, BF16 rounding and masks are unchanged. The
native rewrite identifies the final full-tile store on an unconditional chain
of staged blocks; earlier stores to reused output slots remain BF16. It applies
the same raw PACK scale clamp as the standalone quantizer. The Atlas host ABI
reassembles the two physical column halves and publishes the FP8 codes and the
original int32 exponent only after all launches/readbacks succeed. The original
row frame remains allocated, including unused upper output halves. This change
reduces transfers, not the reserved device arena.

This feature requires Merlin commit `dadab98d7` on the companion
`perf/model-traffic-reduction` branch. Merlin's `render_row_softmax_c` now accepts an independently selected output
byte width. It only copies dense rows and trims tails; it makes no numerical or
hardware decisions. Atlas owns matching its quantizer semantics, final-store
proof, physical PACK/layout handling, source binding, and replay verification.

Matched full-layer Rust runs, compared with the zero-admission implementation:

| Layer | Before cycles | After cycles | Saved | Dispatches | DRAM bytes removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| TinyLlama, B1/T64 | 71,519,269 | **71,126,873** | 392,396 (0.549%) | 56 → 55 | 524,320 |
| SmolVLA vision, B1/T1024 | 440,156,657 | **402,703,972** | 37,452,685 (8.509%) | 113 → 112 | 50,331,680 |

The removed traffic is one BF16 write/read pair plus the standalone quantizer's
32-byte exponent publication. Every remaining region output, including the
new fused FP8 codes and exponent, matches its prior byte string. All 54
unchanged TinyLlama launches and 111 unchanged SmolVLA launches retain identical
cycles, instruction histograms, busy counters and transfer counts. Both final
layer outputs match their independently retained arithmetic references exactly:
131,072 and 786,432 BF16 words respectively.

After fusion, TinyLlama reads 66,138,112 bytes and writes 12,819,232; SmolVLA reads
296,982,528 and writes 112,495,168. Their serial same-work bounds are 53,456,785
and 277,233,916 cycles. DMA busy fractions are 82.30% and 74.23%; SmolVLA's busy
fraction decreases while execution gets faster because substantially less DMA
work remains. These are simulator device cycles, excluding host packing and
launch latency, and are not hardware or full-model measurements.

Validation includes 21 fusion tests with eight native/C/Rust execution cases:
FP32 and BF16 score transport, plain/additive-mask/guarded softmax, fully masked
rows, partial rows/columns, both int32 scale extremes, shared-result and dynamic
scale rejection, final-store selection, output alias rejection, late read failure
with unchanged public outputs, and refusal to replay a different output encoding
or exponent. Existing softmax/row-loop regressions add 37 passing tests and two
native subcases; one Python-instruction-model test remains skipped. Merlin's
nine packing tests also exercise 1-, 3-, and 8-byte output elements, tails,
guards and invalid widths. Structure and target-name gates pass. The repository
regex gate reports three pre-existing calls in unchanged
`src/merlin/compile/baremetal_harness.py`; this feature adds no regex calls.

Reproduction artifacts are in `out/traffic-audit/`:
`compile_quantized_softmax_layer.py`, `snapshot-quantize-64m`,
`tinyllama-quantized-softmax`, `smolvla-quantized-softmax`, the corresponding
`rust-*` directories, and `softmax-quantize-comparison.json`. The fixture hashes,
Rust commit/binary and numerical settings are unchanged from the first feature.

The third implementation addresses the repeated input reads in exact bias/residual
kernels. They currently validate a whole chunk, then reload it for computation.
Their application wrapper already stages outputs privately until every status
succeeds. Proving that ownership for a fused validation/computation loop may
remove the same traffic as a projection epilogue without first changing MXU
partitioning or its weight reuse. Raw native calls must retain their existing
failure-publication contract unless explicitly compiled for private results.

## Third implementation: single-pass private exact additions

The exact BF16 add/subtract application wrapper already retains leaf results
in disjoint private scratch and checks every device status before publishing
the logical output. Those application kernels now validate a tile and consume
its operands while they are resident. They do not reload the complete input
for a second computation pass. The vector, pair-row and scalar arithmetic
routes are unchanged, including RNE, signed zero and exceptional-value rules.

An invalid tile makes the status sticky and prevents subsequent result stores.
Earlier private tiles may have been written; the wrapper discards the complete
result on failure. Raw device compilation remains two-pass by default. Opting
into private results is an explicit plan/manifest contract, authenticated in
source replay and canonical materialization rebinding. It cannot be changed by
rehashing only a parent schedule. Legacy raw application artifacts still replay.
This is Atlas device scheduling and ABI work; it requires no new Merlin pass.

Matched Rust results against the softmax-quantization implementation:

| Layer | Before cycles | After cycles | Saved | Dispatches | DRAM reads removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| TinyLlama, B1/T64 | 71,126,873 | **69,857,257** | 1,269,616 (1.785%) | 55 → 55 | 1,638,400 |
| SmolVLA vision, B1/T1024 | 402,703,972 | **375,891,604** | 26,812,368 (6.658%) | 112 → 112 | 34,603,008 |

Every final BF16 word matches the retained independent reference. All 63
TinyLlama and 62 SmolVLA retained region outputs also match their previous byte
strings. The 51/46 unchanged launches retain identical cycles, instructions,
busy counters and transfer counts. Only four TinyLlama and 66 SmolVLA exact-add
launches change. DRAM writes are unchanged. Total instructions are 5,076,293
and 24,121,344; total DRAM bytes are 77,318,944 and 374,874,688.

The focused audit passes 37 tests covering native Rust arithmetic, generated C
publication, canonical materialization and replay. Cases include partial and
multiple tiles, shared operands, add/subtract, both vector-admission settings,
signed zero, subnormals, infinities, early/late invalid inputs, output aliases,
single/multiple dispatches, and execution/image-budget rejection. The broader
exact-float and instruction-bound regressions pass; Python instruction-model
tests remain skipped in favor of Rust execution.

Artifacts: `out/traffic-audit/{tinyllama,smolvla}-private-float-v2`, matching
`rust-*` directories, `private-float-comparison.json`, and the
`private-float-{audit,final-tests,limits}.log` files. Compiler/tool/model options
and reference hashes are unchanged. The serial same-work bounds are 52,495,677
and 256,821,488 cycles; DMA busy fractions are 82.06% and 72.71%.

The fourth implementation uses the unused exact-binary execution budget to reduce sliced dispatches.
The current 131,072-element body is bounded at 76.6M instructions against the
128M application budget. Larger slices still need independent native bounds,
the unchanged 128-KiB IMEM check and atomic publication across all slices.

## Fourth implementation: bounded, equal private slices

Atlas now sizes private binary candidates with a conservative per-tile
instruction envelope instead of scaling the old 1,024-elements-per-million
policy. The final emitted body still needs its independent worst-case native
instruction proof and image-size check. Raw leaf sizing remains unchanged.
The private route has no per-tile route table, so its old 512-tile table limit
does not apply. I/O capacity still bounds every frame.

Merlin's new `uniform_slice_elements` helper prefers equal, aligned slices
only when they preserve the minimum invocation count. It contains no hardware
or numeric policy. This avoids generating another body solely for the tail.
The Atlas candidate capacity is 199,680 elements at the selected 128M budget;
the SmolVLA additions divide into equal 196,608-element bodies. Each actual
body is independently bounded at 114,906,530 instructions and occupies 6,328
bytes of the unchanged 128-KiB IMEM. This requires companion Merlin commit
`6f81ccd9b` on `perf/model-traffic-reduction`.

Saved schedules replay their original checked slice size. Rebinding an older
artifact does not silently reslice its code; malformed, misaligned or oversized
saved chunks refuse, and every native child still undergoes source replay.

| Layer | Before cycles | After cycles | Dispatches | Meaning |
| --- | ---: | ---: | ---: | --- |
| TinyLlama, B1/T64 | 69,857,257 | **69,857,257** | 55 → 55 | Additions already fit in one dispatch. |
| SmolVLA vision, B1/T1024 | 375,891,604 | **375,876,512** | 112 → **90** | Bias launches 54 → 36; residual launches 12 → 8. |

SmolVLA saves only 15,092 device cycles (0.0040%). This is principally a host
dispatch reduction, not a large device speedup; host dispatch latency is not
included in these cycle counts. It executes 24,119,210 instructions and removes
704 bytes of status writes. All final words and all 63/62 retained region outputs
remain exact. Every unchanged launch (55 TinyLlama, 46 SmolVLA) retains its
cycles, instructions, busy counters and transfer counts.

Using the largest candidate without balancing produced extra tail bodies:
checked SmolVLA compilation took 545 seconds. Equal slices reduced this to
358 seconds with the original number of compiler tool calls. The preceding
smaller-slice control took 331 seconds. These are single wall-time observations
on the shared machine, not repeated timing averages or a claimed compile-time
improvement over that control.

Validation includes 16 sizing tests spanning budgets, I/O capacities, full/tail
slices, both native pipelines, mixed admission settings, shared operands,
large private loops, legacy replay and resource refusal. The new Merlin helper
passes 18 tests including exhaustive small partitions and malformed inputs.
The exact-float, C/Rust layout and canonical-materialization regressions pass;
the two Python instruction-model cases remain skipped. Merlin structure and
target-name gates pass; the previously recorded three regex failures remain
in the unchanged upstream file.

Artifacts: `out/traffic-audit/snapshot-balanced-64m`, `compile_balanced_layer.py`,
`{tinyllama,smolvla}-balanced-sizing`, corresponding `rust-*` directories,
`balanced-sizing-comparison.json`, `private-sizing-tests.log` and
`balanced-sizing-replay.log`. The naive larger-slice experiment is retained
separately under `*-private-sizing` for its compile-time comparison.

## Fifth implementation: invocation-local repeated operand cache

Merlin proves when an affine dense read repeats its complete physical input.
The proof uses strides, bounds and unit-extent axes, independently of dtype or
target. It refuses partial source slices, permutations and reversed/strided
reads. Atlas uses that proof for private exact BF16 add/subtract inputs: load
one complete cycle of tiles into VMEM, then reuse those immutable tiles.
The cache initializes on every invocation; it is not persistent across calls.

A period of 768 elements needs three 1,024-element tiles, including rotated
phases. Generated host packing preserves the phase at each slice boundary.
Partial or unprofitable leaves retain dense transport. The candidate shares the
existing 96-tile staging ceiling, and every selected image still passes native
allocation, instruction, memory and timing proofs. Arithmetic, RNE, exceptional
value handling and atomic public output remain unchanged. Saved dense packages
replay as dense; changing only a parent's cache marker cannot relabel its child.
The example simulator harness delegates the new transport geometry check to
Atlas rather than putting cache/VMEM policy in shared Merlin code.
The companion Merlin commit is `719dbbd04` on `perf/model-traffic-reduction`.

| Layer | Before cycles | After cycles | Dispatches | DRAM reads removed |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama, B1/T64 | 69,857,257 | **69,857,257** | 55 → 55 | 0 |
| SmolVLA vision, B1/T1024 | 375,876,512 | **367,647,056** | 90 → 90 | 13,934,592 |

SmolVLA saves 8,229,456 cycles (2.189%) at the cost of 263,268 more scalar/control
instructions for cache selection. Writes do not change. All 62 SmolVLA and 63
TinyLlama retained intermediate outputs and every final BF16 word match their
controls. All 54 unchanged SmolVLA launches retain exact counters. TinyLlama
has no eligible repeated add operand: all generated C, headers, program binaries
and its compiled host library are byte-identical to the preceding control.

An initial TinyLlama run nevertheless produced 1,091 differing final words,
first appearing in one row of the unchanged gate projection. Three reruns,
including full traffic tracing, plus an address/undefined-behavior-sanitized
host run pass exactly. The initial failure is retained and its cause is still
unresolved; it is not counted as a passing run or attributed to this cache.
The accepted comparison uses `rust-tinyllama-cache-census` and
`rust-smolvla-cache-runtime`; the failed run remains `rust-tinyllama-input-cache`.

Directed coverage includes rotated phases, both operand positions, multiple
widths, full and partial tails, cache-disabled controls, signed zeros, tiny
values, invalid-domain rejection, aliases, guards, legacy replay and the largest
admitted cache. The shared indexing proof passes 22 tests. Atlas tests also
check actual Rust DMA instruction reductions against dense execution and reject
mutated transport geometry. The final Atlas/layout/runtime suite passes 47 tests
and 169 subtests; 23 additional exact-float/application regressions pass, with
two Python instruction-model cases skipped. Receipts are in `out/traffic-audit/`, particularly
`input-cache-comparison.json`, `cache-final-tests.log` and
`cache-asan-comparison.log`.

## Current focus: one device dispatch per layer

The user has narrowed the next work to dispatch reduction. Broad traffic,
arithmetic and batching optimizations are deferred. First remove software-budget
slicing where the complete counted body fits the physical frame and IMEM;
then connect device-resident regions and share kernel bodies across stages.

The accepted applications contain 35/25 distinct images totaling 187,424/154,924
bytes (TinyLlama/SmolVLA), against 131,072 bytes of physical IMEM. These totals
count identical images only once. Simple concatenation is insufficient even
before handling host-side packing, layout changes, status publication and
shared-buffer lifetimes. The execution watchdog is a separate software setting;
raising it does not enlarge IMEM or establish that a composed image fits.

### Verified software-budget experiment: 90 to 54 SmolVLA launches

With Atlas `874f6ac` and Merlin `719dbbd04`, select
`--execution-instruction-limit 2100000000` in the otherwise unchanged application
recipe. This exercises the existing counted kernels with a larger software
watchdog; it is not a new optimization pass or a larger hardware instruction
memory. The backend still enforces 131,072 encoded bytes per image.

| Layer | Before launches | After launches | Before cycles | After cycles |
| --- | ---: | ---: | ---: | ---: |
| TinyLlama B1/T64 | 55 | **55** | 69,857,257 | **69,857,257** |
| SmolVLA vision B1/T1024 | 90 | **54** | 367,647,056 | **367,502,579** |

The six bias operations and two residual operations each fit one dispatch,
instead of 36 and eight dispatches respectively. SmolVLA saves 144,477 device
cycles (0.0393%), 184,320 read bytes and 1,152 write bytes. Its instruction count
increases by 3,297 to 24,385,775. This is chiefly a reduction in host launches;
the model does not include host launch latency in these device-cycle totals.
All final words and all 63/62 intermediate outputs match exactly. Unchanged
55/46 program launches retain identical cycles, instructions and traffic.

The largest private-add body covers 3,145,728 elements. Its independent bound is
1,819,874,486 executed instructions for the worst admitted numeric route, but
its image occupies only 6,892 bytes. A larger watchdog does not require larger
IMEM. Different complete tensor sizes do create one extra distinct image:
SmolVLA now has 26 unique images totaling 161,816 bytes. Reusable bodies and a
common device ABI are still necessary for a one-image layer.

The initial larger-budget build hit a buffer-verifier error on an unchanged QK
machine file. That file has the same SHA-256 as the preceding passing control;
16 fresh file-based checks, 12 fresh stdin checks and a complete fresh application
build subsequently passed without changing code or disabling any check. The
first failure remains retained and unresolved, like the isolated runtime outlier
above. These observations do not establish a hardware result.

Artifacts: `compile_dispatch_budget_layer.py`, `tinyllama-dispatch-budget`,
`smolvla-dispatch-budget-v2`, their `rust-*` runs, and
`dispatch-budget-comparison.json` in `out/traffic-audit/`. The failed first build
is retained as `smolvla-dispatch-budget`. All simulator, source and reference
hashes match the preceding cache experiment. The shared-body compiler prototype
uses a separate `build-shared-dispatch/` and was not used for these measurements.
