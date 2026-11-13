# Fold dense input layouts into exact-float packing

Opt into `float_input_layout` in `ATLAS_ADJACENT_PATTERNS` to absorb a private
slice, broadcast, reversal or permutation into an exact-float consumer's
existing input packing. The native arithmetic images, logical slice sizes,
rounding and domain checks remain unchanged. No full-tensor layout temporary
is allocated, and no floating-point arithmetic moves to the host.
The shared packing and ownership APIs require
[Merlin 55bbb37f3](https://github.com/ucb-bar/merlin/commit/55bbb37f3) or later.

Merlin owns the reusable transformation contract: `private_operand_layouts`
proves that the view has no external reader and that the boundary
representations agree. `bounded_dense_read_map` proves an identity-written,
byte-preserving dense affine read, including signed-index bounds;
`render_indexed_slice_copy` generates byte packing with full source/destination
span, overflow and overlap checks. The existing complete-permutation API
continues to admit only permutations.

Atlas admits a single exact-float consumer and integrates the packer into its
C ABI. Each scheduled body receives the original source pointer and the
logical starting element. Ordinary operands retain their existing contiguous
slice offsets. All packing checks precede device I/O. Every body still passes
native resource and source-replay verification, and public output is copied
only after every device status succeeds. The original source descriptor is
retained and checked alongside the ordinary exact-float publication.

Eligible arithmetic is BF16 add/subtract, BF16/F32 negation, exact BF16-to-F32
widening and F32-to-BF16 RNE conversion. Shared or published views, unbounded
maps, changed dtypes/policies, non-dense boundaries, unrelated inputs and a
source reused through both the mapped and direct input ports are refused.
Unselected candidates keep the existing separate regions. One input layout is
folded per operation; chains and multiple independently mapped operands are
not yet combined.

## Validation

Shared C tests cover rank changes, broadcasts, slices, negative strides,
permutations, overlapping affine reads and scalar-source broadcasts over raw
1-, 2-, 4- and 8-byte elements. Invalid maps, spans, pointers and aliases must
not write output. Ownership tests retain shared or published values and reject
representation/placement mismatches. The installed core-only wheel and source
archive are checked independently of the checkout.

The native tests execute generated C through Rust for all six arithmetic
operations, both binary input positions, multiple dispatches, tails, sign
bits, subnormals, infinities and permitted NaN payloads. Unread NaNs outside a
slice do not reject it; a NaN in a checked logical input rejects the invocation
without publishing any public output, including after an earlier successful
slice. Output words are compared with an independent exact-arithmetic C
reference. Publication/source mutations and pointer overlap are rejected.

Complete TinyLlama and calibrated SmolVLA vision-layer comparisons are required
before publishing performance results. Device-cycle savings exclude host
packing time; they are not measurements of silicon or full-policy quality.

## Complete-layer results

| Identical fixture and arithmetic | Before cycles | After cycles | Dispatches |
| --- | ---: | ---: | ---: |
| TinyLlama B=1, 64 tokens | 77,504,366 | 76,325,780 | 59 → 56 |
| SmolVLA vision block B=1, 1,024 tokens | 591,239,242 | 566,685,538 | 119 → 113 |

The baseline includes [quantizer permutation packing](quantizer-input-permutation.md).
TinyLlama removes two slices and one broadcast, saving **1,178,586 cycles
(1.52%)** and **819,200 bytes** of device DRAM traffic. SmolVLA removes six bias
broadcasts, saving **24,553,704 cycles (4.15%)** and **27 MiB** of traffic. Every
remaining native image and its executed cycle/instruction count is unchanged.
All **131,072 TinyLlama** and **786,432 SmolVLA** BF16 output words match their
independent original-FX Atlas-arithmetic references. No weights, calibration,
inputs or numerical policies changed; this is a calibrated SmolVLA vision
block, not a full-policy accuracy evaluation.

The final device instruction counts are 7,966,821 and 78,595,862. TinyLlama
reads/writes 69,791,744/13,081,408 bytes; SmolVLA reads/writes
422,723,584/142,379,616 bytes. Their same-work serial-launch lower bounds are
57,020,968 and 424,750,396 cycles; those bounds are not predicted speedups.
Useful MAC utilization is 3.63% and 1.53% of one MXU's peak, respectively.

Runs use Rust `be1b521015ff54ec75699c2d5d63e567ab0ae22a`, CPU, RTL numerics,
`register_rows`, 64 MiB I/O, the 128,000,000-instruction execution budget and
unchanged 128 KiB IMEM cap. Artifacts under `perf-studies/smolvla-dispatch` are
`tinyllama-float-input-layout`, `application-float-input-layout-v2`,
`rust-tinyllama-float-input-layout` and `rust-float-input-layout-v2`. Each run
has an exact comparison, roofline and deletion-only image comparison receipt.

The shared regression suite passes 159 tests. The focused Atlas suite and
existing quantizer/float-publication regressions pass 19 tests, with an
additional final-slice error assertion rerun over all nine new tests. Merlin's
structure and target-independence gates pass. Its repository-wide regex gate
still reports three pre-existing calls in unchanged `baremetal_harness.py`.

An initial SmolVLA compilation stopped at the sizing identity guard, and an
initial simulation stopped at the trace-word accounting guard. Failed artifacts
and logs are retained; neither guard was bypassed. Fresh compilation and a
complete instrumented rerun passed. These intermittent failures remain
unexplained and are not claimed fixed by this optimization.
