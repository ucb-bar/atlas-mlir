# Fold private permutations into quantizer input packing

Opt into `quantizer_input_permutation` in `ATLAS_ADJACENT_PATTERNS` to absorb
eligible input permutations into exact activation quantization. The compiler
replaces the quantizer's existing dense input copy with indexed byte packing.
It does not allocate another full-tensor intermediate or change the native
quantizer image. The shared packing helper requires
[Merlin f94b0b14d](https://github.com/ucb-bar/merlin/commit/f94b0b14d) or later.

Merlin's existing `private_operand_permutations` proves that the layout result
is private to the consumer and that the boundary representations agree. Its
`permutation_axes` checks a complete, offset-free permutation with unchanged
element type and numerical policy. The reusable `render_permutation_slice_copy`
helper generates checked byte copies from the original dense tensor into each
consumer slice. All of this belongs in Merlin and contains no target facts.

Atlas admits its exact activation-quantizer consumer and supplies the native
ABI integration. The selected quantizer schedule still owns the logical shape,
static exponent or dynamic reduction groups, slice boundaries, output order,
instruction budget and memory limits. Each body's input context carries the
logical starting element; the wrapper passes the original input pointer, and
the packer maps that slice back to source coordinates. The ordinary native
compilation and source replay checks remain mandatory.

The optimization refuses escaping or shared layout results, incomplete maps,
slices, broadcasts, changed dtypes/policies, non-dense boundaries and unrelated
ports. Unsupported candidates retain the original layout and quantizer choices.
Generated C validates full input/output/workspace spans and aliases before
packing or device I/O; output publication remains atomic across all slices.
No floating-point arithmetic moves to the host.

## Validation

The shared C helper is tested over matrix transposes, batch/head swaps, cyclic
axis permutations, singleton axes, several element widths and arbitrary raw
bit patterns. Invalid slices, overflowing pointers and overlapping buffers
must leave outputs untouched.

The Atlas tests compile and execute generated C through Rust for static,
head, token and head-block quantization. They cover multiple bodies, nonzero
slice offsets, differing per-row scales, all exceptional BF16 categories,
source/output alias rejection, output guards and changed publication metadata.
Every code and exponent is compared with the original quantizer's independent
arithmetic reference. Graph selection retains a shared or published view.

## Complete-layer comparison

| Identical fixture and arithmetic | Before cycles | After cycles | Dispatches |
| --- | ---: | ---: | ---: |
| TinyLlama B=1, 64 tokens | 77,504,366 | 77,504,366 | 59 → 59 |
| SmolVLA vision block B=1, 1,024 tokens | 606,944,206 | 591,239,242 | 122 → 119 |

The baseline includes [private staging and immutable input caching](private-staging.md)
and Nicolas's existing common-value elimination for SmolVLA. The new fold
removes three Q/K/V layout launches, each costing 5,234,988 cycles:
**15,704,964 cycles (2.59%)** and **9 MiB of device DRAM traffic** saved.
Every remaining SmolVLA image and its cycle/instruction counts are unchanged;
all 59 TinyLlama images and counts are unchanged. TinyLlama has no eligible
immediate permutation-to-quantizer edge in this captured layer.

All 786,432 SmolVLA and 131,072 TinyLlama BF16 output words match their
independent original-FX Atlas-arithmetic references. The same weights, inputs,
calibration and arithmetic policies are retained. This evaluates the calibrated
SmolVLA vision block, not full-policy quality.

SmolVLA now reads 436,879,360 bytes and writes 156,535,392 bytes on the device.
Its same-work serial-launch bound is 443,918,561 cycles, and useful MAC
utilization is 1.46% of one MXU's peak. The bound is not a speedup prediction.
These are Rust CPU device cycles, excluding host packing and silicon timing;
no claim about measured host latency is made.

The runs use Rust `be1b521015ff54ec75699c2d5d63e567ab0ae22a`, RTL numerics,
`register_rows`, 64 MiB I/O, a 128,000,000-instruction execution budget and the
unchanged 128 KiB IMEM cap. Artifacts under `perf-studies/smolvla-dispatch` are
`application-quantizer-permutation`, `tinyllama-quantizer-permutation`,
`rust-quantizer-permutation` and `rust-tinyllama-quantizer-permutation`.

The focused and existing quantizer suite passes 39 tests and 14 subtests.
Merlin's packing/layout/ownership suite passes 132 tests. Its wheel and source
archive contain the checked helper, and the wheel imports and runs outside the
checkout with core dependencies only. Structure and target-independence gates
pass; the repository-wide regex gate still reports three pre-existing calls
in unchanged `baremetal_harness.py`.
