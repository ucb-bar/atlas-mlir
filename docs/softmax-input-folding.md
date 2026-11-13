# Private softmax input widening

Select `softmax_input_widening` in `ATLAS_ADJACENT_PATTERNS` to consume original
BF16 scores when a private exact BF16-to-f32 cast feeds a selected softmax.
It composes with `softmax_narrowing` to keep both boundaries in BF16.

Merlin's `split_private_bf16_widening` proves the dense identity mapping,
exact source conversion, unchanged consumer boundary and private ownership.
Merlin's reusable mask encoder accepts raw BF16 score bytes as an explicit
transport option. Atlas proves that direct code validation preserves the
selected bounded softmax's arithmetic and input rejection behavior, then binds
those bytes to its existing native program. No target fact belongs in Merlin.

Exact widening followed by BF16 RNE returns every admitted score code,
including signed zeros. Both paths reject nonfinite and out-of-domain scores
before device I/O. Additive-mask sentinel checks, affine mask indexing,
Boolean masks, guarded empty rows and output publication remain unchanged.
Input/output/workspace overlap checks use the selected boundary's actual byte
width. The default f32 transport remains available.

## Complete-layer comparison

| Fixture | Before cycles | After cycles | Before dispatches | After dispatches |
| --- | ---: | ---: | ---: | ---: |
| SmolVLA vision block, B=1, T=1024, width 768 | 1,025,795,466 | 881,588,507 | 156 | 151 |
| TinyLlama decoder, B=1, T=64, width 2048 | 82,772,819 | 82,772,819 | 59 | 59 |

The SmolVLA saving is **144,206,959 cycles (14.06%)**, entirely from the
removed score-widening region. Its unchanged softmax still takes 24 launches
and 180,322,848 cycles. TinyLlama has no additional eligible boundary.
Every output word matches the independent original-FX Atlas-arithmetic
reference: 786,432 for SmolVLA and 131,072 for TinyLlama. These are Rust model
measurements, with RTL numerics and `pack_layout=register_rows`, not hardware
measurements or a new numerical policy.

Both runs retain the same 64-MiB I/O window, 128,000,000-instruction execution
budget, 128-KiB instruction image limit, panel-cache
policy and optimized native pipeline. The SmolVLA calibration scope and
quality limitation are documented in [dispatch-reduction.md](dispatch-reduction.md).

The integration artifacts under `perf-studies/smolvla-dispatch` are
`application-private-inputs-v2`, `rust-private-inputs-v3`,
`tinyllama-private-inputs` and `rust-tinyllama-private-inputs`. Each comparison
binds the execution and independent reference by hashes. The simulator is
`be1b521015ff54ec75699c2d5d63e567ab0ae22a`; both output hashes remain identical
to the preceding complete-layer comparison.

Directed checks cover all 65,536 BF16 input encodings against exact widening
and the original encoder, plain/masked/guarded softmax, multidimensional and
partial tiles, escaping casts, changed semantics, input/output overlap,
finite sentinels, empty rows and atomic rejection of invalid late inputs.
Generated C is executed through Rust with independent expected probabilities
and unchanged-memory checks.
