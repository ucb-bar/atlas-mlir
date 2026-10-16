# Guarded BF16 to FP8 conversion proposal

`atlas-opt --lower-atlas-guarded-pack` explicitly selects an alternative implementation of `atlas.virtual_pack_fp8`. It acts before allocation, input-to-DMA conversion and virtual scheduling. Existing pipelines remain unchanged unless the pass is requested.

Only single-block functions are rewritten; multi-block functions and standalone streams are unchanged. The existing virtual verifier permits only i1/i32 entry arguments, prohibits accumulator block arguments and cross-block handle captures, and requires each accumulator to be consumed before block exit. The pass verifies that contract before counting live selected-unit accumulators, so no incoming handle is omitted from its slot accounting. More general CFG resource analysis is outside this proposal.

The pass converts scale codes 0 through 255 when the exact BF16 source is finite. For codes `s=0..246`, it clamps to positive BF16 bits `((s+8)<<7)|96` and the corresponding negative bound, loads the tile into a selected MXU BF16 accumulator, and reads it out as FP8 with inverse scale code `254-s`. At code 127 these bounds are ±448. For codes 247 through 255, it omits the clamps and uses inverse scale code `254-min(s,254)`; retained hardware codes 254 and 255 alias. Constants are shared only for the same PACK code within each block. The pass threads the existing state through the new accumulator operations. It keeps PACK if both selected-unit accumulator slots are live. Existing state, resource, allocation, machine metadata and timing verifiers check the resulting ordinary virtual operations. Contractions and arithmetic policies are unchanged.

The default conversion unit is MXU0, selected from full-clamp measurements on the retained RTL harness. `--lower-atlas-guarded-pack=unit=1` requests MXU1 explicitly. This option selects the conversion engine only; it never moves a contraction. Engine ranking is memory-backend dependent and requires measurement on the intended target.

The retained converter equations establish equality for all non-NaN BF16 encodings at codes 0..246 with these clamps, and for all BF16 encodings at codes 247..255 without clamps. Low-code clamping changes NaNs, so the pass requires the stronger finite premise. It conservatively keeps that same finite-source requirement at high codes, even though their all-bit equation proof permits a broader rewrite. This proposal makes no engine-ranking claim for a different memory backend. The frozen native PACK lowering and oracle admit only codes 122..128: when this pass retains PACK because a contract or resource is unavailable, downstream native lowering may still reject other codes. The opt-in rewrite does not expand native PACK admission.

## Source contract

The optional unit attribute `finite_src` on `atlas.virtual_pack_fp8` is an explicit caller precondition. For every execution of that operation, every lane of its exact `src` SSA value must be finite. A source containing this promise has undefined behavior if the caller violates it. The attribute adds no runtime check and does not prove the producer's arithmetic safe. Its scope is this operation's source operand, not another producer, another PACK or a name-based graph match. Frontends must establish the premise analytically or impose and document sufficient public input preconditions; fixture observations cannot establish it.

```mlir
%packed = "atlas.virtual_pack_fp8"(%producer) {finite_src, scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
```

Without that explicit contract, the pass recognizes only finite BF16 splats, mov/ReLU of proven finite values, and min/max with both operands proven finite. It excludes NaN and infinity splats, unknown inputs, block arguments, contractions, reciprocal/exp and overflow-capable arithmetic. Unknown cases retain PACK. The pass does not infer domains from arbitrary attributes, fixture data or SSA spelling. The contract is honored only as an assumption of the source program; the verifier checks its type, not runtime truth.

The proposal uses full clamps at codes 0..246 even for probabilities and ReLU. High-code clamp removal follows the complete converter equation proof, not a producer range or sample values. Automatic input-domain analysis, range-dependent clamp removal and automatic target-dependent engine selection remain follow-up work.
