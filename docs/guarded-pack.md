# Guarded BF16 to FP8 conversion proposal

`atlas-opt --lower-atlas-guarded-pack` explicitly selects an alternative implementation of `atlas.virtual_pack_fp8`. It acts before allocation, input-to-DMA conversion and virtual scheduling. Existing pipelines remain unchanged unless the pass is requested.

Only single-block functions are rewritten; multi-block functions and standalone streams are unchanged. The existing virtual verifier permits only i1/i32 entry arguments, prohibits accumulator block arguments and cross-block handle captures, and requires each accumulator to be consumed before block exit. The pass verifies that contract before counting live selected-unit accumulators, so no incoming handle is omitted from its slot accounting. More general CFG resource analysis is outside this proposal.

The pass converts only scale code 127 and a finite BF16 source. It computes `min(max(src, -448), 448)`, loads the tile into a selected MXU BF16 accumulator and reads it out as FP8 with scale code 127. It shares the clamp constants within each block and threads the existing state through the new accumulator operations. It keeps PACK if both selected-unit accumulator slots are live. Existing state, resource, allocation, machine metadata and timing verifiers check the resulting ordinary virtual operations. Contractions and arithmetic policies are unchanged.

The default conversion unit is MXU0, selected from full-clamp measurements on the retained RTL harness. `--lower-atlas-guarded-pack=unit=1` requests MXU1 explicitly. This option selects the conversion engine only; it never moves a contraction. Engine ranking is memory-backend dependent and requires measurement on the intended target.

The selected target's PACK and clamp/copy/pop routes agree on non-NaN BF16 encodings at unit scale; NaNs differ, so the pass deliberately requires the stronger finite premise. Scale codes other than 127 retain PACK. This proposal makes no engine-ranking claim for a different memory backend.

## Source contract

The optional unit attribute `finite_src` on `atlas.virtual_pack_fp8` is an explicit caller precondition. For every execution of that operation, every lane of its exact `src` SSA value must be finite. A source containing this promise has undefined behavior if the caller violates it. The attribute adds no runtime check and does not prove the producer's arithmetic safe. Its scope is this operation's source operand, not another producer, another PACK or a name-based graph match. Frontends must establish the premise analytically or impose and document sufficient public input preconditions; fixture observations cannot establish it.

```mlir
%packed = "atlas.virtual_pack_fp8"(%producer) {finite_src, scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
```

Without that explicit contract, the pass recognizes only finite BF16 splats, mov/ReLU of proven finite values, and min/max with both operands proven finite. It excludes NaN and infinity splats, unknown inputs, block arguments, contractions, reciprocal/exp and overflow-capable arithmetic. Unknown cases retain PACK. The pass does not infer domains from arbitrary attributes, fixture data or SSA spelling. The contract is honored only as an assumption of the source program; the verifier checks its type, not runtime truth.

The proposal uses full clamps even for probabilities and ReLU. Automatic input-domain analysis, range-dependent clamp removal, broader scales and automatic target-dependent engine selection remain follow-up work.
