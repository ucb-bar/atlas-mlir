# Raw PACK scale domain

The raw virtual PACK interface admits all 256 ERF byte codes, including 119. For raw code `c`, the effective exponent is `s = min(c - 127, 127)`. VPU PACK divides by `2^s`; MXU FP8 pop multiplies by `2^s`. Code 0 means `2^-127`, not zero. Codes 254 and 255 alias exponent 127. Neither consumer treats scale byte 255 as NaN. Exponents outside `[-127, 127]`, non-byte codes, and incorrectly typed attributes are not admitted.

The mandatory PACK scale-domain test regenerates model receipts for all 256 codes in the current build. Emitted program hashes can change with later scheduling and allocation. The checked-in hardware PACK test has cases 126 and 128, with powers of two and zero. No additional RTL execution is claimed here. Hardware status remains [Limited](hardware-status.md).

## Source evidence

RTL paths below are relative to Atlas RTL revision `2ae0bef209df6db78c3de18e8f651bb43855cce9`. The floating-point dependency is `9a0cc09c41ab918a3548580185f39cca8d559e0d`. Model paths are relative to unmodified npu_model `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`. The older hardware-status revision remains the authority for historical hardware runs.

The inspected scale decoders, PACK, UNPACK, MXU helpers, scalar scale writes, and both MXU sequencers are byte-identical to the compiler's selected RTL revision `0079c0541111197741a231c002e3843fa6f545b2`. Both RTL revisions pin the same floating-point dependency, whose inspected output converter also matches that pin. Thus the source proof applies to the selected compiler target as well as the live checkout.

| ID | Source | Established behavior |
| --- | --- | --- |
| S1 | RTL `src/main/scala/atlas/scalar/ScalarCore.scala:461-464`; `vector/VectorEngine.scala:53-59`; `mxu/FPUtils.scala:146-156` under `src/main/scala/atlas/` | ERF writes retain eight bits; signed exponent clamps at +127, so 255 aliases 254. |
| S2 | RTL `src/main/scala/atlas/vector/laneBoxes/FP8Pack.scala:48-95` | PACK subtracts the exponent shift, handles exceptional inputs, rounds significands, flushes, and clamps. |
| S3 | Dependency `src/main/scala/sp26-fp-units/OutputConvStage.scala:38-118`; RTL `src/main/scala/atlas/mxu/FPUtils.scala:335-346` | Both MXU units use a converter that adds the exponent shift and can emit reserved FP8 codes. |
| S4 | Model `npu_model/configs/rtl_math.py:63-110`; `numerics.py:70-83`; `isa_definition.py:577-584,901-926` | RTL numerics and actual instruction dispatch implement these scale directions and special cases. |
| S5 | RTL `src/main/scala/atlas/vector/laneBoxes/FP8Unpack.scala:69-96` | Unit UNPACK decodes PACK's output alphabet exactly. |

Both PACK and FP8 pop flush BF16 signed zero, subnormals, and NaNs to positive FP8 zero. BF16 infinities saturate to signed finite maximum, `0x7e` or `0xfe`, at every scale. For normal inputs, both round the eight-bit significand to four bits using nearest-even, then apply exponent carry. A final exponent below -6 flushes to positive zero; above 8 saturates to signed 448. Carry can round an input just below minimum normal up to minimum normal. Neither converter produces E4M3 subnormals.

PACK also clamps final exponent 8 and fraction 7 to signed 448. FP8 pop emits `0x7f` or `0xff` for that case, representing rounded magnitude 480 with reserved E4M3FN codes. Values rounding past that bin saturate back to 448. The converter's header comment about never generating NaNs does not describe this implementation case. Compare raw bytes: UNPACK flushes reserved codes and FP8 subnormals to signed BF16 zero, so decoded comparisons can erase differences. For other inputs, UNPACK multiplies by its scale, flushes BF16 exponent at or below zero to signed zero, and clamps exponent at or above 255 to signed finite BF16 maximum.

For an unchanged representation `real = decode(fp8) * 2^t`, literal PACK uses exponent `t`, and direct pop uses exponent `-t`. On canonical scale bytes 0..254 the inverse raw code is `254 - c`; including the alias it is `254 - min(c, 254)`. This relationship does not authorize changing weight bytes or exponents and does not remove the reserved-code difference.

## Existing guarded conversion

The finite guarded pass already admits all 256 codes. It proves source finiteness, clamps to `448 * 2^s` when `c <= 246`, copies BF16 to an accumulator, and pops at inverse scale. At code 119 the bound is BF16 `0x3fe0`, or 1.75, and inverse code is 135. Without the clamp, exactly 30 finite BF16 words differ from PACK: `0x3fe9..0x3ff7` and their negative counterparts produce reserved pop codes. The existing clamp fixes these cases, but maps all 254 NaN words differently from PACK. Its finite proof must remain. For codes 247..255 no finite BF16 value reaches this overflow boundary, so no clamp is needed.

Unknown boundary inputs and overflow-capable arithmetic generally fail that finite proof. Their existing generic route performs literal PACK, unit UNPACK, BF16 accumulator copy, and unit FP8 pop. PACK first converts exceptional inputs; its 239-code output alphabet is positive zero, `0x08..0x7e`, and `0x88..0xfe`. Unit UNPACK and unit pop reproduce those bytes exactly. This route now admits all raw scale bytes. When both units have all accumulator slots live, literal PACK and the existing scalar interleave remain the fallback, with the same widened admission.

## Execution evidence and scope

End-to-end evidence from an independent reference covers scale bytes 117 through 126 on a single PACK operation; it does not qualify a complete graph, model, or wider scale domain.

The mandatory [scale-domain test](../test/test_pack_scale_domain.py) checks all 65,536 BF16 encodings at every raw code against an independent scalar oracle and actual `RtlNumerics.to_fp8`: 16,777,216 model comparisons. It also executes 512 independently decoded emitted programs in npu_model with `numerics="rtl"`: every code through both scalar fallback and default round trip, 1,024 directed lanes per program, and a complete 66,624-byte DRAM comparison covering input and output guards. Cases include scale-relative ties, carry, flush boundaries, saturation, NaNs, infinities, signed zeros, and subnormals. Invalid -1 and 256 codes remain rejected. Optional `ATLAS_PACK_SCALE_RECEIPT` records per-code output and program hashes and cycles.

The pre-existing guarded exactness test additionally checked 65,280 finite BF16 words at all 256 codes through actual min/max and MXU conversion, totaling 33,423,360 comparisons. An all-input probe checked literal PACK followed by unit UNPACK and unit pop for all 16,777,216 input-scale pairs. Existing evaluator arithmetic tests check FP8 pop against the pinned arithmetic fixture at every raw scale, including reserved-code cases. These are software execution results, not new hardware runs.

Raw virtual/machine PACK and its verification reference admit all byte codes. The native exact-normal descriptor keeps its narrower `122..128` ABI and attestations. The explicit native rounded policy uses the versioned `clamp-0-255-v1` name and admits the same raw byte domain, including code 119 and the 255 alias, while permitting PACK rounding, flush, and saturation. That rounded policy does not provide an exact-normal guarantee. No prequantized weights, exponents, downstream folds, or kit artifacts are changed.
