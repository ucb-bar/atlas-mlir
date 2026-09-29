# E8M0 pack, unpack, and FP8 consumption observation

This is bounded evidence for selected `VFP8PACK` and `VFP8UNPACK` modes in
the hand-authored OOT reference, not a block-scaling or full numerical policy.

The selected RTL revision is `atlas-npu`
`0079c0541111197741a231c002e3843fa6f545b2`.
`ScalarCore.scala` reads the scale register selected by `vs1[4:0]`.
`VectorEngine.scala:53-60` maps E8M0 code `c` to exponent shift `c-127`.
`FP8Pack.scala` subtracts that shift from the BF16 exponent and rounds its
mantissa to E4M3; `FP8Unpack.scala` adds the shift back. Thus code 128 means
scale factor 2 and code 126 means factor 1/2. Code 127 is unit scale.

`VectorFSM.scala` streams all 32 physical rows of the first BF16 register,
then all 32 of the next. `FP8Pack.scala` joins **successive read rows** into
each 32-byte FP8 row: packed row 0 contains physical BF16 rows 0 and 1 of
the first register. `FP8Unpack.scala` expands each FP8 row into two
successive BF16 output rows, reversing this arrangement in the tested
round trip. The separately pinned local `npu_model-atlas` revision
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` uses the same `2^(c-127)`
scale convention but `VPACK_BF16_FP8.exec` concatenates the two BF16 register
halves **by logical row**, and `VUNPACK_FP8_BF16.exec` splits by columns.
Those source-level layouts differ. A pack/unpack round trip can return the
input under either layout, so it cannot resolve the difference by itself.

The typed 65-word `test/examples/vpu_e8m0_pack_chain.mlir` loads a BF16
pair, sets scale register 3, packs to one FP8 register, unpacks back to a
BF16 pair, then uses the packed FP8 register for both MXU1 weights and
activations. It stores packed bytes, both unpacked halves, and both MXU1
output halves in nonoverlapping DRAM regions. The source words match the
selected independent assembler and LLVM RISC-V object, including SELI,
PACK, and UNPACK words. A code-126 mutation changes only the SELI word.
Invalid BF16 source/destination pair and scale-register 32 are rejected.

`test/test_vpu_e8m0_pack_reference.py` uses exact rational decoding and an
E4M3 value map on a deliberately restricted set of exactly representable
BF16 powers of two and zero. It independently predicts every packed byte in
the selected physical row order, the unpacked pair, and all 1,024 MXU1
BF16 output cells. It also constructs the model-style same-row concatenation
and confirms that its first packed row differs. The expected values do not
come from `npu_model`, Torch, or the OOT emitter.

Two selected standalone `AtlasCore` runs passed, for E8M0 codes 128 and
126 with different BF16 panels. Each checked 1,024 packed FP8 bytes, 2,048
unpacked BF16 bytes, 2,048 MXU1 output bytes, 64 DMA reads, 160 DMA writes,
both preserved input tiles, and a DRAM guard. The fixed delays and preloaded
reset-entry program are diagnostic only. General FP8 rounding, saturation,
NaNs, subnormals, scale register lifetimes, alternative schedules, and
integrated hardware execution remain unqualified. Neither mode is software
admitted from these two cases.
