# Bounded VMIN.BF16 observation

This is a selected-RTL standalone-core diagnostic for pairwise BF16 minimum
in the hand-authored, ACT-independent OOT Atlas dialect. The selected
`atlas-npu` revision is `0079c0541111197741a231c002e3843fa6f545b2`;
the inspected `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.

`Instructions.scala` fixes `VMIN_BF16` opcode `0x57` and funct7 `0x04`;
`IDecode.scala` selects `VPU_PAIRMIN`. `PairwiseMin.scala` invokes
`VectorParam.scala::compareReturnMin`, which transforms each 16-bit BF16
encoding into an unsigned sort key by complementing negative encodings or
flipping the positive sign bit, then chooses the lower key. It compares raw
encodings, including NaNs. The inspected model uses Torch `minimum`, whose
source is a numerical operation; its output was not used as the oracle.

The [typed 49-word stream](../test/examples/vpu_min_pair.mlir) loads two
BF16 register pairs, computes one output pair, and stores both 1,024-byte
halves. Its [selected-assembler transcription](../test/examples/vpu_min_pair.S),
OOT emitter, and RISC-V LLVM-object `.text` words agree. The registered
Atlas-to-LLVM pass emits ordered side-effecting inline assembly with a memory
clobber. Parser/printer round-trip and verifier negatives cover odd and
out-of-range pair bases plus an unknown binary mode.

`test/test_vpu_min_reference.py` computes expected words with a small
independent integer-order oracle. Two 1,024-cell panels combine directed
signed-zero, NaN, infinity, and sign cases with deterministic sampled raw
16-bit encodings. Both panels matched the selected-source-linked standalone
`AtlasCore` bit for bit. Each run halted, observed 128 DMA reads and 64
writes, preserved both complete 2,048-byte inputs and a 32-byte guard, and
overwrote output preloads. Negative NaN bits outranking finite positive
values and negative zero outranking positive zero are sharp discriminators.

The diagnostic `DELAY 33` and `DELAY 64` instructions follow the public VPU
example; they are not qualified minimum availability bounds. This evidence
covers sampled raw encodings, one pair geometry and stream, and the standalone
core with ModeLIR's external TileLink driver. In-place use, general temporal
behavior, and integrated SoC execution remain unqualified. The mode remains
outside a frozen executable software contract. Local logs are under
`out/qualifications/oot-vpu-min-r1/`.
