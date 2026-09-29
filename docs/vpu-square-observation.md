# Bounded VSQUARE.BF16 observation

This is a selected-RTL standalone-core diagnostic for one unary VPU mode in
the hand-authored, ACT-independent OOT Atlas machine dialect. The selected
`atlas-npu` revision is `0079c0541111197741a231c002e3843fa6f545b2`;
the inspected `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.

`Instructions.scala` and `IDecode.scala` assign `VSQUARE_BF16` opcode `0x57`,
funct7 `0x46`, and the `VPU_SQUARE` dispatch. The inspected model class
instead declares funct7 `0x4e`. The OOT emitter and the selected assembler
use `0x46`; no test assumes that the model's word dispatch is valid for the
selected RTL. `VectorParam.scala::squareCube` explicitly maps input zero,
subnormal, or NaN to positive zero; input infinity or overflow maps to
positive infinity. Normal powers of two square exactly when the result
exponent stays in range. The model's execution routine calls Torch `x * x`;
it was not used as an expected-output oracle.

The [typed 36-word stream](../test/examples/vpu_square_pair.mlir) loads two
1,024-byte BF16 input halves into `m0/m1`, squares them into `m4/m5`, and
stores both halves. Its [assembler transcription](../test/examples/vpu_square_pair.S),
OOT emitter, and RISC-V LLVM-object `.text` words agree. The registered
Atlas-to-LLVM pass emits ordered, side-effecting inline assembly with a memory
clobber. The parser/printer round-trips the typed unary operation. Verifier
negatives reject odd or out-of-range pair bases and an unknown mode.

`test/test_vpu_square_reference.py` uses an independent raw-bit oracle over
zero, signed zero, subnormals, signed infinities, signed NaNs, and normal BF16
powers of two. For the normal subset it computes the exact squared exponent
arithmetically, with explicit underflow and overflow outcomes; it does not
copy the RTL mantissa multiplier or call Torch. Two 1,024-cell panels permute
16 directed input encodings over both physical register halves. Both panels
matched the selected-source-linked standalone `AtlasCore` bit for bit. Each
run halted, observed 64 DMA reads and 64 writes, preserved the complete
2,048-byte input and a 32-byte guard, and overwrote the output preloads.
NaN-to-zero is the sharp semantic discriminator for the selected RTL behavior.

The `DELAY 33` and `DELAY 64` instructions are diagnostic spacing inherited
from public VPU examples, not qualified minimum availability bounds. This
evidence covers this finite 16-code subset, pair geometry, stream, and
standalone core with ModeLIR's external TileLink driver. It does not qualify
normal inputs with nonzero fraction, all exceptional cases, overlapping
execution, integrated SoC behavior, or the LLVM function as a launch ABI.
The mode remains outside a frozen executable software contract. Local test
logs are under `out/qualifications/oot-vpu-square-r1/`.
