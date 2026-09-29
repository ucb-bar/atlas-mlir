# Bounded VCUBE.BF16 observation

This is a selected-RTL standalone-core diagnostic for the cube mode of the
square/cube VPU unit in the hand-authored, ACT-independent OOT Atlas dialect.
The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`; the inspected
`npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.

`Instructions.scala` and `IDecode.scala` assign `VCUBE_BF16` opcode `0x57`,
funct7 `0x47`, and `VPU_CUBE` dispatch. The inspected model class declares
funct7 `0x4f` and implements `x * x * x` with Torch. The selected
`VectorParam.scala::squareCube` retains input sign, maps zero, subnormal, and
NaN to **signed zero**, and maps infinity or overflow to signed infinity.
Neither the model encoding nor its execution routine determines the oracle.

The [typed 36-word stream](../test/examples/vpu_cube_pair.mlir) loads two
1,024-byte BF16 halves into `m0/m1`, cubes them into `m4/m5`, and stores both
halves. Its [assembler transcription](../test/examples/vpu_cube_pair.S), OOT
emitter, and RISC-V LLVM-object `.text` words agree. The registered
Atlas-to-LLVM pass emits ordered side-effecting inline assembly with a memory
clobber. The parser/printer round-trips the typed unary operation; verifier
negatives reject odd and out-of-range pair bases and an unknown mode.

`test/test_vpu_cube_reference.py` uses an independent raw-bit oracle for
signed zeros, subnormals, infinities, NaNs, and normal BF16 powers of two.
For a normal power of two, it computes the mathematical cube by tripling the
unbiased exponent and retaining sign, with explicit underflow and overflow.
It does not copy the RTL mantissa multiplier or call Torch. Two 1,024-cell
panels permute 16 directed input encodings across both physical register
halves. Both panels matched the selected-source-linked standalone `AtlasCore`
bit for bit. Each run halted, observed 64 DMA reads and 64 writes, preserved
the complete 2,048-byte input and a 32-byte guard, and overwrote output
preloads. Positive and negative NaN becoming corresponding signed zeros is
the sharp selected-RTL semantic discriminator.

The `DELAY 33` and `DELAY 64` instructions are diagnostic spacing inherited
from public VPU examples, not qualified minimum availability bounds. This
evidence covers the finite 16-code subset, one pair geometry, this stream,
and the standalone core with ModeLIR's external TileLink driver. Normal
inputs with nonzero fractions, all exceptional interactions, overlapping
execution, and integrated SoC behavior remain unqualified. The mode remains
outside a frozen executable software contract. Local logs are under
`out/qualifications/oot-vpu-cube-r1/`.
