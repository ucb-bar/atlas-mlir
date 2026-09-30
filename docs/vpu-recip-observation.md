# Bounded VRECIP.BF16 observation

This is one ACT-independent hand OOT diagnostic on the selected standalone
`AtlasCore`. The selected RTL is `atlas-npu`
`0079c0541111197741a231c002e3843fa6f545b2`; the inspected model is
`npu_model-atlas` `5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.

`Instructions.scala` and `IDecode.scala` bind `VRECIP_BF16` to opcode `0x57`,
funct7 `0x41`, and `VPU_RCP`. `VectorEngine.scala` sends each 16-lane physical
row to `Rcp.scala`. That lane box uses `RcpLUT` in the selected Atlas
`sp26-fp-units` dependency, pinned by the selected Atlas commit at `9a0cc09c`.
The LUT is an approximation for nonzero fractions. Its conversion maps zero
and subnormal inputs to signed infinity, infinity and NaN inputs to signed
zero, and a normal power whose reciprocal exponent reaches zero to signed
zero. The inspected model instead calls `torch.reciprocal` on a BF16 pair; it
was not used as an expected-output oracle.

The [typed 36-word stream](../test/examples/vpu_recip_pair.mlir) loads two
1,024-byte BF16 input halves into `m0/m1`, computes `m4/m5`, and stores both
halves. Its [selected-assembler transcription](../test/examples/vpu_recip_pair.S),
OOT emitter, and LLVM RISC-V object `.text` agree word for word. The
Atlas-to-LLVM pass produces ordered side-effecting inline assembly with a
memory clobber. The parser/printer round-trips the unary operation; verifier
negatives reject odd or out-of-range pair bases and an unknown mode.

The [independent bounded test](../test/test_vpu_recip_reference.py) computes
reciprocals of exact BF16 powers of two from their exponents, with the
selected boundary rules above. It tests 22 input encodings, including both
signs of zero, subnormal, infinity, NaN, finite 1/2/4 and their reciprocals,
and the smallest and largest normal powers. Two permuted 1,024-cell panels
exercise both physical register halves. Both selected-source-linked ARC runs
matched all 2,048 output bytes. Each halted, observed 64 DMA reads and 64
writes, preserved all 2,048 input bytes and a 32-byte guard, and overwrote
the output preloads. The focused suite passed 4/4 methods with no skips.
After the ledger update, the full source-linked CTest passed 141/141 Python
methods with no skips or failures in 249.25 seconds. The source-bound
99-mode checker passed with 54 bounded modes, 0 software-admitted modes,
and 99 remaining full-qualification blockers.

The `DELAY 33` and `DELAY 64` instructions are diagnostic spacing, not
qualified minimum latency. This result does not cover normal inputs with
nonzero fractions, the full exceptional domain, arbitrary register pairs,
overlapping operations, or integrated SoC execution. It does not establish a
callable LLVM launch ABI or close gate D. Local run logs are under
`out/qualifications/oot-vpu-recip-r1/`.
