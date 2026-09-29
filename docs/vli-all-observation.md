# Raw-bit VLI.ALL observation on the selected standalone core

This is a bounded observation for one VLI mode in the hand-authored OOT Atlas
machine dialect. It does not qualify VLI.ROW/COL/ONE, pack/unpack scaling, a
general temporal schedule, or the integrated Atlas SoC.

The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`. Its
`Instructions.scala` assigns VLI.ALL opcode `0x5f` and function `000`;
`IDecode.scala` routes it to `VPU_LI_ALL`. `VectorLoadImm.scala` assigns the
raw 16-bit immediate to every output lane without a floating-point
conversion. `VectorFSM.scala` assigns 64 output rows to VLI.ALL, covering
both 32-row physical BF16 registers in an even/odd pair. The selected public
`baremetal/assembly/vpu_vli.S` also describes VLI.ALL as a full-pair fill.
The inspected `npu_model-atlas` revision
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` implements its VLI.ALL
with a raw `uint16` bit view. That routine was inspected but was not used to
calculate expected outputs.

The OOT `atlas.vli` operation already had a typed state token, conservative
read/write effects, verifier checks for an even pair base and a raw 16-bit
immediate, and selected-RTL word emission. This increment adds a focused
36-word physical stream and independent checks. The stream deliberately
loads unrelated data into `m0/m1`, executes VLI.ALL into `m4/m5`, and stores
both output registers. `test/test_vli_all_reference.py` computes its expected
2,048-byte output by repeating the immediate's little-endian 16-bit encoding
1,024 times; it does not call Torch or the model implementation.

The selected assembler transcription, OOT emitter, and LLVM-object `.text`
words agreed for the 36-word stream. The parser/printer round-tripped the
operation. Negative cases rejected an odd pair base, a pair beginning at
register 63, an out-of-range immediate, and an unknown VLI mode.

Four selected-source-linked standalone `AtlasCore` ARC executions used
immediates `0x3f80`, `0xbf80`, `0x8000`, and `0x7fc1` (positive one, negative
one, negative zero, and a NaN encoding). Every one of the 1,024 BF16 cells
matched the raw-bit expected output in each run, including both register
halves. Each run halted, observed 64 DMA reads and 64 writes, preserved the
unrelated 2,048-byte input and a 32-byte guard, and overwrote both output
preloads. The runs used the selected-source-linked ARC model at the locally
recorded rebuild and ModeLIR's external TileLink driver; they did not invoke
the LLVM function through a C ABI.

The `DELAY 64` after VLI.ALL is diagnostic spacing inherited from the public
unary VPU example, not a proven minimal availability bound. The result is
evidence for this mode, geometry, stream, and four raw immediates only.
