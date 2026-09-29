# BF16 VRELU observation on the selected standalone core

This is a bounded observation for the hand-authored OOT reference, not a
qualification of all VPU modes or the integrated Atlas SoC.

The selected `atlas-npu` source is
`0079c0541111197741a231c002e3843fa6f545b2`. Its
`src/main/scala/atlas/scalar/Instructions.scala:144` assigns VRELU opcode
`0x57` and function `0x48`; `IDecode.scala:135` routes that pattern to
`VPU_RELU`. `VectorEngineTop.scala:163-172` requires an even primary source and
even destination for a BF16 pair operation. `VectorEngine.scala:260-266` sends
the selected input vector into `Relu`, and lines 467/489 select its response
for the two VPU pipelines. The selected public
`baremetal/assembly/vpu_unary_simple.S:58-73` uses the same 64-cycle
diagnostic delay and stores both result registers. That delay is not a general
availability bound.

The separately pinned local `npu_model-atlas` revision
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` defines `VRELU_BF16` in
`npu_model/configs/isa_definition.py:632-638` as a BF16 pair read, `torch.relu`,
and BF16 pair write. This model routine was inspected but was **not** called by
the independent expected-value check.

`test/examples/vpu_relu_pair.mlir` hand-authors the physical stream. The
selected assembler independently assembles `test/examples/vpu_relu_pair.S`;
all 36 words equal the OOT emitter output and the RISC-V object words from
`--convert-atlas-to-llvm`. The parser/printer round-trips the operation. The
verifier rejects an odd destination, a pair starting at register 63, and an
unknown mode.

`test/test_vpu_relu_reference.py` loads two 1,024-byte BF16 register halves,
applies VRELU from `m0/m1` to `m4/m5`, and stores both halves. Its test-only
reference maps each selected finite negative BF16 encoding to `+0` and
preserves each selected nonnegative finite encoding bit for bit. Two dense
32-by-32 panels, each containing positive, negative, and positive-zero inputs
in both halves, matched all 1,024 output cells each on the freshly rebuilt
selected-source-linked standalone `AtlasCore` ARC model. Each execution
observed 64 DMA reads and 64 writes; inputs and an output guard were preserved.
Changing the selected instruction to VMOV on the same input preserved the
original negative values and disagreed with the VRELU expectation, so the
result is sensitive to the VPU mode.

The admitted check excludes BF16 subnormals, negative zero, infinities, NaNs,
and broad timing/resource schedules. The stream uses external memory
preloads and an external diagnostic driver; it does not provide an Atlas
launch ABI or an integrated SoC execution result.
