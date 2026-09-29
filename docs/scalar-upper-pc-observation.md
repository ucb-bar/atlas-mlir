# Selected-core LUI and AUIPC PC units

The selected RTL revision `0079c0541111197741a231c002e3843fa6f545b2`
uses an instruction **word index** for `PcControl.io.s1_pc`. In
`ScalarCore.scala`, `OP1_PC` feeds that index directly to the ALU. AUIPC
therefore adds the word index to its shifted upper immediate. A typed
three-word program with `ADDI` at word zero and `AUIPC` at word one produced
`0x90000001` for upper immediate `0x90000`. The selected assembler and LLVM
object emitted the same words as the hand OOT emitter. The selected-source-linked
standalone `AtlasCore` reproduced this result. LUI in the same position
produced `0x90000000`.

The inspected `npu_model-atlas` `AUIPC.exec` uses `state.pc -
PIPELINE_LATENCY * 4`, a byte-PC formulation that would produce
`0x90000004` at this instruction position under its usual execution state.
That is a source discrepancy, not a reason to change the selected RTL result.
The selected executable compatibility contract uses the RTL word-PC behavior.
The [bounded test](../test/test_scalar_upper_reference.py) also checks upper
immediate `0xfffff`, output register contents, selected assembler words, and
LLVM object words. This does not qualify arbitrary programs, calls, or the
integrated SoC timing.
