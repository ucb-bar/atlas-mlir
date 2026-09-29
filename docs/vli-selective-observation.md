# Bounded raw-bit VLI.ROW/COL/ONE observation

This is a selected-RTL standalone-core diagnostic for three modes in the
hand-authored OOT Atlas machine dialect. The selected `atlas-npu` revision is
`0079c0541111197741a231c002e3843fa6f545b2`; the inspected
`npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`. No ACT compiler package,
backend, or generated rule is used.

`Instructions.scala` assigns opcode `0x5f` and function codes `001`, `010`,
and `011` to ROW, COL, and ONE. `IDecode.scala` routes them to the corresponding
VPU modes. `VectorLoadImm.scala` converts the signed 16-bit immediate with
`asUInt` and writes those same 16 raw bits. It selects lane zero for COL/ONE
and the five-bit row index zero for ROW/ONE. `VectorFSM.scala` counts 64 writes
for ROW, covering an even/odd BF16 register pair; it counts 32 writes for
COL/ONE, covering one register. The public `baremetal/assembly/vpu_vli.S`
describes the same physical width. The inspected model routines instead
assign the integer immediate numerically into a BF16 tensor before viewing
its bits. For example, raw immediate `0x4000` represents BF16 `2.0`, whereas
numeric BF16 conversion of integer `16384` yields bits `0x4680`. The model
routines therefore were not used as the expected-output oracle.

The typed [36-word stream](../test/examples/vli_selective_pair.mlir) first
loads different 1,024-byte noise panels into `m4` and `m5`, then runs VLI and
stores both registers. A separate [assembler transcription](../test/examples/vli_selective_pair.S)
is checked against the hand OOT emitter and the `.text` words from the
registered Atlas-to-LLVM pass and RISC-V `llc`. The LLVM lowering uses
side-effecting inline assembly with a memory clobber, preserving instruction
order in the emitted stream. The dialect verifier rejects odd ROW pair bases,
out-of-range registers and raw immediates, and unknown modes.

`test/test_vli_selective_reference.py` independently builds exact expected
bytes. ROW places the raw code in all 16 cells of row zero in **both** register
halves and zeros every other row. COL places it in lane zero of every row in
`m5`, zeros the other lanes there, and preserves all of `m4`. ONE places it
only in row zero/lane zero of `m5`, zeros the rest of `m5`, and preserves `m4`.
The reference does not import Torch or call the npu_model execution routines.

Six selected-source-linked standalone `AtlasCore` ARC runs covered the three
modes at raw immediates `0x4000` and `0x8000`. Each run halted and matched all
2,048 output bytes, 2,048 preserved input bytes, and a 32-byte guard. Each
observed 64 DMA reads and 64 DMA writes. Selected-assembler, OOT emitter,
and LLVM-object words matched for all six corresponding streams. The local
diagnostic logs are under `out/qualifications/oot-vli-selective-r1/`; they are
not source artifacts or a published hardware certificate.

The `DELAY 33` and `DELAY 64` instructions are diagnostic spacing inherited
from public VPU examples, not qualified minimum availability bounds. This
tests one 16-lane, 32-row-per-register geometry, two raw values, and the
standalone core with ModeLIR's external TileLink driver. It does not qualify
all immediate/register combinations, overlapping execution, integrated SoC
behavior, or a model compiler. None of the three modes is admitted to a
frozen executable software contract by this observation alone.
