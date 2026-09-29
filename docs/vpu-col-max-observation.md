# Bounded VREDMAX.BF16 layout observation

This is a hand-authored, ACT-independent OOT Atlas diagnostic for selected
`atlas-npu` revision `0079c0541111197741a231c002e3843fa6f545b2` and
inspected `npu_model-atlas` revision
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.

`Instructions.scala` assigns `VREDMAX_BF16` opcode `0x57`, funct7 `0x07`;
`IDecode.scala` selects `VPU_CMAX`. `VectorFSM.scala` recognizes `cmax` as a
column reduction and scans both BF16 banks as 64 rows of 16 physical lanes.
`VectorEngine.scala` uses `PairWiseMax` for its accumulation. The inspected
model's `VREDMAX_BF16.exec` instead reduces each of 32 logical columns of a
32-by-32 BF16 tile. The source reading predicts a layout mismatch; the
bounded execution below tests that prediction.

The [typed 36-word stream](../test/examples/vpu_col_max_pair.mlir) loads a
BF16 register pair and stores both destination halves. Its
[independent selected-assembler transcription](../test/examples/vpu_col_max_pair.S),
OOT emitter, and RISC-V LLVM-object `.text` words agree. Parser/printer and
verifier negatives cover odd and out-of-range pair bases and an unknown mode.

`test/test_vpu_col_max_reference.py` constructs two 32-by-32 finite-normal
panels. Every logical column has a distinct positive maximum in a different
row. The high bank wins every physical lane in panel zero; the low bank wins
in panel two. A separate exact-rational BF16 value decoder computes both the
32 logical-column maxima and the selected RTL's 16 physical-lane maxima.
Both selected-source-linked standalone `AtlasCore` runs matched the 64-by-16
physical-lane result bit for bit and differed from the 32-by-32 model result.
Each run halted, observed 64 DMA reads and 64 writes, preserved all 2,048
input bytes and a 32-byte guard, and overwrote the output preload.
The focused test passed 4/4 methods; after the inventory count was updated,
the full source-linked CTest passed 129/129 Python methods with no skips or
failures. The retained CTest summary is under
`out/qualifications/oot-vpu-col-max-r1/full-source-linked-ctest.log`.

`DELAY 256` is diagnostic slack, not a qualified minimum latency. The tests
cover one register pair, two finite-normal panels, this flat program, and a
standalone core driven by ModeLIR. Exceptional values, arbitrary register
pairs, timing, and integrated SoC behavior remain unqualified. The
intent/RTL layout discrepancy prevents software admission without a reviewed
compatibility decision. Gate D remains blocked.
