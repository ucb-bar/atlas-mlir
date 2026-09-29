# Bounded VREDMIN.BF16 layout discrepancy

This is a selected-RTL standalone-core diagnostic for BF16 column minimum
in the hand-authored, ACT-independent OOT Atlas dialect. The selected
`atlas-npu` revision is `0079c0541111197741a231c002e3843fa6f545b2`;
the inspected `npu_model-atlas` revision is
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d`.

`Instructions.scala` fixes opcode `0x57` and funct7 `0x05`; `IDecode.scala`
selects `VPU_CMIN`. The architectural specification and model describe one
32-by-32 BF16 tile occupying two physical register banks, with a separate
minimum for every one of its 32 logical columns. The selected RTL's
`VectorFSM.scala` instead toggles the input bank every 32 rows over a
64-row accumulation pass. `VectorEngine.scala` feeds one 16-lane
`PairWiseMin` accumulator, then the FSM broadcasts its 16 result lanes to
both destination banks. This source reading predicts a 64-by-16 physical
column reduction; it does not settle whether the RTL or the architectural
intent should define a future executable compatibility contract.

The [typed 36-word stream](../test/examples/vpu_col_min_pair.mlir) loads a
32-by-32 architectural BF16 tile into a register pair and stores both output
halves. Its [selected-assembler transcription](../test/examples/vpu_col_min_pair.S),
OOT emitter, and RISC-V LLVM-object `.text` words agree. The registered
Atlas-to-LLVM pass emits ordered side-effecting inline assembly with a memory
clobber. Parser/printer round-trip and verifier negatives cover odd and
out-of-range pair bases plus an unknown reduction mode.

`test/test_vpu_col_min_reference.py` builds two finite-normal panels with a
unique negative winner in each of 32 architectural columns, located in
different rows. An independent exact-rational oracle computes both results:
32 separate architectural minima and 16 physical minima over the 64 rows of
each physical lane. In panel zero, the high bank wins every physical lane;
in panel two, the low bank wins. The initial direct architectural comparison
failed on both panels, and the failure logs are retained. The discriminating
follow-up matched the 64-by-16 result bit for bit on the selected-source-linked
standalone `AtlasCore` and rejected the 32-by-32 result in both panels.
Each run halted, observed 64 DMA reads and 64 writes, preserved the full
2,048-byte input and a 32-byte guard, and overwrote output preloads.

`DELAY 256` is diagnostic slack for the two-pass column operation, not a
qualified minimum availability bound. This evidence covers one register
pair, the finite-normal panels, this program, and the standalone core with
ModeLIR's external TileLink driver. The intent/RTL layout mismatch, general
numerical behavior, temporal scheduling, and integrated SoC execution
remain unresolved. `VREDMIN_BF16` stays outside a frozen executable
software contract. Local logs are under `out/qualifications/oot-vpu-col-min-r1/`.
