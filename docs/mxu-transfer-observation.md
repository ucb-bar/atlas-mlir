# Bounded MXU transfer observation on the selected core

The hand-authored OOT dialect now has a selected-core test for all ten MXU
transfer decoder modes: weight push, FP8/BF16 accumulator push, and FP8/BF16
accumulator pop on each unit. This is independent of ACT and does not establish
the full Atlas target compiler or an integrated SoC qualification.

The test is [`test_mxu_transfer_reference.py`](../test/test_mxu_transfer_reference.py).
It builds twelve typed stream variants: three push kinds × two units × two
physical slots, then executes sixteen payload cases. Both accumulator programs
include the corresponding readout. Every
program's emitted words match an unmodified LLVM RISC-V object's text words;
the MXU words also match the selected Atlas assembler. Parser/printer
round-trips pass.

Each program runs on the selected-source-linked standalone `AtlasCore` ARC
model for `atlas-npu` revision `0079c0541111197741a231c002e3843fa6f545b2`.
Weight pushes use a full 1 KiB byte pattern and inspect all 1,024 weight
register bytes in the selected unit/slot; the other weight slot remains zero.
BF16 accumulator pushes use nonuniform full-pair BF16 panels, including signs
and signed zero. They compare the 2 KiB physical accumulator buffer with an
independently constructed row/half layout and compare the two external BF16
outputs bit for bit. FP8 accumulator pushes use finite, exactly representable
powers of two with both signs; the physical BF16 buffer is compared against an
explicit FP8-to-BF16 table before the FP8 readout is checked. Pop uses an
explicit unit E8M0 scale (`SELI e3, 127`). All programs preserve their DRAM
inputs and an unrelated guard. Read/write beat counts are checked.

An additional four FP8 accumulator runs place every 8-bit input code four
times across the tile. Their physical BF16 buffer is checked against a
separate bit-field conversion with directed normal, subnormal, signed-zero,
and NaN anchors. These runs do not assert a full-domain FP8 readout result:
the FP8 pop conversion remains qualified only on the finite normal panel.

The selected RTL computes ECALL halt from decoded validity even when DELAY is
stalling. A weight push followed by `DELAY 64; ECALL` can halt before all
rows are written. The test includes a scalar `ADDI` after DELAY to let the
stall drain. `atlas-emit` and `--verify-atlas-machine-stream` now reject a
trap directly after a nonzero DELAY. This is a bounded safeguard, not a
general asynchronous completion protocol.

The test covers two units, both slots, and the stated payload panels. It does
not establish every FP8 readout code, exception handling outside the inspected
flush policy, every scale code, concurrent
MXU/VPU/LSU access, all physical register addresses, dynamic DMA completion,
or full timing. The selected ARC binary and state manifest hashes match the
previously recorded selected-source build receipt, but that build chain was
not independently reattested in this test run. The ten rows move from zero
to bounded standalone-core evidence,
bringing the mode numerator to **99/99**. All 99 remain unadmitted and blocked
for the full D gate.
