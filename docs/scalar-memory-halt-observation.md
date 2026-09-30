# Selected-core scalar memory and terminal halt observation

The hand-authored [`test_scalar_memory_reference.py`](../test/test_scalar_memory_reference.py)
emits typed Atlas operations for LB, LBU, LH, LHU, LW, SB, SH, and SW. Each
program's words match the selected `atlas-npu/baremetal/assembler.py` and
LLVM-produced RISC-V object bytes. On the selected-source-linked standalone
AtlasCore ARC model, two initial words with different signs are written to VMEM.
Masked byte and halfword stores modify only their addressed lanes. The test
compares seven scalar registers with independently computed sign extensions,
zero extensions, full words, and an adjacent unchanged word.

The terminal negative variant removes the otherwise harmless ADDI x0, x0, 0
between DELAY 8 and ECALL. The selected core then reports ECALL halt reason 2
while `scalar/memLoadPending` is still 1 and the last LW destination remains
zero. With the ADDI, that destination contains the expected adjacent word.
`ScalarCore.scala` forms `ecall_ebreak` from decoded validity without gating it
on `delay_stall`, and forms `halt_now` from `ecall_ebreak`. The test proves this
interaction for the selected program; a compiler scheduler must drain required
scalar readback before termination. DELAY alone immediately before ECALL is
insufficient on this revision.

This is bounded standalone-core evidence. It does not qualify all 12-bit byte
offsets, alignment behavior, load/store hazards, program termination sequences,
or integrated-SoC execution. The eight mode rows remain software-unadmitted.
