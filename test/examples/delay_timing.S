# Independently selected-assembler transcription of delay_timing.mlir.
ADDI x1, x0, 7
CSRRW x0, 0xC10, x1
DELAY 4
ADDI x2, x1, 1
CSRRW x0, 0xC11, x2
ECALL
