# Independent selected-assembler transcription of jalr_load_delay.mlir.
ADDI x2, x0, 11
SW x2, x0, 0
DELAY 4
ADDI x1, x0, 8
LW x1, x0, 0
DELAY 8
JALR x10, x1, 0
ADDI x12, x0, 1
ADDI x12, x0, 2
CSRRW x0, 0xC10, x12
ECALL
ADDI x12, x0, 3
CSRRW x0, 0xC10, x12
ECALL
