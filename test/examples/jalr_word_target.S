# Selected-assembler transcription of jalr_word_target.mlir.
# Selected core PC is a word index: 6 + (-1) reaches odd word index 5.
ADDI x1, x0, 6
JALR x10, x1, -1
ADDI x12, x0, 1
ADDI x12, x0, 2
ADDI x12, x0, 3
ADD x11, x10, x12
CSRRW x0, 0xC10, x11
ECALL
