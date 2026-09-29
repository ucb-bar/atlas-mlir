# Independent selected-assembler transcription. Each label offset is in
# instruction words; assembler doubles it into the encoded byte displacement.
ADDI x10, x0, 17
ADDI x11, x0, 34
BEQ x0, x0, after_forward
CSRRW x0, 0xC10, x10
CSRRW x0, 0xC11, x11
after_forward:
ADDI x12, x0, 0
ADDI x13, x0, 0
ADDI x14, x0, 3
ADDI x15, x0, 0
loop:
ADDI x12, x12, 1
BLT x12, x14, loop
ADDI x13, x13, 1
ADDI x15, x15, 1
ECALL
