# Independently assembled counterpart of jal_direct_target.mlir.
# The selected assembler takes a label in word-index space for JAL.
ADDI x12, x0, 0
JAL x10, target
ADDI x12, x0, 1
ADDI x12, x0, 2
ADDI x12, x0, 4
target:
ADD x11, x10, x12
CSRRW x0, 0xC10, x11
ECALL
