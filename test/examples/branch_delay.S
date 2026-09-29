# Independently assembled counterpart of branch_delay.mlir.
ADDI x10, x0, 0
BEQ x0, x0, target
ADDI x10, x0, 170
ADDI x10, x0, 187
target:
ADDI x1, x10, -169
CSRRW x0, 0xC10, x1
ECALL
