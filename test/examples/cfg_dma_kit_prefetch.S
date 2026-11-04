# Minimized from ResNet50 capture-r2a/kernel-00/program.S.
# Original SHA-256: 235b05962804344f8d5bcdd3197283ed7fedda14b48c52cb1734fcfaf168f335
# Keeps the first column prefetch and transpose loop, with its exact five-iteration trip count.
# Lifted constants are the original x5, x8, and x15 definitions; all other work is removed.
# One scalar NOP guards the final DELAY before the added ECALL.
li x5, 1024
li x8, 2415919104
li x15, 2007040
li x19, 0
add x20, x15, x0
li x21, 5
add x20, x20, x8
dma.load x19, x20, x5, ch0
sub x20, x20, x8
dma.wait ch0
block_0_b:
addi x24, x19, 256
addi x20, x20, 1024
addi x21, x21, -1
beq x21, x0, block_0_b_last
nop
add x20, x20, x8
dma.load x24, x20, x5, ch1
sub x20, x20, x8
vload m1, 0(x19)
delay 34
vtrpose.xlu m1, m1
delay 66
vstore m1, 0(x19)
delay 34
dma.wait ch1
add x19, x24, x0
jal x0, block_0_b
nop
block_0_b_last:
vload m1, 0(x19)
delay 34
vtrpose.xlu m1, m1
delay 66
vstore m1, 0(x19)
delay 34
nop
ecall
