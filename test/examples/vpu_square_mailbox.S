# Independent selected-assembler transcription of vpu_square_mailbox.mlir.
# Input/output pointers are loaded from a runtime mailbox, not assembled here.
ADDI x28, x0, 1
ADDI x5, x0, 0
dma.config x5, ch0
ADDI x2, x0, 32
ADDI x6, x0, 0
LUI x1, 0x90000
dma.load x6, x1, x2, ch0
dma.wait ch0
lw x1, 0(x0)
DELAY 8
lw x3, 4(x0)
DELAY 8
ADDI x2, x0, 1024
dma.load x6, x1, x2, ch0
dma.wait ch0
vload m0, 0(x6)
DELAY 33
ADDI x7, x0, 256
ADDI x1, x1, 1024
dma.load x7, x1, x2, ch0
dma.wait ch0
vload m1, 0(x7)
DELAY 33
vsquare.bf16 m4, m0
DELAY 64
ADDI x8, x0, 512
vstore m4, 0(x8)
DELAY 33
dma.store x3, x8, x2, ch1
dma.wait ch1
ADDI x8, x0, 768
vstore m5, 0(x8)
DELAY 33
ADDI x3, x3, 1024
dma.store x3, x8, x2, ch1
dma.wait ch1
ADDI x1, x0, 1
CSRRW x0, 0xC10, x1
ECALL
