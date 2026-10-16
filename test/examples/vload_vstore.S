# Independent selected-assembler transcription of vload_vstore.mlir.
ADDI x28, x0, 1
ADDI x5, x0, 0
dma.config x5, ch0
ADDI x2, x0, 1024
ADDI x6, x0, 0
LUI x1, 0x90000
dma.load x6, x1, x2, ch0
dma.wait ch0
vload m4, 0(x6)
DELAY 33
DELAY 1
DELAY 64
ADDI x8, x0, 256
vstore m4, 0(x8)
DELAY 33
LUI x3, 0x90000
ADDI x3, x3, 1024
dma.store x3, x8, x2, ch1
dma.wait ch1
ADDI x8, x0, 512
vstore m4, 0(x8)
DELAY 33
LUI x3, 0x90001
ADDI x3, x3, -2048
dma.store x3, x8, x2, ch1
dma.wait ch1
ADDI x1, x0, 1
CSRRW x0, 0xC10, x1
ECALL
