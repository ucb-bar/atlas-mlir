# Independent selected-assembler transcription of dma_pointer_lifetime.mlir.
ADDI x28, x0, 1
ADDI x5, x0, 0
dma.config x5, ch0
ADDI x6, x0, 0
LUI x1, 0x90000
ADDI x2, x0, 128
dma.load x6, x1, x2, ch0
LUI x1, 0x90001
dma.wait ch0
ADDI x10, x0, 1
LUI x3, 0x90000
ADDI x3, x3, 1024
dma.store x3, x6, x2, ch1
dma.wait ch1
ADDI x11, x0, 1
dma.load x6, x1, x2, ch0
dma.wait ch0
LUI x4, 0x90000
ADDI x4, x4, 1536
dma.store x4, x6, x2, ch1
dma.wait ch1
ADDI x12, x0, 1
ADDI x1, x0, 1
CSRRW x0, 0xC10, x1
ECALL
