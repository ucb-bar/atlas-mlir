# Independent selected-assembler transcription of mxu1_k2.mlir.
ADDI x28, x0, 1
ADDI x5, x0, 0
dma.config x5, ch0
ADDI x2, x0, 1024
ADDI x6, x0, 0
LUI x1, 0x90000
dma.load x6, x1, x2, ch0
dma.wait ch0
vload m0, 0(x6)
DELAY 33
ADDI x7, x0, 256
LUI x1, 0x90001
ADDI x1, x1, -2048
dma.load x7, x1, x2, ch0
dma.wait ch0
vload m4, 0(x7)
DELAY 33
vmatpush.weight.mxu1 w0, m4
DELAY 31
vmatmul.mxu1 acc0, m0, w0
DELAY 95
ADDI x6, x0, 512
LUI x1, 0x90000
ADDI x1, x1, 1024
dma.load x6, x1, x2, ch0
dma.wait ch0
vload m2, 0(x6)
DELAY 33
ADDI x7, x0, 768
LUI x1, 0x90001
ADDI x1, x1, -1024
dma.load x7, x1, x2, ch0
dma.wait ch0
vload m6, 0(x7)
DELAY 33
vmatpush.weight.mxu1 w0, m6
DELAY 31
vmatmul.acc.mxu1 acc0, m2, w0
DELAY 95
vmatpop.bf16.acc.mxu1 m8, acc0
DELAY 31
ADDI x8, x0, 1024
vstore m8, 0(x8)
DELAY 33
LUI x3, 0x90001
dma.store x3, x8, x2, ch1
dma.wait ch1
ADDI x8, x0, 1280
vstore m9, 0(x8)
DELAY 33
LUI x3, 0x90001
ADDI x3, x3, 1024
dma.store x3, x8, x2, ch1
dma.wait ch1
ADDI x1, x0, 1
CSRRW x0, 0xC10, x1
ECALL
