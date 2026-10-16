# Hand-maintained selected-assembler transcription of vpu_e8m0_pack_chain.mlir.
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
LUI x1, 0x90000
ADDI x1, x1, 1024
dma.load x7, x1, x2, ch0
dma.wait ch0
vload m1, 0(x7)
DELAY 33
seli e3, 128
vpack.bf16.fp8 m2, m0, e3
DELAY 95
vunpack.fp8.bf16 m4, m2, e3
DELAY 127
vmatpush.weight.mxu1 w0, m2
DELAY 31
vmatmul.mxu1 acc0, m2, w0
DELAY 95
vmatpop.bf16.acc.mxu1 m6, acc0
DELAY 31
ADDI x8, x0, 512
vstore m2, 0(x8)
DELAY 33
LUI x3, 0x90001
ADDI x3, x3, -2048
dma.store x3, x8, x2, ch1
dma.wait ch1
ADDI x8, x0, 768
vstore m4, 0(x8)
DELAY 33
LUI x3, 0x90001
ADDI x3, x3, -1024
dma.store x3, x8, x2, ch1
dma.wait ch1
ADDI x8, x0, 1024
vstore m5, 0(x8)
DELAY 33
LUI x3, 0x90001
dma.store x3, x8, x2, ch1
dma.wait ch1
ADDI x8, x0, 1280
vstore m6, 0(x8)
DELAY 33
LUI x3, 0x90001
ADDI x3, x3, 1024
dma.store x3, x8, x2, ch1
dma.wait ch1
ADDI x8, x0, 1536
vstore m7, 0(x8)
DELAY 33
LUI x3, 0x90002
ADDI x3, x3, -2048
dma.store x3, x8, x2, ch1
dma.wait ch1
ADDI x1, x0, 1
CSRRW x0, 0xC10, x1
ECALL
