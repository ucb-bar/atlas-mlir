# Independent selected-assembler transcription of vpu_square_mailbox.mlir.
# Input/output pointers are loaded from a runtime mailbox, not assembled here.
ADDI x28, x0, 1
ADDI x5, x0, 0
DMA.CONFIG x5, 0
ADDI x2, x0, 32
ADDI x6, x0, 0
LUI x1, 0x90000
DMA.LOAD x6, x1, x2, 0
DMA.WAIT 0
LW x1, x0, 0
DELAY 8
LW x3, x0, 4
DELAY 8
ADDI x2, x0, 1024
DMA.LOAD x6, x1, x2, 0
DMA.WAIT 0
VLOAD 0, x6, 0
DELAY 33
ADDI x7, x0, 256
ADDI x1, x1, 1024
DMA.LOAD x7, x1, x2, 0
DMA.WAIT 0
VLOAD 1, x7, 0
DELAY 33
VSQUARE.BF16 4, 0
DELAY 64
ADDI x8, x0, 512
VSTORE 4, x8, 0
DELAY 33
DMA.STORE x3, x8, x2, 1
DMA.WAIT 1
ADDI x8, x0, 768
VSTORE 5, x8, 0
DELAY 33
ADDI x3, x3, 1024
DMA.STORE x3, x8, x2, 1
DMA.WAIT 1
ADDI x1, x0, 1
CSRRW x0, 0xC10, x1
ECALL
