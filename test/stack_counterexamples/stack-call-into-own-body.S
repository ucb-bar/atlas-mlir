.option norvc
.section .text
.globl _start
_start: li sp,0x80300000
  li a0,0
  call leaf
spin: j spin
# Indirect self-recursion through a mid-function entry: taken addresses are matched to function starts only.
leaf: nop
1: addi sp,sp,-16
  la a5,tbl
  ld a5,0(a5)
  jalr a5
  addi sp,sp,16
  ret
.section .rodata
.balign 8
tbl: .dword 1b
