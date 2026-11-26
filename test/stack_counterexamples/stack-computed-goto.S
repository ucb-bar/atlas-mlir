.option norvc
.section .text
.globl _start
_start: li sp,0x80300000
  li a0,0
  call leaf
spin: j spin
# (a) GCC computed goto shape (goto *tbl[i]): a .rodata table of local label addresses, read with ld; jr.
leaf: nop
1: addi sp,sp,-16
  la a5,tbl
  ld a5,0(a5)
  jr a5
.section .rodata
.balign 8
tbl: .dword 1b
