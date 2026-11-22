.option norvc
.section .text
.globl _start
_start: li sp,0x80300000
  li a0,0
  call leaf
spin: j spin
# (b) bltu bounds a0 to 0, but a0 is rewritten to 1 before slli; entry 1 (not read by the analysis) loops.
leaf:
1: addi sp,sp,-16
  li a0,0
  li a5,0
  bltu a5,a0,2f
  li a0,1
  slli a0,a0,2
  la a4,tbl
  add a0,a0,a4
  lw a5,0(a0)
  add a5,a5,a4
  jr a5
2: j 2b
.section .rodata
.balign 4
tbl: .word 2b-tbl
  .word 1b-tbl
