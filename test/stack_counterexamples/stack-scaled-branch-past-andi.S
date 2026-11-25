.option norvc
.section .text
.globl _start
_start: li sp,0x80300000
  li a0,0
  call leaf
spin: j spin
# (c) a branch lands on the slli itself, skipping the andi that bounds the index.
leaf: addi sp,sp,-16
  li a0,-5
  bnez a0,1f
  andi a0,a0,0
1: slli a1,a0,2
  auipc t0,0
  add a1,a1,t0
  jr a1
