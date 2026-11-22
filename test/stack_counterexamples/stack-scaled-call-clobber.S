.option norvc
.section .text
.globl _start
_start: li sp,0x80300000
  li a0,0
  call leaf
spin: j spin
# (c) andi bounds a0 to 0, then a call returns a new a0 (-5); the scaled jump lands on leaf's first row.
leaf: addi sp,sp,-16
  sd ra,8(sp)
  andi a0,a0,0
  jal get
  slli a1,a0,2
  auipc t0,0
  add a1,a1,t0
  jr a1
get: li a0,-5
  ret
