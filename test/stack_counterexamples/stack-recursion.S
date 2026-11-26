.section .text
.globl _start
_start: li sp,0
  call leaf
spin: j spin
leaf: addi sp,sp,-16
  la a5,leaf
  jalr a5
  addi sp,sp,16
  ret
