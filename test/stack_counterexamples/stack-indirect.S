.section .text
.globl _start
_start: li sp,0
  call leaf
spin: j spin
leaf: auipc a0,0
  addi sp,sp,-16
  jr a0
