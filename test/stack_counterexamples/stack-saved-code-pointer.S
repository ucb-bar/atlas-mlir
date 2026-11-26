.option norvc
.section .text
.globl _start
_start: li sp,0x80300000
  li a0,0
  call leaf
spin: j spin
# (a) jr through a loaded pointer that holds a local code address, not a function: loops, growing sp.
leaf: auipc a1,0
  addi sp,sp,-16
  sd a1,0(sp)
  ld a0,0(sp)
  jr a0
