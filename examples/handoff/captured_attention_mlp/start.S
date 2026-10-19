    .section .text.init,"ax",@progbits
    .global _start
    .type _start,@function
_start:
    la sp, _stack_top
    la t0, _bss_start
    la t1, _bss_end
1:
    bgeu t0, t1, 2f
    sd zero, 0(t0)
    addi t0, t0, 8
    j 1b
2:
#if defined(__riscv_flen) && defined(__riscv_vector)
    li t0, 0x6600  /* mstatus.FS and VS = Dirty */
#elif defined(__riscv_flen)
    li t0, 0x6000  /* mstatus.FS = Dirty */
#elif defined(__riscv_vector)
    li t0, 0x0600  /* mstatus.VS = Dirty */
#endif
#if defined(__riscv_flen) || defined(__riscv_vector)
    csrs mstatus, t0
#endif
#if defined(__riscv_flen)
    csrwi fcsr, 0
#endif
#if defined(__riscv_vector)
    csrwi vcsr, 0
#endif
    call main
3:
    wfi
    j 3b
