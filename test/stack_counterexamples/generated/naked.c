/* a naked function: su reports 0 B, its body is asm */
/* Documents a stated limit: the rule assumes generated inline asm neither writes sp nor transfers control, so this passes with a finite bound by design. */
__attribute__((naked)) void run(void) { __asm__ volatile("1: addi sp,sp,-16\n j 1b"); }
