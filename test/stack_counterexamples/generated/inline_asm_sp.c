/* inline asm inside a parsed function moves sp in a loop; su reports the C frame only */
/* Documents a stated limit: the rule assumes generated inline asm neither writes sp nor transfers control, so this passes with a finite bound by design. */
void run(void) { __asm__ volatile("1: addi sp,sp,-16\n j 1b"); }
