/* top-level asm defines untyped code (no .type, so NOTYPE) that grows the stack and loops; its address is taken */
__asm__(".section .text.grow,\"ax\",@progbits\n.globl grow\ngrow: addi sp,sp,-16\n j grow\n.text");
extern void grow(void);
void other(void) {}
void (*volatile hook)(void) = other;
void run(void) { hook = grow; hook(); }
