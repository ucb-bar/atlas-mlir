/* same, with a local untyped label */
__asm__(".section .text.grow,\"ax\",@progbits\ngrow: addi sp,sp,-16\n j grow\n.text");
extern void grow(void);
void other(void) {}
void (*volatile hook)(void) = other;
void run(void) { hook = grow; hook(); }
