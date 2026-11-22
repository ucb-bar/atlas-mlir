/* untyped code under a local label in a section whose legal ELF name contains a space; its address is taken */
__asm__(".section \"extra code\",\"ax\",@progbits\ngrow: addi sp,sp,-16\n j grow\n.text");
extern void grow(void);
void other(void) {}
void (*volatile hook)(void) = other;
void run(void) { hook = grow; hook(); }
