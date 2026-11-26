/* a call through an alias in another object, without a cycle: its callee's frame counts */
__attribute__((noinline)) int f(unsigned int n) { volatile char x[256]; x[0] = n; return x[0]; }
extern int h(unsigned int) __attribute__((alias("f")));
