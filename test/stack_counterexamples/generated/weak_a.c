/* a weak generated function recursing through a pointer that another generated object stores */
extern void (*volatile hook)(int);
__attribute__((weak)) void f(int n) { volatile char b[512]; b[0] = n; hook(n + 1); b[1] = 0; }
