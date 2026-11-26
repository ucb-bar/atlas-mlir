/* recursion through an alias whose address is taken */
void (*volatile p)(int);
void f(int n) { volatile char b[256]; b[0] = n; p(n + 1); b[1] = 0; }
extern void h(int) __attribute__((alias("f")));
void other(int n) { (void)n; }
void g(void) { p = other; p = h; p(0); }
