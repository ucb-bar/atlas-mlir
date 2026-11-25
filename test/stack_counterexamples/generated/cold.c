/* static cold function recursing through a stored pointer */
void (*volatile p)(int);
__attribute__((cold, noinline)) static void f(int n) { volatile char b[256]; b[0] = n; p(n + 1); b[1] = 0; }
void other(int n) { (void)n; }
void g(void) { p = other; p = f; p(0); }
