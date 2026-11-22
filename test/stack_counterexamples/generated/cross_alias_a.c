/* f is defined here with an alias h; another object calls h, so f -> g -> h (= f) recurses */
extern int g(unsigned int);
int f(unsigned int n) { volatile char x[256]; x[0] = n; return g(n + 1) + x[0]; }
extern int h(unsigned int) __attribute__((alias("f")));
