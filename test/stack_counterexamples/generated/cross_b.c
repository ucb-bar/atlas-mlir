extern void (*volatile tbl)(int);
static void f(int n) { volatile char b[512]; b[0] = n; tbl(n + 1); }
void b_init(void) { tbl = f; }
