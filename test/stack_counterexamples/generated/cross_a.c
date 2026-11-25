/* static functions of one name in two objects; the pointer table in b points at b's static f, which recurses */
void (*volatile tbl)(int);
static void f(int n) { volatile char b[16]; b[0] = n; }
void a_run(void) { f(1); tbl(0); }
