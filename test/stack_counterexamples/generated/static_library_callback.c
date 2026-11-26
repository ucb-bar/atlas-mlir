/* recursion through a library callback (qsort) */
#include <stdlib.h>
static int depth;
static int cmp(const void *a, const void *b) { volatile char buf[256]; int x[2] = {2, 1}; buf[0] = 1; depth++; qsort(x, 2, sizeof x[0], cmp); return *(const int *)a - *(const int *)b + buf[0]; }
void g(void) { int x[2] = {2, 1}; qsort(x, 2, sizeof x[0], cmp); }
