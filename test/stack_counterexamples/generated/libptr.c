/* indirect call to a library function through a pointer; one generated address is also taken */
#include <string.h>
void *(*volatile p)(void *, int, size_t);
void other(void) {}
void (*volatile q)(void) = other;
void g(char *d) { p = memset; p(d, 0, 4); }
