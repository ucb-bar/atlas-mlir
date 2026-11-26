extern int h(unsigned int);
int g(unsigned int n) { volatile char b[64]; b[0] = n; return h(n) + b[0]; }
