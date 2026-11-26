extern int h(unsigned int);
__attribute__((noinline)) int f(unsigned int n) {
 volatile char frame[256]; frame[0] = n;
 return h(n + 1) + frame[0];
}
extern int h(unsigned int) __attribute__((alias("f")));
