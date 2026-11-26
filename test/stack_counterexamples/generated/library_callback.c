extern void qsort(void *, unsigned long, unsigned long, int (*)(const void *, const void *));
int run(int);
__attribute__((noinline)) int compare(const void *a, const void *b) {
 volatile char frame[256]; frame[0] = *(const int *)a;
 return run(frame[0]) + *(const int *)b;
}
__attribute__((noinline)) int run(int n) {
 volatile char frame[512]; int a[2] = {n, n + 1}; frame[0] = (char)n;
 qsort(a, 2, sizeof(int), compare);
 return frame[0] + a[0];
}
