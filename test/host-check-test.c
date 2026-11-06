#include "atlas_host_check.h"
#include <assert.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static void check(uint8_t *a, uint8_t *e, size_t n) {
  for (size_t i = 0; i < n; ++i) a[i] = e[i] = (uint8_t)(i * 37 + 19);
  atlas_host_check_result r = atlas_host_check_exact(a, e, n);
  assert(!r.mismatch_bytes && r.first_offset == SIZE_MAX);
  for (size_t i = 0; i < n; ++i) {
    a[i] ^= 128;
    r = atlas_host_check_exact(a, e, n);
    assert(r.mismatch_bytes == 1 && r.first_offset == i);
    assert(r.actual == a[i] && r.expected == e[i]);
    a[i] ^= 128;
  }
  if (n > 8) {
    /* Equal bit errors in separate words would cancel an XOR reduction. */
    a[0] ^= 1; a[8] ^= 1;
    r = atlas_host_check_exact(a, e, n);
    assert(r.mismatch_bytes == 2 && r.first_offset == 0);
    a[0] ^= 1; a[8] ^= 1;
  }
}

int main(void) {
  uint8_t a[160] __attribute__((aligned(64))), e[160] __attribute__((aligned(64)));
  for (size_t x = 0; x < 8; ++x)
    for (size_t y = 0; y < 8; ++y)
      for (size_t n = 0; n <= 129; ++n) {
        memset(a, 0xa5, sizeof(a)); memset(e, 0x5a, sizeof(e));
        check(a + x, e + y, n);
        for (size_t i = 0; i < x; ++i) assert(a[i] == 0xa5);
        for (size_t i = x + n; i < sizeof(a); ++i) assert(a[i] == 0xa5);
        for (size_t i = 0; i < y; ++i) assert(e[i] == 0x5a);
        for (size_t i = y + n; i < sizeof(e); ++i) assert(e[i] == 0x5a);
      }
  size_t page = (size_t)sysconf(_SC_PAGESIZE);
  uint8_t *pa = mmap(0, 3 * page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  uint8_t *pe = mmap(0, 3 * page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(pa != MAP_FAILED && pe != MAP_FAILED);
  assert(!mprotect(pa + page, page, PROT_READ | PROT_WRITE));
  assert(!mprotect(pe + page, page, PROT_READ | PROT_WRITE));
  for (size_t n = 0; n <= 129; ++n) {
    check(pa + 2 * page - n, pe + 2 * page - n, n);
    check(pa + page, pe + page, n);
    /* Independently put either actual or expected at an inaccessible boundary. */
    for (size_t offset = 0; offset < 8; ++offset) {
      check(pa + 2 * page - n, pe + page + offset, n);
      check(pa + page + offset, pe + 2 * page - n, n);
    }
  }
  assert(!munmap(pa, 3 * page)); assert(!munmap(pe, 3 * page));
  return 0;
}
