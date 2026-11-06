#ifndef ATLAS_HOST_CHECK_H
#define ATLAS_HOST_CHECK_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
  size_t mismatch_bytes;
  size_t first_offset; /* SIZE_MAX when equal or a mismatch cannot be located. */
  uint8_t actual, expected;
} atlas_host_check_result;

/* DMA must be complete and ownership returned to the host before entry. The
 * spans must remain stable for both passes. No byte outside either span is read.
 * GCC/Clang's may_alias makes word reads legal for byte-backed images. Other
 * compilers retain the exact byte path. A reduction with OR cannot cancel two
 * mismatches: it is zero exactly when every compared bit is equal. */
static inline atlas_host_check_result atlas_host_check_exact(
    const volatile void *actual_span, const void *expected_span, size_t bytes) {
  const volatile uint8_t *actual = (const volatile uint8_t *)actual_span;
  const uint8_t *expected = (const uint8_t *)expected_span;
  atlas_host_check_result result = {0, SIZE_MAX, 0, 0};
  size_t i = 0;
  uint64_t different = 0;
#if defined(__GNUC__) || defined(__clang__)
  typedef uint64_t atlas_check_word __attribute__((__may_alias__));
  if (((uintptr_t)actual & 7u) == ((uintptr_t)expected & 7u)) {
    while (i < bytes && ((uintptr_t)(actual + i) & 7u)) {
      different |= (uint64_t)(actual[i] ^ expected[i]);
      ++i;
    }
    while (bytes - i >= 8) {
      different |= *(const volatile atlas_check_word *)(actual + i) ^ *(const atlas_check_word *)(expected + i);
      i += 8;
    }
  }
#endif
  while (i < bytes) {
    different |= (uint64_t)(actual[i] ^ expected[i]);
    ++i;
  }
  if (!different) return result;
#ifdef ATLAS_HOST_CHECK_TEST_BEFORE_RESCAN
  /* Test-only seam for deterministic changes between the two volatile scans. */
  ATLAS_HOST_CHECK_TEST_BEFORE_RESCAN(actual_span, expected_span, bytes);
#endif
  /* Failure-only byte scan supplies exact diagnostics for stable spans. */
  for (i = 0; i < bytes; ++i) {
    uint8_t got = actual[i], want = expected[i];
    if (got != want) {
      if (!result.mismatch_bytes) {
        result.first_offset = i;
        result.actual = got;
        result.expected = want;
      }
      ++result.mismatch_bytes;
    }
  }
  /* A disappearing mismatch still invalidates the first-pass comparison. */
  if (!result.mismatch_bytes) {
    result.mismatch_bytes = 1;
    result.first_offset = SIZE_MAX;
  }
  return result;
}
#endif
