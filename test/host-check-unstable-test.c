#include <assert.h>
#include <stddef.h>
#include <stdint.h>

static void restore_before_rescan(const volatile void *, const void *, size_t);
#define ATLAS_HOST_CHECK_TEST_BEFORE_RESCAN restore_before_rescan
#include "atlas_host_check.h"

static size_t rescans;

static void restore_before_rescan(const volatile void *actual_span,
                                  const void *expected_span, size_t bytes) {
  /* The fixture owns writable storage. Simulate a late DMA store without a
   * timing-dependent thread or changing the checker's volatile load semantics. */
  volatile uint8_t *actual = (volatile uint8_t *)actual_span;
  const uint8_t *expected = (const uint8_t *)expected_span;
  ++rescans;
  for (size_t i = 0; i < bytes; ++i) actual[i] = expected[i];
}

int main(void) {
  uint8_t a[32] __attribute__((aligned(64))) = {0};
  uint8_t e[32] __attribute__((aligned(64))) = {0};
  /* Cover aligned words, an unaligned prefix/tail and mismatched residues. */
  const size_t offsets[][2] = {{0, 0}, {1, 1}, {0, 1}};
  for (size_t c = 0; c < sizeof(offsets) / sizeof(offsets[0]); ++c) {
    uint8_t *actual = a + offsets[c][0];
    uint8_t *expected = e + offsets[c][1];
    const size_t bytes = 17;
    for (size_t i = 0; i < bytes; ++i) {
      actual[i] = 1;
      const size_t before = rescans;
      atlas_host_check_result r = atlas_host_check_exact(actual, expected, bytes);
      assert(rescans == before + 1); /* The first pass observed the mismatch. */
      assert(r.mismatch_bytes == 1 && r.first_offset == SIZE_MAX);
      for (size_t j = 0; j < bytes; ++j) assert(actual[j] == expected[j]);
      r = atlas_host_check_exact(actual, expected, bytes);
      assert(!r.mismatch_bytes && r.first_offset == SIZE_MAX);
      assert(rescans == before + 1); /* Equal inputs do not enter the rescan. */
    }
  }
  return 0;
}
