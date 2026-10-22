#ifndef ATLAS_SEGMENT_H
#define ATLAS_SEGMENT_H
#include <stddef.h>
#include <stdint.h>

/* One end-state region: live CPU or DMA memory compared with captured expected bytes in 64-bit words. */
typedef struct {
  const char *name;
  const void *actual, *expected;
  size_t bytes;
} atlas_segment_region;

/* Generated per segment: launch index, statement count, expected runtime state and end-state regions. */
typedef struct {
  unsigned index, statements, regions;
  int activation_scale, rescale_exponent;
  const float *input;
  atlas_segment_region region[8];
} atlas_segment_info;

extern const atlas_segment_info atlas_segment;
int atlas_resnet_segment_run(const float *, float *, void *,
                             int (*)(void *, unsigned, const uint8_t *, size_t, int, uint8_t *, size_t));
#endif
