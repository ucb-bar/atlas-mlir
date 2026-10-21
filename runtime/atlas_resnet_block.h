#ifndef ATLAS_RESNET_BLOCK_H
#define ATLAS_RESNET_BLOCK_H
#include "atlas_host.h"

/* This ABI is deliberately one frozen first-bottleneck scope, with an external
 * 64x56x56 FP32 NCHW entry and four already-qualified device programs. */
enum { ATLAS_RESNET_SPATIAL = 56 * 56, ATLAS_RESNET_SMALL = 64 * 56 * 56, ATLAS_RESNET_LARGE = 256 * 56 * 56 };
typedef struct {
  const uint32_t *program;
  size_t program_words;
  const uint8_t *weights;
  const float *affine_scale_bias;
  size_t activation_bytes, weight_bytes, output_bytes, output_offset, dram_bytes;
  unsigned kernel, output_channels;
  int activation_scale;
} atlas_resnet_block_kernel;
typedef struct {
  const float *entry;
  const uint32_t *expected_final_bits;
  atlas_resnet_block_kernel kernels[4];
} atlas_resnet_block_inputs;
typedef struct {
  uint8_t *dram;
  size_t dram_capacity;
  uintptr_t device_dram_address;
  float *small, *branch, *identity;
  size_t small_words, branch_words, identity_words;
  size_t max_polls;
  /* Retained across a failed/timeout launch: never reuse this context or any
   * workspace until success or externally established engine quiescence. */
  atlas_host_buffer active_buffer;
  unsigned stage, completed_stages;
  atlas_host_result runtime_result;
  int observed_scale[4];
  uint32_t completion_sequence[4], atlas_cycles[4], first_bad_word;
} atlas_resnet_block_workspace;
typedef enum {
  ATLAS_RESNET_BLOCK_OK = 0,
  ATLAS_RESNET_BLOCK_INVALID,
  ATLAS_RESNET_BLOCK_PACK,
  ATLAS_RESNET_BLOCK_SCALE,
  ATLAS_RESNET_BLOCK_DEVICE,
  ATLAS_RESNET_BLOCK_GUARD,
  ATLAS_RESNET_BLOCK_MATH,
  ATLAS_RESNET_BLOCK_OUTPUT
} atlas_resnet_block_result;
atlas_resnet_block_result atlas_resnet_block_run(atlas_host *, const atlas_resnet_block_inputs *,
                                                 atlas_resnet_block_workspace *);
#endif
