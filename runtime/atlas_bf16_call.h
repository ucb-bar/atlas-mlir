#ifndef ATLAS_BF16_CALL_H
#define ATLAS_BF16_CALL_H
#include "atlas_model_tile.h"

/* One admitted raw-BF16 split-pair program. These values are bound from the
 * native plan and the caller's pre-reserved device intervals, not inferred. */
typedef struct {
  const uint32_t *words;
  size_t word_count;
  size_t rows, columns, first_columns, payload_bytes;
  uintptr_t input_address, output_address;
  size_t device_alignment;
} atlas_bf16_program;

typedef struct {
  atlas_host host;
  atlas_tile_memory_io io;
  atlas_host_buffer active_buffers[2];
  const atlas_bf16_program *program;
  uint8_t *workspace;
  size_t workspace_bytes, max_polls;
  atlas_host_result device_result;
  int initialized, failed;
} atlas_bf16_call;

/* Platform callbacks and map are selected explicitly. No IO is performed by
 * initialization. Caller zero-initializes a fresh context and owns it, its
 * workspace and device intervals until an explicit drain/reset is complete. */
atlas_model_tile_status atlas_bf16_call_init(
    atlas_bf16_call *context, const atlas_bf16_program *program,
    atlas_host_map map, atlas_host_mmio mmio, atlas_host_memory memory,
    atlas_tile_memory_io io, size_t max_polls, void *workspace, size_t workspace_bytes);

/* One dense BF16 input and one dense BF16 output. Exact bit-domain checking and
 * all alias/extent checks precede every device IO; output is unchanged on error.
 * A post-IO error is sticky, even if the device has already halted. */
int atlas_bf16_call_run(void *context, const void *const *inputs, void *const *outputs);
atlas_host_result atlas_bf16_call_drain(atlas_bf16_call *context, size_t max_polls);
#endif
