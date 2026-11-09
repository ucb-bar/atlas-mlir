#ifndef ATLAS_GRAPH_MEMREF_H
#define ATLAS_GRAPH_MEMREF_H

#include "atlas_model_tile.h"
#include "finite_float_boundary.h"

/* One fixed physical slot for a source tensor argument. The source value is
 * read through an MLIR rank-two f32 memref and encoded before any device write. */
typedef struct {
  uintptr_t address;
  size_t rows;
  size_t columns;
  size_t payload_bytes;
  const merlin_f32_u8_codebook *codebook;
} atlas_graph_input_slot;

typedef struct {
  uintptr_t address;
  size_t rows;
  size_t columns;
  size_t first_columns;
  size_t payload_bytes;
} atlas_graph_output_slot;

typedef struct {
  uintptr_t address;
  const uint8_t *bytes;
  size_t payload_bytes;
} atlas_graph_constant_slot;

typedef struct {
  atlas_host *host;
  atlas_tile_memory_io io;
  const uint32_t *program_words;
  size_t program_word_count;
  const atlas_graph_input_slot *inputs;
  size_t input_count;
  const atlas_graph_constant_slot *constants;
  size_t constant_count;
  const atlas_graph_output_slot *outputs;
  size_t output_count;
  size_t max_polls;
  /* The host retains these descriptors across a timed-out launch. */
  atlas_host_buffer *active_buffers;
  size_t active_buffer_capacity;
} atlas_graph_memref_binding;

/* Explicit dense-pointer context. All pointed-to binding/host/active-buffer
 * objects belong to the caller and survive until the device is quiescent.
 * The workspace is separate from logical inputs/outputs and persistent state.
 * No output pointer is retained, and no output is modified on failure. */
typedef struct {
  atlas_graph_memref_binding *binding;
  void *workspace;
  size_t workspace_bytes;
  size_t state_budget_bytes;
  atlas_host_result device_result;
} atlas_graph_pointer_context;

/* Native sizeof/alignment-derived requirements, not an assumed host layout.
 * state_bytes includes context, binding, host and active-buffer descriptors;
 * immutable program/slot/codebook arrays remain separately owned artifacts. */
atlas_model_tile_status atlas_graph_pointer_requirements(
    const atlas_graph_memref_binding *binding, size_t *workspace_bytes,
    size_t *workspace_alignment, size_t *state_bytes, size_t *state_alignment);
atlas_model_tile_status atlas_graph_pointer_init(
    atlas_graph_pointer_context *context, atlas_graph_memref_binding *binding,
    void *workspace, size_t workspace_bytes, size_t state_budget_bytes);

/* int(void *, const void *const *, void *const *) wrapper. Zero means the
 * selected launch and required visibility operations completed and every
 * dense f32 output is valid. A nonzero result leaves driver/context state
 * retained; drain/reset before reuse. Inputs/outputs follow declared slots. */
int atlas_graph_run_pointers(void *context, const void *const *arguments,
                             void *const *results);

/* One synchronous graph invocation. `results` remains unchanged on failure.
 * On success each result owns its allocated dense f32 storage. The caller
 * frees every result's `allocated` pointer. A timeout leaves host->active set;
 * the owner must finish or reset that launch before reusing this binding. */
atlas_model_tile_status atlas_graph_run_memrefs(
    atlas_graph_memref_binding *binding,
    const atlas_f32_memref2 *const *arguments, size_t argument_count,
    atlas_f32_memref2 *results, size_t result_count,
    atlas_host_result *device_result);

#endif
