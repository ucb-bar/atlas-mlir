#include "atlas_graph_memref.h"

#if defined(ATLAS_FREESTANDING)
extern void *malloc(size_t bytes);
extern void free(void *address);
#else
#include <stdlib.h>
#endif

typedef struct {
  size_t inputs;
  size_t physical;
  size_t decoded;
  size_t total;
  size_t buffers;
} graph_storage;

/* Use the selected C compiler's aggregate layout, including native padding. */
typedef struct {
  atlas_graph_pointer_context context;
  atlas_graph_memref_binding binding;
  atlas_host host;
  atlas_host_buffer active[];
} graph_context_storage;

static int checked_cells(size_t rows, size_t columns, size_t width, size_t bytes) {
  return rows && columns && rows <= SIZE_MAX / columns &&
         rows * columns <= SIZE_MAX / width && rows * columns * width == bytes;
}

static int add_bytes(size_t *total, size_t bytes) {
  if (*total > SIZE_MAX - bytes) return 0;
  *total += bytes;
  return 1;
}

static int span(const void *pointer, size_t bytes) {
  return pointer && bytes && (uintptr_t)pointer <= UINTPTR_MAX - bytes;
}

static int overlaps(const void *a, size_t an, const void *b, size_t bn) {
  return (uintptr_t)a < (uintptr_t)b + bn && (uintptr_t)b < (uintptr_t)a + an;
}

static int valid_view(const atlas_f32_memref2 *view, size_t rows, size_t columns) {
  if (!view || !view->aligned || view->offset < 0 || rows > INT64_MAX ||
      columns > INT64_MAX || view->sizes[0] != (int64_t)rows ||
      view->sizes[1] != (int64_t)columns || view->strides[0] <= 0 || view->strides[1] <= 0)
    return 0;
  int64_t remaining = INT64_MAX - view->offset;
  if (rows > 1) {
    if (view->strides[0] > remaining / (int64_t)(rows - 1)) return 0;
    remaining -= view->strides[0] * (int64_t)(rows - 1);
  }
  return columns <= 1 || view->strides[1] <= remaining / (int64_t)(columns - 1);
}

static int storage(const atlas_graph_memref_binding *b, graph_storage *s) {
  if (!b || !b->host || !b->io.write || !b->io.read || !b->program_words ||
      !b->program_word_count || !b->inputs || !b->input_count || !b->outputs ||
      !b->output_count || !b->max_polls || !b->active_buffers ||
      (b->constant_count && !b->constants) ||
      b->input_count > SIZE_MAX - b->constant_count ||
      b->input_count + b->constant_count > SIZE_MAX - b->output_count)
    return 0;
  *s = (graph_storage){0};
  s->buffers = b->input_count + b->constant_count + b->output_count;
  if (b->active_buffer_capacity < s->buffers || s->buffers > SIZE_MAX / sizeof(atlas_host_buffer)) return 0;
  for (size_t i = 0; i < b->input_count; ++i) {
    const atlas_graph_input_slot *slot = &b->inputs[i];
    if (!checked_cells(slot->rows, slot->columns, 1, slot->payload_bytes) ||
        slot->payload_bytes > SIZE_MAX / sizeof(float) ||
        !merlin_f32_u8_valid(slot->codebook) || !add_bytes(&s->inputs, slot->payload_bytes)) return 0;
  }
  for (size_t i = 0; i < b->constant_count; ++i)
    if (!b->constants[i].bytes || !b->constants[i].payload_bytes) return 0;
  for (size_t i = 0; i < b->output_count; ++i) {
    const atlas_graph_output_slot *slot = &b->outputs[i];
    if (!checked_cells(slot->rows, slot->columns, 2, slot->payload_bytes) ||
        !slot->first_columns || slot->first_columns >= slot->columns ||
        slot->payload_bytes > SIZE_MAX / 2 || !add_bytes(&s->decoded, slot->payload_bytes * 2)) return 0;
    if (slot->payload_bytes > s->physical) s->physical = slot->payload_bytes;
  }
  s->total = s->inputs;
  if (!add_bytes(&s->total, s->physical) ||
      !add_bytes(&s->total, _Alignof(float) - 1)) return 0;
  s->total -= s->total % _Alignof(float);
  return add_bytes(&s->total, s->decoded);
}

atlas_model_tile_status atlas_graph_pointer_requirements(
    const atlas_graph_memref_binding *binding, size_t *workspace_bytes,
    size_t *workspace_alignment, size_t *state_bytes, size_t *state_alignment) {
  graph_storage s;
  if (!workspace_bytes || !workspace_alignment || !state_bytes || !state_alignment || !storage(binding, &s))
    return ATLAS_MODEL_TILE_INVALID;
  size_t state = sizeof(graph_context_storage);
  if (!add_bytes(&state, s.buffers * sizeof(atlas_host_buffer))) return ATLAS_MODEL_TILE_INVALID;
  *workspace_bytes = s.total;
  *workspace_alignment = _Alignof(float);
  *state_bytes = state;
  *state_alignment = _Alignof(graph_context_storage);
  return ATLAS_MODEL_TILE_OK;
}

static int disjoint_device_buffers(atlas_graph_memref_binding *b, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    uintptr_t address = b->active_buffers[i].address;
    size_t bytes = b->active_buffers[i].size_bytes;
    if (!bytes || address > UINT32_MAX || bytes - 1 > UINT32_MAX - address) return 0;
    for (size_t j = 0; j < i; ++j) {
      uintptr_t other = b->active_buffers[j].address;
      size_t length = b->active_buffers[j].size_bytes;
      if (address < other + length && other < address + bytes) return 0;
    }
  }
  return 1;
}

static int source_span(const atlas_graph_input_slot *slot, const atlas_f32_memref2 *view,
                       const void *dense, const void **start, size_t *bytes) {
  if (view) {
    if (!valid_view(view, slot->rows, slot->columns)) return 0;
    uint64_t last = (uint64_t)view->offset + (slot->rows - 1) * (uint64_t)view->strides[0] +
                    (slot->columns - 1) * (uint64_t)view->strides[1];
    if (last >= SIZE_MAX / sizeof(float)) return 0;
    *start = view->aligned;
    *bytes = ((size_t)last + 1) * sizeof(float);
  } else {
    *start = dense;
    *bytes = slot->payload_bytes * sizeof(float);
  }
  return span(*start, *bytes) && (uintptr_t)*start % _Alignof(float) == 0;
}

static int separate_state(const atlas_graph_memref_binding *b, const graph_storage *s,
                          const void *pointer, size_t bytes) {
  return !overlaps(pointer, bytes, b, sizeof(*b)) &&
         !overlaps(pointer, bytes, b->host, sizeof(*b->host)) &&
         !overlaps(pointer, bytes, b->active_buffers, s->buffers * sizeof(atlas_host_buffer));
}

static atlas_model_tile_status run_into(
    atlas_graph_memref_binding *b, const atlas_f32_memref2 *const *views,
    const void *const *dense, void *const *results, void *workspace,
    size_t workspace_bytes, atlas_host_result *device_result) {
  if (device_result) *device_result = ATLAS_HOST_INVALID;
  graph_storage s;
  if (!storage(b, &s) || (!views && !dense) || !results || workspace_bytes < s.total ||
      !span(workspace, workspace_bytes) || (uintptr_t)workspace % _Alignof(float) ||
      !separate_state(b, &s, workspace, workspace_bytes)) return ATLAS_MODEL_TILE_INVALID;
  /* Do not overwrite descriptors retained by a previous timed-out launch. */
  if (b->host->active) {
    if (device_result) *device_result = ATLAS_HOST_BUSY;
    return ATLAS_MODEL_TILE_DEVICE_ERROR;
  }
  for (size_t i = 0; i < b->input_count; ++i) {
    const void *start;
    size_t bytes;
    if ((views && !views[i]) || !source_span(&b->inputs[i], views ? views[i] : NULL,
          dense ? dense[i] : NULL, &start, &bytes) || overlaps(start, bytes, workspace, workspace_bytes) ||
        !separate_state(b, &s, start, bytes)) return ATLAS_MODEL_TILE_INVALID;
    for (size_t j = 0; j < b->output_count; ++j) {
      size_t output_bytes = b->outputs[j].payload_bytes * 2;
      if (!span(results[j], output_bytes) || (uintptr_t)results[j] % _Alignof(float) ||
          overlaps(start, bytes, results[j], output_bytes)) return ATLAS_MODEL_TILE_INVALID;
    }
  }
  for (size_t i = 0; i < b->output_count; ++i) {
    size_t bytes = b->outputs[i].payload_bytes * 2;
    if (overlaps(results[i], bytes, workspace, workspace_bytes) ||
        !separate_state(b, &s, results[i], bytes)) return ATLAS_MODEL_TILE_INVALID;
    for (size_t j = 0; j < i; ++j)
      if (overlaps(results[i], bytes, results[j], b->outputs[j].payload_bytes * 2)) return ATLAS_MODEL_TILE_INVALID;
  }
  uint8_t *packed = workspace;
  uint8_t *physical = packed + s.inputs;
  float *decoded = (float *)(packed + s.total - s.decoded);
  size_t offset = 0;
  for (size_t i = 0; i < b->input_count; ++i) {
    const atlas_graph_input_slot *slot = &b->inputs[i];
    for (size_t row = 0; row < slot->rows; ++row) {
      for (size_t column = 0; column < slot->columns; ++column) {
        const float *source;
        if (views) {
          const atlas_f32_memref2 *view = views[i];
          source = view->aligned + view->offset + row * (size_t)view->strides[0] + column * (size_t)view->strides[1];
        } else source = (const float *)dense[i] + row * slot->columns + column;
        uint32_t bits;
        __builtin_memcpy(&bits, source, sizeof(bits));
        if (merlin_f32_u8_encode_bits(bits, slot->codebook,
              &packed[offset + row * slot->columns + column]) != MERLIN_F32_U8_OK)
          return ATLAS_MODEL_TILE_INPUT_DOMAIN;
      }
    }
    offset += slot->payload_bytes;
  }
  /* Every source conversion completed before any device or output effect. */
  for (size_t i = 0; i < b->input_count; ++i)
    b->active_buffers[i] = (atlas_host_buffer){b->inputs[i].address, b->inputs[i].payload_bytes, ATLAS_HOST_DEVICE_READ};
  for (size_t i = 0; i < b->constant_count; ++i)
    b->active_buffers[b->input_count + i] = (atlas_host_buffer){b->constants[i].address, b->constants[i].payload_bytes, ATLAS_HOST_DEVICE_READ};
  for (size_t i = 0; i < b->output_count; ++i)
    b->active_buffers[b->input_count + b->constant_count + i] = (atlas_host_buffer){b->outputs[i].address, b->outputs[i].payload_bytes, ATLAS_HOST_DEVICE_WRITE};
  if (!disjoint_device_buffers(b, s.buffers)) return ATLAS_MODEL_TILE_INVALID;
  atlas_host_result host_result = atlas_host_load(b->host, b->program_words, b->program_word_count);
  if (device_result) *device_result = host_result;
  if (host_result != ATLAS_HOST_OK) return ATLAS_MODEL_TILE_DEVICE_ERROR;
  offset = 0;
  for (size_t i = 0; i < b->input_count; ++i) {
    if (b->io.write(b->io.context, b->inputs[i].address, packed + offset, b->inputs[i].payload_bytes))
      return ATLAS_MODEL_TILE_MEMORY_ERROR;
    offset += b->inputs[i].payload_bytes;
  }
  for (size_t i = 0; i < b->constant_count; ++i)
    if (b->io.write(b->io.context, b->constants[i].address, b->constants[i].bytes, b->constants[i].payload_bytes))
      return ATLAS_MODEL_TILE_MEMORY_ERROR;
  host_result = atlas_host_start(b->host, b->active_buffers, s.buffers);
  if (host_result == ATLAS_HOST_OK) host_result = atlas_host_wait(b->host, b->max_polls);
  if (device_result) *device_result = host_result;
  if (host_result != ATLAS_HOST_OK) return ATLAS_MODEL_TILE_DEVICE_ERROR;
  offset = 0;
  for (size_t i = 0; i < b->output_count; ++i) {
    const atlas_graph_output_slot *slot = &b->outputs[i];
    if (b->io.read(b->io.context, slot->address, physical, slot->payload_bytes)) return ATLAS_MODEL_TILE_MEMORY_ERROR;
    if (merlin_unpack_split_bf16_f32(physical, (float *)((uint8_t *)decoded + offset),
          slot->rows, slot->columns, slot->first_columns) != MERLIN_F32_U8_OK) return ATLAS_MODEL_TILE_INVALID;
    offset += slot->payload_bytes * 2;
  }
  offset = 0;
  for (size_t i = 0; i < b->output_count; ++i) {
    size_t bytes = b->outputs[i].payload_bytes * 2;
    __builtin_memcpy(results[i], (uint8_t *)decoded + offset, bytes);
    offset += bytes;
  }
  return ATLAS_MODEL_TILE_OK;
}

atlas_model_tile_status atlas_graph_pointer_init(
    atlas_graph_pointer_context *context, atlas_graph_memref_binding *binding,
    void *workspace, size_t workspace_bytes, size_t state_budget_bytes) {
  size_t required, alignment, state, state_alignment;
  graph_storage storage_info;
  if (!context || (uintptr_t)context % _Alignof(atlas_graph_pointer_context) || !span(context, sizeof(*context)) || !storage(binding, &storage_info) ||
      !separate_state(binding, &storage_info, context, sizeof(*context)) ||
      atlas_graph_pointer_requirements(binding, &required, &alignment, &state, &state_alignment) != ATLAS_MODEL_TILE_OK ||
      state_budget_bytes < state || workspace_bytes < required || !span(workspace, workspace_bytes) ||
      (uintptr_t)workspace % alignment || !separate_state(binding, &storage_info, workspace, workspace_bytes) ||
      binding->host->active ||
      overlaps(context, sizeof(*context), workspace, workspace_bytes)) return ATLAS_MODEL_TILE_INVALID;
  *context = (atlas_graph_pointer_context){binding, workspace, workspace_bytes, state_budget_bytes, ATLAS_HOST_NOT_READY};
  return ATLAS_MODEL_TILE_OK;
}

int atlas_graph_run_pointers(void *opaque, const void *const *arguments, void *const *results) {
  atlas_graph_pointer_context *context = opaque;
  size_t required, alignment, state, state_alignment;
  if (!context || (uintptr_t)context % _Alignof(atlas_graph_pointer_context) || atlas_graph_pointer_requirements(context->binding, &required, &alignment, &state, &state_alignment) != ATLAS_MODEL_TILE_OK ||
      context->state_budget_bytes < state || context->workspace_bytes < required ||
      !span(context->workspace, context->workspace_bytes) || (uintptr_t)context->workspace % alignment ||
      overlaps(context, sizeof(*context), context->workspace, context->workspace_bytes)) return ATLAS_MODEL_TILE_INVALID;
  if (arguments && results) {
    for (size_t i = 0; i < context->binding->input_count; ++i)
      if (span(arguments[i], context->binding->inputs[i].payload_bytes * sizeof(float)) &&
          overlaps(context, sizeof(*context), arguments[i], context->binding->inputs[i].payload_bytes * sizeof(float)))
        return ATLAS_MODEL_TILE_INVALID;
    for (size_t i = 0; i < context->binding->output_count; ++i)
      if (span(results[i], context->binding->outputs[i].payload_bytes * 2) &&
          overlaps(context, sizeof(*context), results[i], context->binding->outputs[i].payload_bytes * 2))
        return ATLAS_MODEL_TILE_INVALID;
  }
  return run_into(context->binding, NULL, arguments, results, context->workspace,
                  context->workspace_bytes, &context->device_result);
}

atlas_model_tile_status atlas_graph_run_memrefs(
    atlas_graph_memref_binding *binding, const atlas_f32_memref2 *const *arguments,
    size_t argument_count, atlas_f32_memref2 *results, size_t result_count,
    atlas_host_result *device_result) {
  if (device_result) *device_result = ATLAS_HOST_INVALID;
  graph_storage s;
  if (!storage(binding, &s) || !arguments || !results || argument_count != binding->input_count ||
      result_count != binding->output_count || binding->output_count > SIZE_MAX / sizeof(void *))
    return ATLAS_MODEL_TILE_INVALID;
  void *workspace = malloc(s.total);
  void **outputs = malloc(binding->output_count * sizeof(void *));
  if (outputs) for (size_t i = 0; i < binding->output_count; ++i) outputs[i] = NULL;
  atlas_model_tile_status status = ATLAS_MODEL_TILE_ALLOC_ERROR;
  if (!workspace || !outputs) goto done;
  for (size_t i = 0; i < binding->output_count; ++i) {
    outputs[i] = malloc(binding->outputs[i].payload_bytes * 2);
    if (!outputs[i]) goto done;
  }
  status = run_into(binding, arguments, NULL, outputs, workspace, s.total, device_result);
  if (status == ATLAS_MODEL_TILE_OK) {
    for (size_t i = 0; i < binding->output_count; ++i) {
      const atlas_graph_output_slot *slot = &binding->outputs[i];
      results[i] = (atlas_f32_memref2){outputs[i], outputs[i], 0,
          {(int64_t)slot->rows, (int64_t)slot->columns}, {(int64_t)slot->columns, 1}};
      outputs[i] = NULL;
    }
  }
done:
  if (outputs) for (size_t i = 0; i < binding->output_count; ++i) free(outputs[i]);
  free(outputs);
  free(workspace);
  return status;
}
