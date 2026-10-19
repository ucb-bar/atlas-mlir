#include "atlas_graph_memref.h"

#if defined(ATLAS_FREESTANDING)
extern void *malloc(size_t bytes);
extern void free(void *address);
#else
#include <stdlib.h>
#endif

static int checked_cells(size_t rows, size_t columns, size_t bytes_per_cell,
                         size_t declared_bytes) {
  if (!rows || !columns || rows > SIZE_MAX / columns) return 0;
  size_t count = rows * columns;
  return count <= SIZE_MAX / bytes_per_cell &&
         count * bytes_per_cell == declared_bytes;
}

static int valid_view(const atlas_f32_memref2 *view, size_t rows,
                      size_t columns) {
  if (!view || !view->aligned || view->offset < 0 ||
      rows > INT64_MAX || columns > INT64_MAX ||
      view->sizes[0] != (int64_t)rows ||
      view->sizes[1] != (int64_t)columns ||
      view->strides[0] <= 0 || view->strides[1] <= 0)
    return 0;
  int64_t remaining = INT64_MAX - view->offset;
  if (rows > 1) {
    if (view->strides[0] > remaining / (int64_t)(rows - 1)) return 0;
    remaining -= view->strides[0] * (int64_t)(rows - 1);
  }
  return columns <= 1 || view->strides[1] <= remaining / (int64_t)(columns - 1);
}

static int disjoint_buffers(const atlas_host_buffer *buffers, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    uintptr_t address = buffers[index].address;
    size_t bytes = buffers[index].size_bytes;
    if (!bytes || address > UINT32_MAX || bytes - 1 > UINT32_MAX - address)
      return 0;
    for (size_t earlier = 0; earlier < index; ++earlier) {
      uintptr_t first = buffers[earlier].address;
      size_t length = buffers[earlier].size_bytes;
      if (address < first + length && first < address + bytes) return 0;
    }
  }
  return 1;
}

atlas_model_tile_status atlas_graph_run_memrefs(
    atlas_graph_memref_binding *binding,
    const atlas_f32_memref2 *const *arguments, size_t argument_count,
    atlas_f32_memref2 *results, size_t result_count,
    atlas_host_result *device_result) {
  if (device_result) *device_result = ATLAS_HOST_INVALID;
  if (!binding || !binding->host || !binding->io.write || !binding->io.read ||
      !binding->program_words || !binding->program_word_count ||
      !binding->inputs || !binding->input_count || !binding->outputs ||
      !binding->output_count || !binding->max_polls ||
      !binding->active_buffers || !arguments || !results ||
      argument_count != binding->input_count ||
      result_count != binding->output_count ||
      (binding->constant_count && !binding->constants) ||
      binding->input_count > SIZE_MAX - binding->constant_count ||
      binding->input_count + binding->constant_count > SIZE_MAX - binding->output_count)
    return ATLAS_MODEL_TILE_INVALID;
  size_t buffer_count = binding->input_count + binding->constant_count + binding->output_count;
  if (binding->active_buffer_capacity < buffer_count ||
      binding->input_count > SIZE_MAX / sizeof(uint8_t *) ||
      binding->output_count > SIZE_MAX / sizeof(float *))
    return ATLAS_MODEL_TILE_INVALID;

  size_t input_bytes = 0, largest_output = 0;
  for (size_t index = 0; index < binding->input_count; ++index) {
    const atlas_graph_input_slot *slot = &binding->inputs[index];
    if (!checked_cells(slot->rows, slot->columns, 1, slot->payload_bytes) ||
        !merlin_f32_u8_valid(slot->codebook) ||
        !valid_view(arguments[index], slot->rows, slot->columns) ||
        input_bytes > SIZE_MAX - slot->payload_bytes)
      return ATLAS_MODEL_TILE_INVALID;
    input_bytes += slot->payload_bytes;
    binding->active_buffers[index] = (atlas_host_buffer){
        slot->address, slot->payload_bytes, ATLAS_HOST_DEVICE_READ};
  }
  for (size_t index = 0; index < binding->constant_count; ++index) {
    const atlas_graph_constant_slot *slot = &binding->constants[index];
    if (!slot->bytes || !slot->payload_bytes) return ATLAS_MODEL_TILE_INVALID;
    binding->active_buffers[binding->input_count + index] = (atlas_host_buffer){
        slot->address, slot->payload_bytes, ATLAS_HOST_DEVICE_READ};
  }
  for (size_t index = 0; index < binding->output_count; ++index) {
    const atlas_graph_output_slot *slot = &binding->outputs[index];
    if (!checked_cells(slot->rows, slot->columns, 2, slot->payload_bytes) ||
        slot->rows * slot->columns > SIZE_MAX / sizeof(float) ||
        !slot->first_columns || slot->first_columns >= slot->columns)
      return ATLAS_MODEL_TILE_INVALID;
    if (slot->payload_bytes > largest_output) largest_output = slot->payload_bytes;
    binding->active_buffers[binding->input_count + binding->constant_count + index] =
        (atlas_host_buffer){slot->address, slot->payload_bytes, ATLAS_HOST_DEVICE_WRITE};
  }
  if (!disjoint_buffers(binding->active_buffers, buffer_count))
    return ATLAS_MODEL_TILE_INVALID;

  uint8_t *packed = malloc(input_bytes);
  float **decoded = malloc(binding->output_count * sizeof(float *));
  uint8_t *physical = NULL;
  atlas_model_tile_status status = ATLAS_MODEL_TILE_ALLOC_ERROR;
  if (decoded)
    for (size_t index = 0; index < binding->output_count; ++index) decoded[index] = NULL;
  if (!packed || !decoded) goto done;
  size_t offset = 0;
  for (size_t index = 0; index < binding->input_count; ++index) {
    const atlas_graph_input_slot *slot = &binding->inputs[index];
    const atlas_f32_memref2 *view = arguments[index];
    for (size_t row = 0; row < slot->rows; ++row) {
      for (size_t column = 0; column < slot->columns; ++column) {
        int64_t position = view->offset + (int64_t)row * view->strides[0] +
                           (int64_t)column * view->strides[1];
        uint32_t bits;
        __builtin_memcpy(&bits, &view->aligned[position], sizeof(bits));
        int converted = merlin_f32_u8_encode_bits(
            bits, slot->codebook, &packed[offset + row * slot->columns + column]);
        if (converted != MERLIN_F32_U8_OK) {
          /* The table and destination were checked above; a conversion
           * failure here is an input value outside the selected domain. */
          status = ATLAS_MODEL_TILE_INPUT_DOMAIN;
          goto done;
        }
      }
    }
    offset += slot->payload_bytes;
  }

  atlas_host_result host_result = atlas_host_load(
      binding->host, binding->program_words, binding->program_word_count);
  if (device_result) *device_result = host_result;
  if (host_result != ATLAS_HOST_OK) {
    status = ATLAS_MODEL_TILE_DEVICE_ERROR;
    goto done;
  }
  offset = 0;
  for (size_t index = 0; index < binding->input_count; ++index) {
    const atlas_graph_input_slot *slot = &binding->inputs[index];
    if (binding->io.write(binding->io.context, slot->address,
                          packed + offset, slot->payload_bytes)) {
      status = ATLAS_MODEL_TILE_MEMORY_ERROR;
      goto done;
    }
    offset += slot->payload_bytes;
  }
  for (size_t index = 0; index < binding->constant_count; ++index) {
    const atlas_graph_constant_slot *slot = &binding->constants[index];
    if (binding->io.write(binding->io.context, slot->address,
                          slot->bytes, slot->payload_bytes)) {
      status = ATLAS_MODEL_TILE_MEMORY_ERROR;
      goto done;
    }
  }
  host_result = atlas_host_start(binding->host, binding->active_buffers, buffer_count);
  if (device_result) *device_result = host_result;
  if (host_result != ATLAS_HOST_OK) {
    status = ATLAS_MODEL_TILE_DEVICE_ERROR;
    goto done;
  }
  host_result = atlas_host_wait(binding->host, binding->max_polls);
  if (device_result) *device_result = host_result;
  if (host_result != ATLAS_HOST_OK) {
    status = ATLAS_MODEL_TILE_DEVICE_ERROR;
    goto done;
  }

  physical = malloc(largest_output);
  if (!physical) goto done;
  for (size_t index = 0; index < binding->output_count; ++index) {
    const atlas_graph_output_slot *slot = &binding->outputs[index];
    size_t cells = slot->rows * slot->columns;
    decoded[index] = malloc(cells * sizeof(float));
    if (!decoded[index]) goto done;
    if (binding->io.read(binding->io.context, slot->address,
                         physical, slot->payload_bytes)) {
      status = ATLAS_MODEL_TILE_MEMORY_ERROR;
      goto done;
    }
    if (merlin_unpack_split_bf16_f32(physical, decoded[index],
                                     slot->rows, slot->columns,
                                     slot->first_columns) != MERLIN_F32_U8_OK) {
      status = ATLAS_MODEL_TILE_INVALID;
      goto done;
    }
  }
  for (size_t index = 0; index < binding->output_count; ++index) {
    const atlas_graph_output_slot *slot = &binding->outputs[index];
    results[index] = (atlas_f32_memref2){
        decoded[index], decoded[index], 0,
        {(int64_t)slot->rows, (int64_t)slot->columns},
        {(int64_t)slot->columns, 1}};
    decoded[index] = NULL;
  }
  status = ATLAS_MODEL_TILE_OK;

done:
  if (decoded) {
    for (size_t index = 0; index < binding->output_count; ++index)
      free(decoded[index]);
  }
  free(decoded);
  free(physical);
  free(packed);
  return status;
}
