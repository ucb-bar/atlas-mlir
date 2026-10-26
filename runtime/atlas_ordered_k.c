#include "atlas_ordered_k.h"

enum { ATLAS_ORDERED_K_EDGE = 32, ATLAS_ORDERED_K_INPUT_BYTES = 1024,
       ATLAS_ORDERED_K_OUTPUT_BYTES = 2048, ATLAS_ORDERED_K_INPUT_SLOTS = 128 };

static int valid_device_range(uintptr_t address, size_t bytes) {
  return address && address <= UINT32_MAX && bytes &&
         bytes - 1 <= UINT32_MAX - address;
}

static int overlaps(uintptr_t first, size_t first_bytes,
                    uintptr_t second, size_t second_bytes) {
  return first <= second ? second - first < first_bytes
                         : first - second < second_bytes;
}

static int valid_host_span(const void *pointer, size_t bytes) {
  return pointer && bytes && bytes <= UINTPTR_MAX - (uintptr_t)pointer;
}

int atlas_ordered_k_binding_valid(const atlas_ordered_k_tile *tile, const atlas_host *host,
                                const void *scratch, size_t scratch_bytes,
                                const atlas_host_buffer *buffers, size_t buffer_capacity) {
  if (buffer_capacity > SIZE_MAX / sizeof(*buffers)) return 0;
  const void *objects[] = {tile, host, scratch, buffers};
  const size_t bytes[] = {sizeof(*tile), sizeof(*host), scratch_bytes, buffer_capacity * sizeof(*buffers)};
  for (unsigned i = 0; i < 4; ++i) {
    if (!valid_host_span(objects[i], bytes[i])) return 0;
    for (unsigned j = 0; j < i; ++j)
      if (overlaps((uintptr_t)objects[i], bytes[i], (uintptr_t)objects[j], bytes[j])) return 0;
  }
  return 1;
}

int atlas_ordered_k_admit_code(uint8_t code) {
  unsigned exponent = (code >> 3) & 15;
  return code == 0 || (exponent >= 1 && exponent <= 14);
}

static int admitted_mxu0_fp8(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (!atlas_ordered_k_admit_code(bytes[index]))
      return 0;
  }
  return 1;
}

atlas_ordered_k_status atlas_ordered_k_run(
    atlas_ordered_k_tile *tile,
    const uint8_t *activation, size_t activation_bytes,
    const uint8_t *weight, size_t weight_bytes,
    uint8_t *physical_output, size_t physical_output_bytes,
    atlas_host_result *device_result) {
  if (device_result)
    *device_result = ATLAS_HOST_INVALID;
  if (!tile || !tile->host || tile->host->active || !tile->write || !tile->read ||
      !tile->pack || !tile->program_words || !tile->program_word_count ||
      !tile->windows || tile->window_count < 2 ||
      (tile->window_count & 1) || tile->window_count > ATLAS_ORDERED_K_INPUT_SLOTS ||
      !tile->max_polls || !tile->scratch ||
      tile->scratch_bytes < ATLAS_ORDERED_K_OUTPUT_BYTES ||
      !tile->active_buffers || tile->active_buffer_capacity <= tile->window_count ||
      !valid_host_span(activation, activation_bytes) ||
      !valid_host_span(weight, weight_bytes) ||
      !valid_host_span(physical_output, physical_output_bytes) ||
      !valid_host_span(tile->scratch, tile->scratch_bytes) ||
      physical_output_bytes != ATLAS_ORDERED_K_OUTPUT_BYTES ||
      !valid_device_range(tile->output_address, physical_output_bytes))
    return ATLAS_ORDERED_K_INVALID;

  size_t source_columns = tile->windows[0].source_columns;
  if (!source_columns || source_columns > SIZE_MAX / ATLAS_ORDERED_K_EDGE ||
      tile->window_count != 2 * (source_columns / ATLAS_ORDERED_K_EDGE
                                + (source_columns % ATLAS_ORDERED_K_EDGE != 0)))
    return ATLAS_ORDERED_K_INVALID;
  size_t logical_bytes = ATLAS_ORDERED_K_EDGE * source_columns;
  if (activation_bytes != logical_bytes || weight_bytes != logical_bytes ||
      overlaps((uintptr_t)activation, activation_bytes,
               (uintptr_t)physical_output, physical_output_bytes) ||
      overlaps((uintptr_t)weight, weight_bytes,
               (uintptr_t)physical_output, physical_output_bytes) ||
      overlaps((uintptr_t)activation, activation_bytes,
               (uintptr_t)tile->scratch, tile->scratch_bytes) ||
      overlaps((uintptr_t)weight, weight_bytes,
               (uintptr_t)tile->scratch, tile->scratch_bytes) ||
      overlaps((uintptr_t)physical_output, physical_output_bytes,
               (uintptr_t)tile->scratch, tile->scratch_bytes))
    return ATLAS_ORDERED_K_INVALID;

  for (size_t index = 0; index < tile->window_count; ++index) {
    const atlas_ordered_k_window *window = &tile->windows[index];
    size_t k_index = index / 2;
    size_t head_count = source_columns % ATLAS_ORDERED_K_EDGE;
    size_t source_count = head_count && k_index == 0
                              ? head_count : ATLAS_ORDERED_K_EDGE;
    size_t source_start = k_index == 0 ? 0
        : head_count ? head_count + (k_index - 1) * ATLAS_ORDERED_K_EDGE
                     : k_index * ATLAS_ORDERED_K_EDGE;
    if (window->source_index != index % 2 ||
        window->rows != ATLAS_ORDERED_K_EDGE ||
        window->source_columns != source_columns ||
        window->k_start != source_start ||
        window->tile_columns != source_count ||
        !valid_device_range(window->address, ATLAS_ORDERED_K_INPUT_BYTES) ||
        overlaps(window->address, ATLAS_ORDERED_K_INPUT_BYTES,
                 tile->output_address, ATLAS_ORDERED_K_OUTPUT_BYTES))
      return ATLAS_ORDERED_K_INVALID;
    for (size_t previous = 0; previous < index; ++previous)
      if (overlaps(window->address, ATLAS_ORDERED_K_INPUT_BYTES,
                   tile->windows[previous].address, ATLAS_ORDERED_K_INPUT_BYTES))
        return ATLAS_ORDERED_K_INVALID;
  }
  if (!admitted_mxu0_fp8(activation, activation_bytes) ||
      !admitted_mxu0_fp8(weight, weight_bytes))
    return ATLAS_ORDERED_K_INPUT_DOMAIN;

  const uint8_t *sources[2] = {activation, weight};
  size_t source_lengths[2] = {activation_bytes, weight_bytes};
  for (size_t index = 0; index < tile->window_count; ++index) {
    const atlas_ordered_k_window *window = &tile->windows[index];
    unsigned source_index = window->source_index;
    if (window->tile_columns == ATLAS_ORDERED_K_EDGE) {
      if (tile->pack(sources[source_index], source_lengths[source_index],
                     tile->scratch, ATLAS_ORDERED_K_INPUT_BYTES,
                     window->rows, window->source_columns, window->k_start,
                     window->tile_columns, 1))
        return ATLAS_ORDERED_K_INVALID;
    } else {
      size_t head_zeros = ATLAS_ORDERED_K_EDGE - window->tile_columns;
      for (size_t row = 0; row < ATLAS_ORDERED_K_EDGE; ++row) {
        for (size_t column = 0; column < head_zeros; ++column)
          tile->scratch[row * ATLAS_ORDERED_K_EDGE + column] = 0;
        for (size_t column = 0; column < window->tile_columns; ++column)
          tile->scratch[row * ATLAS_ORDERED_K_EDGE + head_zeros + column] =
              sources[source_index][row * source_columns + window->k_start + column];
      }
    }
    if (tile->write(tile->memory_context, window->address,
                    tile->scratch, ATLAS_ORDERED_K_INPUT_BYTES))
      return ATLAS_ORDERED_K_MEMORY_ERROR;
  }

  atlas_host_result host_result = atlas_host_ensure_loaded(tile->host, tile->program_words,
                                                  tile->program_word_count);
  if (device_result)
    *device_result = host_result;
  if (host_result != ATLAS_HOST_OK)
    return ATLAS_ORDERED_K_DEVICE_ERROR;
  for (size_t index = 0; index < tile->window_count; ++index)
    tile->active_buffers[index] = (atlas_host_buffer){
        tile->windows[index].address, ATLAS_ORDERED_K_INPUT_BYTES,
        ATLAS_HOST_DEVICE_READ};
  tile->active_buffers[tile->window_count] = (atlas_host_buffer){
      tile->output_address, ATLAS_ORDERED_K_OUTPUT_BYTES, ATLAS_HOST_DEVICE_WRITE};
  host_result = atlas_host_start(tile->host, tile->active_buffers,
                                 tile->window_count + 1);
  if (host_result == ATLAS_HOST_OK)
    host_result = atlas_host_wait(tile->host, tile->max_polls);
  if (device_result)
    *device_result = host_result;
  if (host_result != ATLAS_HOST_OK)
    return ATLAS_ORDERED_K_DEVICE_ERROR;
  if (tile->read(tile->memory_context, tile->output_address,
                 tile->scratch, ATLAS_ORDERED_K_OUTPUT_BYTES))
    return ATLAS_ORDERED_K_MEMORY_ERROR;
  for (size_t index = 0; index < ATLAS_ORDERED_K_OUTPUT_BYTES; ++index)
    physical_output[index] = tile->scratch[index];
  return ATLAS_ORDERED_K_OK;
}
