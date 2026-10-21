#include "atlas_model_tile.h"

#if defined(ATLAS_FREESTANDING)
/* The Shuttle build supplies these C runtime symbols. This declaration does
 * not select an allocator or claim that a bare-metal image can link yet. */
extern void *malloc(size_t bytes);
extern void free(void *address);
extern int memcmp(const void *left, const void *right, size_t bytes);
#else
#include <stdlib.h>
#include <string.h>
#endif

__attribute__((weak, noreturn)) void atlas_model_tile_fail(
    atlas_model_tile_status status, atlas_host_result device_result) {
  (void)status;
  (void)device_result;
  __builtin_trap();
}

static int dense_input(const atlas_f32_memref2 *input, float *dense, unsigned transpose) {
  if (!input || !input->aligned || input->offset < 0 ||
      transpose > 1 ||
      input->sizes[0] != ATLAS_MODEL_TILE_EDGE ||
      input->sizes[1] != ATLAS_MODEL_TILE_EDGE ||
      input->strides[0] <= 0 || input->strides[1] <= 0 ||
      input->strides[0] > (INT64_MAX - input->offset) / (ATLAS_MODEL_TILE_EDGE - 1) ||
      input->strides[1] >
          (INT64_MAX - input->offset - (ATLAS_MODEL_TILE_EDGE - 1) * input->strides[0]) /
              (ATLAS_MODEL_TILE_EDGE - 1))
    return 0;
  for (int64_t row = 0; row < ATLAS_MODEL_TILE_EDGE; ++row)
    for (int64_t column = 0; column < ATLAS_MODEL_TILE_EDGE; ++column)
      dense[(transpose ? column * ATLAS_MODEL_TILE_EDGE + row
                       : row * ATLAS_MODEL_TILE_EDGE + column)] =
          input->aligned[input->offset + row * input->strides[0] +
                         column * input->strides[1]];
  return 1;
}

static int valid_range(uintptr_t address, size_t bytes) {
  return address && address <= UINT32_MAX && bytes && bytes - 1 <= UINT32_MAX - address;
}

static int overlaps(uintptr_t first, size_t first_bytes, uintptr_t second, size_t second_bytes) {
  return first <= second ? second - first < first_bytes : first - second < second_bytes;
}

atlas_model_tile_status atlas_model_tile_run(atlas_model_tile *binding,
    const atlas_f32_memref2 *lhs, const atlas_f32_memref2 *rhs,
    atlas_f32_memref2 *result, atlas_host_result *device_result) {
  if (device_result)
    *device_result = ATLAS_HOST_INVALID;
  if (!binding || !binding->host || !binding->io.write || !binding->io.read ||
      !binding->program_words || !binding->program_word_count ||
      binding->constant_operand > 2 ||
      (binding->constant_operand < 2 &&
       (!binding->constant_bytes ||
        binding->constant_bytes_length != ATLAS_MODEL_TILE_FP8_BYTES)) ||
      (binding->constant_operand == 2 &&
       (binding->constant_bytes || binding->constant_bytes_length)) ||
      !binding->pack[0] || !binding->pack[1] || !binding->unpack ||
      !binding->max_polls || !binding->scratch ||
      binding->scratch_bytes < ATLAS_MODEL_TILE_SCRATCH_BYTES ||
      ((uintptr_t)binding->scratch & (_Alignof(float) - 1)) ||
      !result || !lhs || !rhs ||
      !valid_range(binding->input_addresses[0], ATLAS_MODEL_TILE_FP8_BYTES) ||
      !valid_range(binding->input_addresses[1], ATLAS_MODEL_TILE_FP8_BYTES) ||
      !valid_range(binding->output_address, ATLAS_MODEL_TILE_BF16_BYTES) ||
      overlaps(binding->input_addresses[0], ATLAS_MODEL_TILE_FP8_BYTES,
               binding->input_addresses[1], ATLAS_MODEL_TILE_FP8_BYTES) ||
      overlaps(binding->input_addresses[0], ATLAS_MODEL_TILE_FP8_BYTES,
               binding->output_address, ATLAS_MODEL_TILE_BF16_BYTES) ||
      overlaps(binding->input_addresses[1], ATLAS_MODEL_TILE_FP8_BYTES,
               binding->output_address, ATLAS_MODEL_TILE_BF16_BYTES) ||
      binding->host->active)
    return ATLAS_MODEL_TILE_INVALID;

  uint8_t *scratch = (uint8_t *)binding->scratch;
  float *dense = (float *)scratch;
  uint8_t *packed = scratch + 4096;
  uint8_t *constant_check = packed + ATLAS_MODEL_TILE_FP8_BYTES;
  uint8_t *physical = constant_check + ATLAS_MODEL_TILE_FP8_BYTES;
  const atlas_f32_memref2 *inputs[2] = {lhs, rhs};
  for (unsigned index = 0; index < 2; ++index) {
    if (!dense_input(inputs[index], dense, binding->transpose_input[index]))
      return ATLAS_MODEL_TILE_INVALID;
    uint8_t *destination = index ? constant_check : packed;
    if (binding->pack[index](dense, destination))
      return ATLAS_MODEL_TILE_INPUT_DOMAIN;
  }
  if (binding->constant_operand < 2 &&
      memcmp(binding->constant_operand ? constant_check : packed,
             binding->constant_bytes, ATLAS_MODEL_TILE_FP8_BYTES))
    return ATLAS_MODEL_TILE_CONSTANT_MISMATCH;

  for (unsigned index = 0; index < 2; ++index) {
    const uint8_t *data = index == binding->constant_operand ? binding->constant_bytes :
                          index ? constant_check : packed;
    if (binding->io.write(binding->io.context, binding->input_addresses[index],
                          data, ATLAS_MODEL_TILE_FP8_BYTES))
      return ATLAS_MODEL_TILE_MEMORY_ERROR;
  }

  atlas_host_result host_status = atlas_host_ensure_loaded(binding->host, binding->program_words,
                                                 binding->program_word_count);
  if (device_result)
    *device_result = host_status;
  if (host_status != ATLAS_HOST_OK)
    return ATLAS_MODEL_TILE_DEVICE_ERROR;

  binding->active_buffers[0] = (atlas_host_buffer){binding->input_addresses[0],
      ATLAS_MODEL_TILE_FP8_BYTES, ATLAS_HOST_DEVICE_READ};
  binding->active_buffers[1] = (atlas_host_buffer){binding->input_addresses[1],
      ATLAS_MODEL_TILE_FP8_BYTES, ATLAS_HOST_DEVICE_READ};
  binding->active_buffers[2] = (atlas_host_buffer){binding->output_address,
      ATLAS_MODEL_TILE_BF16_BYTES, ATLAS_HOST_DEVICE_WRITE};
  host_status = atlas_host_start(binding->host, binding->active_buffers, 3);
  if (host_status == ATLAS_HOST_OK)
    host_status = atlas_host_wait(binding->host, binding->max_polls);
  if (device_result)
    *device_result = host_status;
  if (host_status != ATLAS_HOST_OK)
    return ATLAS_MODEL_TILE_DEVICE_ERROR;
  if (binding->io.read(binding->io.context, binding->output_address,
                       physical, ATLAS_MODEL_TILE_BF16_BYTES))
    return ATLAS_MODEL_TILE_MEMORY_ERROR;

  float *output = (float *)malloc(ATLAS_MODEL_TILE_EDGE * ATLAS_MODEL_TILE_EDGE * sizeof(float));
  if (!output)
    return ATLAS_MODEL_TILE_ALLOC_ERROR;
  if (binding->unpack(physical, output)) {
    free(output);
    return ATLAS_MODEL_TILE_INPUT_DOMAIN;
  }
  *result = (atlas_f32_memref2){output, output, 0,
      {ATLAS_MODEL_TILE_EDGE, ATLAS_MODEL_TILE_EDGE}, {ATLAS_MODEL_TILE_EDGE, 1}};
  return ATLAS_MODEL_TILE_OK;
}

static int valid_window_input(const atlas_f32_memref2 *input,
                              size_t rows, size_t columns) {
  if (!input || !input->aligned || input->offset < 0 ||
      input->sizes[0] != (int64_t)rows || input->sizes[1] != (int64_t)columns ||
      input->strides[0] <= 0 || input->strides[1] <= 0)
    return 0;
  int64_t last = INT64_MAX - input->offset;
  if (rows > 1) {
    if (input->strides[0] > last / (int64_t)(rows - 1)) return 0;
    last -= (int64_t)(rows - 1) * input->strides[0];
  }
  return columns <= 1 || input->strides[1] <= last / (int64_t)(columns - 1);
}

atlas_model_tile_status atlas_model_tile_run_window(atlas_model_tile *binding,
    const atlas_f32_memref2 *lhs, const atlas_f32_memref2 *rhs,
    atlas_f32_memref2 *result, size_t m, size_t n, size_t k,
    atlas_host_result *device_result) {
  if (device_result) *device_result = ATLAS_HOST_INVALID;
  if (!result || !m || !n || !k || m > ATLAS_MODEL_TILE_EDGE ||
      n > ATLAS_MODEL_TILE_EDGE || k > ATLAS_MODEL_TILE_EDGE ||
      !valid_window_input(lhs, m, k) || !valid_window_input(rhs, k, n))
    return ATLAS_MODEL_TILE_INVALID;

  float *source = (float *)malloc(2 * ATLAS_MODEL_TILE_EDGE *
                                  ATLAS_MODEL_TILE_EDGE * sizeof(float));
  if (!source) return ATLAS_MODEL_TILE_ALLOC_ERROR;
  for (size_t index = 0; index < 2 * ATLAS_MODEL_TILE_EDGE * ATLAS_MODEL_TILE_EDGE; ++index)
    source[index] = 0.0f;
  float *lhs_full = source;
  float *rhs_full = source + ATLAS_MODEL_TILE_EDGE * ATLAS_MODEL_TILE_EDGE;
  const size_t k_start = ATLAS_MODEL_TILE_EDGE - k;
  for (size_t row = 0; row < m; ++row)
    for (size_t column = 0; column < k; ++column)
      lhs_full[row * ATLAS_MODEL_TILE_EDGE + k_start + column] =
          lhs->aligned[lhs->offset + (int64_t)row * lhs->strides[0] +
                       (int64_t)column * lhs->strides[1]];
  for (size_t row = 0; row < k; ++row)
    for (size_t column = 0; column < n; ++column)
      rhs_full[(k_start + row) * ATLAS_MODEL_TILE_EDGE + column] =
          rhs->aligned[rhs->offset + (int64_t)row * rhs->strides[0] +
                       (int64_t)column * rhs->strides[1]];
  atlas_f32_memref2 lhs_tile = {lhs_full, lhs_full, 0,
      {ATLAS_MODEL_TILE_EDGE, ATLAS_MODEL_TILE_EDGE}, {ATLAS_MODEL_TILE_EDGE, 1}};
  atlas_f32_memref2 rhs_tile = {rhs_full, rhs_full, 0,
      {ATLAS_MODEL_TILE_EDGE, ATLAS_MODEL_TILE_EDGE}, {ATLAS_MODEL_TILE_EDGE, 1}};
  atlas_f32_memref2 physical = {0};
  atlas_model_tile_status status = atlas_model_tile_run(
      binding, &lhs_tile, &rhs_tile, &physical, device_result);
  free(source);
  if (status != ATLAS_MODEL_TILE_OK) return status;
  float *logical = (float *)malloc(m * n * sizeof(float));
  if (!logical) {
    free(physical.allocated);
    return ATLAS_MODEL_TILE_ALLOC_ERROR;
  }
  for (size_t row = 0; row < m; ++row)
    for (size_t column = 0; column < n; ++column)
      logical[row * n + column] = physical.aligned[
          physical.offset + (int64_t)row * physical.strides[0] +
          (int64_t)column * physical.strides[1]];
  free(physical.allocated);
  *result = (atlas_f32_memref2){logical, logical, 0,
      {(int64_t)m, (int64_t)n}, {(int64_t)n, 1}};
  return ATLAS_MODEL_TILE_OK;
}

int atlas_model_tile_direct_write(void *context, uintptr_t address,
                                  const uint8_t *source, size_t bytes) {
  (void)context;
  if (!address || !source)
    return 1;
  volatile uint8_t *destination = (volatile uint8_t *)address;
  for (size_t index = 0; index < bytes; ++index)
    destination[index] = source[index];
  __sync_synchronize();
  return 0;
}

int atlas_model_tile_direct_read(void *context, uintptr_t address,
                                 uint8_t *destination, size_t bytes) {
  (void)context;
  if (!address || !destination)
    return 1;
  __sync_synchronize();
  volatile const uint8_t *source = (volatile const uint8_t *)address;
  for (size_t index = 0; index < bytes; ++index)
    destination[index] = source[index];
  return 0;
}
