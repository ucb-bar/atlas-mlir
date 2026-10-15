/* Diagnostic Shuttle host for a verified source-bound f32 matmul grid. */
#include "atlas_source_mn_data.h"

#include <stddef.h>
#include <stdint.h>

void *memset(void *destination, int value, size_t count) {
  unsigned char *bytes = destination;
  for (size_t index = 0; index < count; ++index)
    bytes[index] = (unsigned char)value;
  return destination;
}

void *memcpy(void *destination, const void *source, size_t count) {
  unsigned char *target = destination;
  const unsigned char *origin = source;
  for (size_t index = 0; index < count; ++index)
    target[index] = origin[index];
  return destination;
}

enum {
  B = ATLAS_SOURCE_MN_BATCHES,
  M = ATLAS_SOURCE_MN_ROWS, N = ATLAS_SOURCE_MN_COLUMNS,
  K = ATLAS_SOURCE_MN_REDUCTION,
};

static atlas_host host;
static atlas_ordered_k_tile tile;
static atlas_host_buffer active_buffers[ATLAS_SOURCE_MN_WINDOW_COUNT + 1u];
static float lhs[B * M * K], rhs[B * K * N];
static uint8_t lhs_codes[M * K], rhs_codes[N * K];
static uint8_t logical_bf16[2u * M * N], staging[2u * M * N];
static uint8_t physical_tile[2048], scratch[2048];
static uint8_t lhs_tile[32u * K], rhs_tile[32u * K];
static float result[B * M * N];
#if ATLAS_SOURCE_MN_BATCHES > 1
static float staged_result[B * M * N];
#endif

volatile uint32_t atlas_source_matrix_status_word;
volatile uint32_t atlas_source_matrix_device_word;
volatile float atlas_source_matrix_outputs[2][B * M * N];

static int direct_write(void *context, uintptr_t address,
                        const uint8_t *source, size_t bytes) {
  (void)context;
  if (!address || !source || !bytes)
    return 1;
  volatile uint8_t *destination = (volatile uint8_t *)address;
  for (size_t index = 0; index < bytes; ++index)
    destination[index] = source[index];
  __asm__ volatile("fence rw, rw" ::: "memory");
  return 0;
}

static int direct_read(void *context, uintptr_t address,
                       uint8_t *destination, size_t bytes) {
  (void)context;
  if (!address || !destination || !bytes)
    return 1;
  __asm__ volatile("fence rw, rw" ::: "memory");
  volatile const uint8_t *source = (volatile const uint8_t *)address;
  for (size_t index = 0; index < bytes; ++index)
    destination[index] = source[index];
  return 0;
}

/* Inputs are created at invocation time and never used to choose the program. */
static void prepare_inputs(unsigned trial) {
  static const float values[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
#if ATLAS_SOURCE_MN_FROZEN_RHS
  _Static_assert(sizeof(float) == sizeof(uint32_t), "frozen RHS requires f32");
  memcpy(rhs, atlas_source_mn_frozen_rhs_bits, sizeof(rhs));
#endif
  for (size_t batch = 0; batch < B; ++batch) {
    for (size_t row = 0; row < M; ++row)
      for (size_t k = 0; k < K; ++k)
        lhs[batch * M * K + row * K + k] =
            values[(row + k + batch + trial) % 5u];
#if !ATLAS_SOURCE_MN_FROZEN_RHS
    for (size_t k = 0; k < K; ++k)
      for (size_t column = 0; column < N; ++column)
        rhs[batch * K * N + k * N + column] =
            values[(2u * k + column + batch + trial) % 5u];
#endif
  }
}

int main(void) {
  atlas_host_mmio mmio = {atlas_host_direct_read32, atlas_host_direct_write32, 0};
  atlas_host_memory memory = {1, 0, 0}; /* Diagnostic coherence premise. */
  atlas_host_result device = atlas_host_init(&host, atlas_source_mn_host_map,
                                             mmio, memory);
  atlas_source_matrix_device_word = (uint32_t)device;
  if (device != ATLAS_HOST_OK)
    return atlas_source_matrix_status_word = 0x100u + (uint32_t)device;
  if (atlas_source_mn_bind(&tile, &host, 0, direct_write, direct_read,
                           10000000u, scratch, sizeof scratch,
                           active_buffers, ATLAS_SOURCE_MN_WINDOW_COUNT + 1u))
    return atlas_source_matrix_status_word = 0x200u;
  for (unsigned trial = 0; trial < 2; ++trial) {
    prepare_inputs(trial);
#if ATLAS_SOURCE_MN_BATCHES > 1
    atlas_ordered_k_status status = atlas_source_batched_mn_run_f32(
        &tile, lhs, sizeof lhs, rhs, sizeof rhs, B, M, N, K,
        atlas_source_mn_encode_lhs, 0, atlas_source_mn_encode_rhs, 0,
        result, sizeof result, staged_result, sizeof staged_result,
        lhs_codes, sizeof lhs_codes, rhs_codes, sizeof rhs_codes,
        logical_bf16, sizeof logical_bf16, staging, sizeof staging,
        physical_tile, sizeof physical_tile,
        lhs_tile, sizeof lhs_tile, rhs_tile, sizeof rhs_tile, &device);
#else
    atlas_ordered_k_status status = atlas_source_mn_run_f32(
        &tile, lhs, sizeof lhs, rhs, sizeof rhs, M, N, K,
        atlas_source_mn_encode_lhs, 0, atlas_source_mn_encode_rhs, 0,
        result, sizeof result,
        lhs_codes, sizeof lhs_codes, rhs_codes, sizeof rhs_codes,
        logical_bf16, sizeof logical_bf16, staging, sizeof staging,
        physical_tile, sizeof physical_tile,
        lhs_tile, sizeof lhs_tile, rhs_tile, sizeof rhs_tile, &device);
#endif
    atlas_source_matrix_device_word = (uint32_t)device;
    if (status != ATLAS_ORDERED_K_OK)
      return atlas_source_matrix_status_word = 0x300u + 16u * trial + (uint32_t)status;
    for (size_t index = 0; index < B * M * N; ++index)
      atlas_source_matrix_outputs[trial][index] = result[index];
  }
  return atlas_source_matrix_status_word = 1u;
}
