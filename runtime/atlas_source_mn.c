#include "atlas_source_mn.h"

static int valid_span(const void *pointer, size_t bytes) {
  return pointer && bytes && bytes <= UINTPTR_MAX - (uintptr_t)pointer;
}

static int overlap(const void *left, size_t left_bytes,
                   const void *right, size_t right_bytes) {
  uintptr_t a = (uintptr_t)left, b = (uintptr_t)right;
  return a < b + right_bytes && b < a + left_bytes;
}

atlas_ordered_k_status atlas_source_mn_run_f32(
    atlas_ordered_k_tile *tile,
    const float *lhs, size_t lhs_bytes,
    const float *rhs, size_t rhs_bytes,
    size_t m_rows, size_t n_rows, size_t k_columns,
    atlas_source_mn_encode encode_lhs, const void *lhs_policy,
    atlas_source_mn_encode encode_rhs, const void *rhs_policy,
    float *result_f32, size_t result_bytes,
    uint8_t *lhs_codes, size_t lhs_code_bytes,
    uint8_t *rhs_codes, size_t rhs_code_bytes,
    uint8_t *logical_bf16, size_t logical_bf16_bytes,
    uint8_t *staging, size_t staging_bytes,
    uint8_t *physical_tile, size_t physical_tile_bytes,
    uint8_t *lhs_tile, size_t lhs_tile_bytes,
    uint8_t *rhs_tile, size_t rhs_tile_bytes,
    atlas_host_result *device_result) {
  if (!tile || !tile->host || tile->host->active || !encode_lhs || !encode_rhs ||
      !m_rows || !n_rows || !k_columns ||
      m_rows > SIZE_MAX / k_columns || n_rows > SIZE_MAX / k_columns ||
      m_rows * k_columns > SIZE_MAX / sizeof(float) ||
      n_rows * k_columns > SIZE_MAX / sizeof(float) ||
      m_rows > SIZE_MAX / n_rows || m_rows * n_rows > SIZE_MAX / sizeof(float) ||
      m_rows * n_rows > SIZE_MAX / 2 ||
      k_columns > SIZE_MAX / 32 ||
      lhs_bytes != m_rows * k_columns * sizeof(float) ||
      rhs_bytes != n_rows * k_columns * sizeof(float) ||
      result_bytes != m_rows * n_rows * sizeof(float) ||
      lhs_code_bytes != m_rows * k_columns ||
      rhs_code_bytes != n_rows * k_columns ||
      logical_bf16_bytes != m_rows * n_rows * 2 ||
      staging_bytes != logical_bf16_bytes ||
      physical_tile_bytes != ATLAS_BF16_TILE_BYTES ||
      lhs_tile_bytes != 32 * k_columns || rhs_tile_bytes != 32 * k_columns ||
      !valid_span(lhs, lhs_bytes) || !valid_span(rhs, rhs_bytes) ||
      !valid_span(result_f32, result_bytes) ||
      !valid_span(lhs_codes, lhs_code_bytes) ||
      !valid_span(rhs_codes, rhs_code_bytes) ||
      !valid_span(logical_bf16, logical_bf16_bytes) ||
      !valid_span(staging, staging_bytes) ||
      !valid_span(physical_tile, physical_tile_bytes) ||
      !valid_span(lhs_tile, lhs_tile_bytes) || !valid_span(rhs_tile, rhs_tile_bytes) ||
      !valid_span(tile->scratch, tile->scratch_bytes) ||
      ((uintptr_t)lhs % _Alignof(float)) ||
      ((uintptr_t)rhs % _Alignof(float)) ||
      ((uintptr_t)result_f32 % _Alignof(float)))
    return ATLAS_ORDERED_K_INVALID;

  const void *writable[] = {result_f32, lhs_codes, rhs_codes, logical_bf16,
                            staging, physical_tile, lhs_tile, rhs_tile,
                            tile->scratch};
  size_t sizes[] = {result_bytes, lhs_code_bytes, rhs_code_bytes,
                    logical_bf16_bytes, staging_bytes, physical_tile_bytes,
                    lhs_tile_bytes, rhs_tile_bytes, tile->scratch_bytes};
  for (size_t i = 0; i < sizeof writable / sizeof writable[0]; ++i) {
    if (overlap(lhs, lhs_bytes, writable[i], sizes[i]) ||
        overlap(rhs, rhs_bytes, writable[i], sizes[i]))
      return ATLAS_ORDERED_K_INVALID;
    for (size_t j = i + 1; j < sizeof writable / sizeof writable[0]; ++j)
      if (overlap(writable[i], sizes[i], writable[j], sizes[j]))
        return ATLAS_ORDERED_K_INVALID;
  }
  if (device_result) {
    if (!valid_span(device_result, sizeof *device_result) ||
        overlap(device_result, sizeof *device_result, lhs, lhs_bytes) ||
        overlap(device_result, sizeof *device_result, rhs, rhs_bytes))
      return ATLAS_ORDERED_K_INVALID;
    for (size_t i = 0; i < sizeof writable / sizeof writable[0]; ++i)
      if (overlap(device_result, sizeof *device_result, writable[i], sizes[i]))
        return ATLAS_ORDERED_K_INVALID;
    *device_result = ATLAS_HOST_INVALID;
  }

  /* Encode the complete invocation before issuing a single Atlas command. */
  for (size_t row = 0; row < m_rows; ++row)
    for (size_t k = 0; k < k_columns; ++k) {
      uint32_t bits;
      __builtin_memcpy(&bits, &lhs[row * k_columns + k], sizeof bits);
      int status = encode_lhs(bits, lhs_policy, &lhs_codes[row * k_columns + k]);
      if (status)
        return status == 2 ? ATLAS_ORDERED_K_INPUT_DOMAIN : ATLAS_ORDERED_K_INVALID;
    }
  for (size_t k = 0; k < k_columns; ++k)
    for (size_t col = 0; col < n_rows; ++col) {
      uint32_t bits;
      __builtin_memcpy(&bits, &rhs[k * n_rows + col], sizeof bits);
      int status = encode_rhs(bits, rhs_policy, &rhs_codes[col * k_columns + k]);
      if (status)
        return status == 2 ? ATLAS_ORDERED_K_INPUT_DOMAIN : ATLAS_ORDERED_K_INVALID;
    }

  atlas_ordered_k_status status = atlas_ordered_mn_run_tails(
      tile, lhs_codes, lhs_code_bytes, rhs_codes, rhs_code_bytes,
      m_rows, n_rows, k_columns, logical_bf16, logical_bf16_bytes,
      staging, staging_bytes, physical_tile, physical_tile_bytes,
      lhs_tile, lhs_tile_bytes, rhs_tile, rhs_tile_bytes, device_result);
  if (status != ATLAS_ORDERED_K_OK)
    return status;
  for (size_t i = 0; i < m_rows * n_rows; ++i) {
    uint16_t code = (uint16_t)logical_bf16[2 * i] |
                    ((uint16_t)logical_bf16[2 * i + 1] << 8);
    uint32_t bits = (uint32_t)code << 16;
    __builtin_memcpy(&result_f32[i], &bits, sizeof bits);
  }
  return ATLAS_ORDERED_K_OK;
}

atlas_ordered_k_status atlas_source_batched_mn_run_f32(
    atlas_ordered_k_tile *tile,
    const float *lhs, size_t lhs_bytes,
    const float *rhs, size_t rhs_bytes,
    size_t batch_count, size_t m_rows, size_t n_rows, size_t k_columns,
    atlas_source_mn_encode encode_lhs, const void *lhs_policy,
    atlas_source_mn_encode encode_rhs, const void *rhs_policy,
    float *result_f32, size_t result_bytes,
    float *staged_result_f32, size_t staged_result_bytes,
    uint8_t *lhs_codes, size_t lhs_code_bytes,
    uint8_t *rhs_codes, size_t rhs_code_bytes,
    uint8_t *logical_bf16, size_t logical_bf16_bytes,
    uint8_t *staging, size_t staging_bytes,
    uint8_t *physical_tile, size_t physical_tile_bytes,
    uint8_t *lhs_tile, size_t lhs_tile_bytes,
    uint8_t *rhs_tile, size_t rhs_tile_bytes,
    atlas_host_result *device_result) {
  if (!tile || !tile->host || tile->host->active || !encode_lhs || !encode_rhs ||
      !batch_count || !m_rows || !n_rows || !k_columns ||
      m_rows > SIZE_MAX / k_columns || n_rows > SIZE_MAX / k_columns ||
      m_rows > SIZE_MAX / n_rows || k_columns > SIZE_MAX / 32)
    return ATLAS_ORDERED_K_INVALID;

  size_t lhs_cells = m_rows * k_columns;
  size_t rhs_cells = n_rows * k_columns;
  size_t result_cells = m_rows * n_rows;
  if (lhs_cells > SIZE_MAX / sizeof(float) ||
      rhs_cells > SIZE_MAX / sizeof(float) ||
      result_cells > SIZE_MAX / sizeof(float) ||
      result_cells > SIZE_MAX / 2 ||
      batch_count > SIZE_MAX / lhs_cells / sizeof(float) ||
      batch_count > SIZE_MAX / rhs_cells / sizeof(float) ||
      batch_count > SIZE_MAX / result_cells / sizeof(float) ||
      lhs_bytes != batch_count * lhs_cells * sizeof(float) ||
      rhs_bytes != batch_count * rhs_cells * sizeof(float) ||
      result_bytes != batch_count * result_cells * sizeof(float) ||
      staged_result_bytes != result_bytes ||
      lhs_code_bytes != lhs_cells || rhs_code_bytes != rhs_cells ||
      logical_bf16_bytes != result_cells * 2 ||
      staging_bytes != logical_bf16_bytes ||
      physical_tile_bytes != ATLAS_BF16_TILE_BYTES ||
      lhs_tile_bytes != 32 * k_columns || rhs_tile_bytes != 32 * k_columns ||
      !valid_span(lhs, lhs_bytes) || !valid_span(rhs, rhs_bytes) ||
      !valid_span(result_f32, result_bytes) ||
      !valid_span(staged_result_f32, staged_result_bytes) ||
      !valid_span(lhs_codes, lhs_code_bytes) ||
      !valid_span(rhs_codes, rhs_code_bytes) ||
      !valid_span(logical_bf16, logical_bf16_bytes) ||
      !valid_span(staging, staging_bytes) ||
      !valid_span(physical_tile, physical_tile_bytes) ||
      !valid_span(lhs_tile, lhs_tile_bytes) ||
      !valid_span(rhs_tile, rhs_tile_bytes) ||
      !valid_span(tile->scratch, tile->scratch_bytes) ||
      ((uintptr_t)lhs % _Alignof(float)) ||
      ((uintptr_t)rhs % _Alignof(float)) ||
      ((uintptr_t)result_f32 % _Alignof(float)) ||
      ((uintptr_t)staged_result_f32 % _Alignof(float)))
    return ATLAS_ORDERED_K_INVALID;

  const void *writable[] = {
      result_f32, staged_result_f32, lhs_codes, rhs_codes, logical_bf16,
      staging, physical_tile, lhs_tile, rhs_tile, tile->scratch};
  size_t sizes[] = {
      result_bytes, staged_result_bytes, lhs_code_bytes, rhs_code_bytes,
      logical_bf16_bytes, staging_bytes, physical_tile_bytes,
      lhs_tile_bytes, rhs_tile_bytes, tile->scratch_bytes};
  for (size_t i = 0; i < sizeof writable / sizeof writable[0]; ++i) {
    if (overlap(lhs, lhs_bytes, writable[i], sizes[i]) ||
        overlap(rhs, rhs_bytes, writable[i], sizes[i]))
      return ATLAS_ORDERED_K_INVALID;
    for (size_t j = i + 1; j < sizeof writable / sizeof writable[0]; ++j)
      if (overlap(writable[i], sizes[i], writable[j], sizes[j]))
        return ATLAS_ORDERED_K_INVALID;
  }
  if (device_result) {
    if (!valid_span(device_result, sizeof *device_result) ||
        overlap(device_result, sizeof *device_result, lhs, lhs_bytes) ||
        overlap(device_result, sizeof *device_result, rhs, rhs_bytes))
      return ATLAS_ORDERED_K_INVALID;
    for (size_t i = 0; i < sizeof writable / sizeof writable[0]; ++i)
      if (overlap(device_result, sizeof *device_result, writable[i], sizes[i]))
        return ATLAS_ORDERED_K_INVALID;
    *device_result = ATLAS_HOST_INVALID;
  }

  size_t lhs_batch_bytes = lhs_cells * sizeof(float);
  size_t rhs_batch_bytes = rhs_cells * sizeof(float);
  size_t result_batch_bytes = result_cells * sizeof(float);
  for (size_t batch = 0; batch < batch_count; ++batch) {
    atlas_ordered_k_status status = atlas_source_mn_run_f32(
        tile, lhs + batch * lhs_cells, lhs_batch_bytes,
        rhs + batch * rhs_cells, rhs_batch_bytes,
        m_rows, n_rows, k_columns,
        encode_lhs, lhs_policy, encode_rhs, rhs_policy,
        staged_result_f32 + batch * result_cells, result_batch_bytes,
        lhs_codes, lhs_code_bytes, rhs_codes, rhs_code_bytes,
        logical_bf16, logical_bf16_bytes, staging, staging_bytes,
        physical_tile, physical_tile_bytes, lhs_tile, lhs_tile_bytes,
        rhs_tile, rhs_tile_bytes, device_result);
    if (status != ATLAS_ORDERED_K_OK)
      return status;
  }
  __builtin_memcpy(result_f32, staged_result_f32, result_bytes);
  return ATLAS_ORDERED_K_OK;
}
