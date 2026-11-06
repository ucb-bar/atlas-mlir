#include "atlas_ordered_mn.h"

#include <stdint.h>

enum { EDGE = 32, HALF_ROW_BYTES = 32, PHYSICAL_TILE_BYTES = ATLAS_BF16_TILE_BYTES };

static int valid_span(const void *pointer, size_t bytes) {
  return pointer && bytes && bytes <= UINTPTR_MAX - (uintptr_t)pointer;
}

static int overlap(const void *left, size_t left_bytes,
                   const void *right, size_t right_bytes) {
  uintptr_t a = (uintptr_t)left;
  uintptr_t b = (uintptr_t)right;
  return a < b + right_bytes && b < a + left_bytes;
}

static atlas_ordered_k_status run_grid(
    atlas_ordered_k_tile *tile,
    const uint8_t *activation, size_t activation_bytes,
    const uint8_t *weight, size_t weight_bytes,
    size_t m_rows, size_t n_rows, size_t k_columns,
    uint8_t *logical_output, size_t logical_output_bytes,
    uint8_t *staging, size_t staging_bytes,
    uint8_t *physical_tile, size_t physical_tile_bytes,
    uint8_t *activation_tile, size_t activation_tile_bytes,
    uint8_t *weight_tile, size_t weight_tile_bytes,
    int allow_tails,
    atlas_host_result *device_result) {
  if (!tile || !tile->host || tile->host->active ||
      !m_rows || !n_rows || !k_columns ||
      (!allow_tails && (m_rows % EDGE || n_rows % EDGE)) ||
      tile->window_count < 2 || (tile->window_count & 1u) ||
      tile->window_count / 2 > SIZE_MAX / EDGE ||
      !tile->windows || k_columns != tile->windows[0].source_columns ||
      tile->window_count != 2 * (k_columns / EDGE + (k_columns % EDGE != 0)) ||
      k_columns > SIZE_MAX / EDGE ||
      m_rows > SIZE_MAX / k_columns || n_rows > SIZE_MAX / k_columns ||
      n_rows > SIZE_MAX / m_rows || m_rows * n_rows > SIZE_MAX / 2 ||
      activation_bytes != m_rows * k_columns ||
      weight_bytes != n_rows * k_columns ||
      logical_output_bytes != m_rows * n_rows * 2 ||
      staging_bytes != logical_output_bytes ||
      physical_tile_bytes != PHYSICAL_TILE_BYTES ||
      !valid_span(activation, activation_bytes) ||
      !valid_span(weight, weight_bytes) ||
      !valid_span(logical_output, logical_output_bytes) ||
      !valid_span(staging, staging_bytes) ||
      !valid_span(physical_tile, physical_tile_bytes) ||
      !valid_span(tile->scratch, tile->scratch_bytes) ||
      (allow_tails && (
          activation_tile_bytes != EDGE * k_columns ||
          weight_tile_bytes != EDGE * k_columns ||
          !valid_span(activation_tile, activation_tile_bytes) ||
          !valid_span(weight_tile, weight_tile_bytes))))
    return ATLAS_ORDERED_K_INVALID;

  if (device_result && (
      !valid_span(device_result, sizeof *device_result) ||
      overlap(device_result, sizeof *device_result, activation, activation_bytes) ||
      overlap(device_result, sizeof *device_result, weight, weight_bytes) ||
      overlap(device_result, sizeof *device_result, logical_output, logical_output_bytes) ||
      overlap(device_result, sizeof *device_result, staging, staging_bytes) ||
      overlap(device_result, sizeof *device_result, physical_tile, physical_tile_bytes) ||
      overlap(device_result, sizeof *device_result, tile->scratch, tile->scratch_bytes) ||
      (allow_tails && (
          overlap(device_result, sizeof *device_result,
                  activation_tile, activation_tile_bytes) ||
          overlap(device_result, sizeof *device_result,
                  weight_tile, weight_tile_bytes)))))
    return ATLAS_ORDERED_K_INVALID;
  if (device_result)
    *device_result = ATLAS_HOST_INVALID;

  const void *written[] = {logical_output, staging, physical_tile, tile->scratch,
                           activation_tile, weight_tile};
  size_t written_bytes[] = {
      logical_output_bytes, staging_bytes, physical_tile_bytes, tile->scratch_bytes,
      activation_tile_bytes, weight_tile_bytes};
  size_t written_count = allow_tails ? 6 : 4;
  for (size_t i = 0; i < written_count; ++i) {
    if (overlap(activation, activation_bytes, written[i], written_bytes[i]) ||
        overlap(weight, weight_bytes, written[i], written_bytes[i]))
      return ATLAS_ORDERED_K_INVALID;
    for (size_t j = i + 1; j < written_count; ++j)
      if (overlap(written[i], written_bytes[i], written[j], written_bytes[j]))
        return ATLAS_ORDERED_K_INVALID;
  }

  /* Check the entire invocation before any device side effect. */
  for (size_t i = 0; i < activation_bytes; ++i)
    if (!atlas_ordered_k_admit_code(activation[i]))
      return ATLAS_ORDERED_K_INPUT_DOMAIN;
  for (size_t i = 0; i < weight_bytes; ++i)
    if (!atlas_ordered_k_admit_code(weight[i]))
      return ATLAS_ORDERED_K_INPUT_DOMAIN;

  for (size_t m = 0; m < m_rows; m += EDGE) {
    for (size_t n = 0; n < n_rows; n += EDGE) {
      const uint8_t *activation_input = activation + m * k_columns;
      const uint8_t *weight_input = weight + n * k_columns;
      if (allow_tails) {
        for (size_t row = 0; row < EDGE; ++row)
          for (size_t column = 0; column < k_columns; ++column) {
            activation_tile[row * k_columns + column] =
                row < m_rows - m ? activation_input[row * k_columns + column] : 0;
            weight_tile[row * k_columns + column] =
                row < n_rows - n ? weight_input[row * k_columns + column] : 0;
          }
        activation_input = activation_tile;
        weight_input = weight_tile;
      }
      atlas_ordered_k_status status = atlas_ordered_k_run(
          tile, activation_input, EDGE * k_columns,
          weight_input, EDGE * k_columns,
          physical_tile, physical_tile_bytes, device_result);
      if (status != ATLAS_ORDERED_K_OK)
        return status;
      size_t valid_m = m_rows - m < EDGE ? m_rows - m : EDGE;
      size_t valid_n = n_rows - n < EDGE ? n_rows - n : EDGE;
      for (size_t row = 0; row < valid_m; ++row) {
        for (size_t col = 0; col < valid_n; ++col) {
          size_t destination = ((m + row) * n_rows + n + col) * 2;
          size_t source = row * HALF_ROW_BYTES + (col % 16) * 2;
          if (col >= 16)
            source += ATLAS_BF16_HALF_BYTES;
          staging[destination] = physical_tile[source];
          staging[destination + 1] = physical_tile[source + 1];
        }
      }
    }
  }
  for (size_t byte = 0; byte < logical_output_bytes; ++byte)
    logical_output[byte] = staging[byte];
  return ATLAS_ORDERED_K_OK;
}

atlas_ordered_k_status atlas_ordered_mn_run(
    atlas_ordered_k_tile *tile,
    const uint8_t *activation, size_t activation_bytes,
    const uint8_t *weight, size_t weight_bytes,
    size_t m_rows, size_t n_rows, size_t k_columns,
    uint8_t *logical_output, size_t logical_output_bytes,
    uint8_t *staging, size_t staging_bytes,
    uint8_t *physical_tile, size_t physical_tile_bytes,
    atlas_host_result *device_result) {
  return run_grid(tile, activation, activation_bytes, weight, weight_bytes,
                  m_rows, n_rows, k_columns, logical_output, logical_output_bytes,
                  staging, staging_bytes, physical_tile, physical_tile_bytes,
                  0, 0, 0, 0, 0, device_result);
}

atlas_ordered_k_status atlas_ordered_mn_run_tails(
    atlas_ordered_k_tile *tile,
    const uint8_t *activation, size_t activation_bytes,
    const uint8_t *weight, size_t weight_bytes,
    size_t m_rows, size_t n_rows, size_t k_columns,
    uint8_t *logical_output, size_t logical_output_bytes,
    uint8_t *staging, size_t staging_bytes,
    uint8_t *physical_tile, size_t physical_tile_bytes,
    uint8_t *activation_tile, size_t activation_tile_bytes,
    uint8_t *weight_tile, size_t weight_tile_bytes,
    atlas_host_result *device_result) {
  return run_grid(tile, activation, activation_bytes, weight, weight_bytes,
                  m_rows, n_rows, k_columns, logical_output, logical_output_bytes,
                  staging, staging_bytes, physical_tile, physical_tile_bytes,
                  activation_tile, activation_tile_bytes,
                  weight_tile, weight_tile_bytes, 1, device_result);
}
