#ifndef ATLAS_ORDERED_MN_H
#define ATLAS_ORDERED_MN_H

#include "atlas_ordered_k.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Run one verified 32x32 ordered-K program over an exact MxN output grid.
 * Inputs are row-major raw FP8 A[M,K] and W[N,K]. The output is row-major
 * BF16, with two bytes per element. The caller supplies separate staging and
 * physical-tile buffers; logical_output is unchanged on failure. One launch
 * is issued per output tile, in row-major tile order. device_result must not
 * alias any input or workspace buffer. */
atlas_ordered_k_status atlas_ordered_mn_run(
    atlas_ordered_k_tile *tile,
    const uint8_t *activation, size_t activation_bytes,
    const uint8_t *weight, size_t weight_bytes,
    size_t m_rows, size_t n_rows, size_t k_columns,
    uint8_t *logical_output, size_t logical_output_bytes,
    uint8_t *staging, size_t staging_bytes,
    uint8_t *physical_tile, size_t physical_tile_bytes,
    atlas_host_result *device_result);

/* The same program also covers partial M/N tiles when the caller supplies
 * two full 32xK host buffers. Missing rows are filled with positive FP8 zero;
 * only valid output cells are copied back. A short first K tile is filled on
 * the left with positive zero before the source reduction elements. */
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
    atlas_host_result *device_result);

#ifdef __cplusplus
}
#endif
#endif
