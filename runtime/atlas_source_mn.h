#ifndef ATLAS_SOURCE_MN_H
#define ATLAS_SOURCE_MN_H

#include "atlas_ordered_mn.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The selected precision policy supplies a pure binary32-bits to FP8-code
 * conversion. Return 0 on success, 1 for an invalid value/policy, or 2 for
 * a value outside the admitted domain. Merlin's finite-float encoder has
 * this interface through a small generated callback. */
typedef int (*atlas_source_mn_encode)(uint32_t bits, const void *policy,
                                      uint8_t *code);

/* Source A is row-major [M,K]; source RHS is row-major [K,N]. The selected
 * Atlas runtime consumes A[M,K] and transposed W[N,K]. All workspace sizes
 * are exact. The caller owns every buffer and must keep it live until return.
 * On failure result_f32 is unchanged. A device failure can leave its state
 * active; the caller must recover it before another invocation. */
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
    atlas_host_result *device_result);

/* Source tensors are contiguous row-major A[B,M,K] and RHS[B,K,N]. One
 * checked tile image is launched serially for each batch, and the complete
 * result is copied to result_f32 only after every batch succeeds. The caller
 * supplies separate full-result staging so a later failure leaves the result
 * unchanged. Tile workspace sizes are for one batch. */
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
    atlas_host_result *device_result);

#ifdef __cplusplus
}
#endif
#endif
