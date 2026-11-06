#include "atlas_source_mn_memref.h"

#if defined(ATLAS_FREESTANDING)
extern void *malloc(size_t bytes);
extern void free(void *address);
#else
#include <stdlib.h>
#endif

typedef struct {
  const float *aligned;
  int64_t offset;
  int64_t strides[3];
} source_view;

static int cells(size_t first, size_t second, size_t *product) {
  if (!first || !second || first > SIZE_MAX / second) return 0;
  *product = first * second;
  return 1;
}

static int valid_view(const float *aligned, int64_t offset,
                      const int64_t *sizes, const int64_t *strides,
                      size_t rank, const size_t *expected) {
  if (!aligned || offset < 0) return 0;
  int64_t last = INT64_MAX - offset;
  for (size_t axis = 0; axis < rank; ++axis) {
    if (!expected[axis] || expected[axis] > (size_t)INT64_MAX ||
        sizes[axis] != (int64_t)expected[axis] || strides[axis] <= 0)
      return 0;
    if (expected[axis] > 1u) {
      if (strides[axis] > last / (int64_t)(expected[axis] - 1u)) return 0;
      last -= (int64_t)(expected[axis] - 1u) * strides[axis];
    }
  }
  return 1;
}

static atlas_ordered_k_status run(
    const atlas_source_mn_memref_binding *binding,
    source_view lhs, source_view rhs, size_t batch_count,
    void *result, unsigned rank, atlas_host_result *device_result) {
  if (device_result) *device_result = ATLAS_HOST_INVALID;
  if (!binding || !binding->tile || !binding->encode_lhs || !binding->encode_rhs ||
      !result || !batch_count || !binding->m || !binding->n || !binding->k ||
      binding->batches != batch_count || binding->m > INT64_MAX ||
      binding->n > INT64_MAX || binding->k > INT64_MAX ||
      batch_count > INT64_MAX || binding->k > SIZE_MAX / 32u)
    return ATLAS_ORDERED_K_INVALID;

  size_t lhs_one, rhs_one, out_one, lhs_count, rhs_count, out_count;
  if (!cells(binding->m, binding->k, &lhs_one) ||
      !cells(binding->k, binding->n, &rhs_one) ||
      !cells(binding->m, binding->n, &out_one) ||
      !cells(batch_count, lhs_one, &lhs_count) ||
      !cells(batch_count, rhs_one, &rhs_count) ||
      !cells(batch_count, out_one, &out_count) ||
      lhs_count > SIZE_MAX / sizeof(float) ||
      rhs_count > SIZE_MAX / sizeof(float) ||
      out_count > SIZE_MAX / sizeof(float) ||
      out_one > SIZE_MAX / 2u || binding->k > SIZE_MAX / 32u)
    return ATLAS_ORDERED_K_INVALID;
  if ((binding->frozen_rhs_bits == NULL) != (binding->frozen_rhs_elements == 0) ||
      (binding->frozen_rhs_bits && binding->frozen_rhs_elements != rhs_count))
    return ATLAS_ORDERED_K_INVALID;

  float *dense_lhs = malloc(lhs_count * sizeof(float));
  float *dense_rhs = malloc(rhs_count * sizeof(float));
  float *output = malloc(out_count * sizeof(float));
  float *staged_result = rank == 3 ? malloc(out_count * sizeof(float)) : NULL;
  uint8_t *lhs_codes = malloc(lhs_one);
  uint8_t *rhs_codes = malloc(rhs_one);
  uint8_t *logical = malloc(out_one * 2u);
  uint8_t *staging = malloc(out_one * 2u);
  uint8_t *physical = malloc(ATLAS_BF16_TILE_BYTES);
  uint8_t *lhs_tile = malloc(binding->k * 32u);
  uint8_t *rhs_tile = malloc(binding->k * 32u);
  atlas_ordered_k_status status = ATLAS_ORDERED_K_MEMORY_ERROR;
  if (!dense_lhs || !dense_rhs || !output || (rank == 3 && !staged_result) ||
      !lhs_codes || !rhs_codes || !logical || !staging || !physical ||
      !lhs_tile || !rhs_tile)
    goto cleanup;

  for (size_t batch = 0; batch < batch_count; ++batch) {
    for (size_t row = 0; row < binding->m; ++row)
      for (size_t column = 0; column < binding->k; ++column)
        dense_lhs[batch * lhs_one + row * binding->k + column] = lhs.aligned[
            lhs.offset + (int64_t)batch * lhs.strides[0] +
            (int64_t)row * lhs.strides[1] + (int64_t)column * lhs.strides[2]];
    for (size_t row = 0; row < binding->k; ++row)
      for (size_t column = 0; column < binding->n; ++column)
        dense_rhs[batch * rhs_one + row * binding->n + column] = rhs.aligned[
            rhs.offset + (int64_t)batch * rhs.strides[0] +
            (int64_t)row * rhs.strides[1] + (int64_t)column * rhs.strides[2]];
  }
  if (binding->frozen_rhs_bits) {
    for (size_t index = 0; index < rhs_count; ++index) {
      uint32_t bits;
      __builtin_memcpy(&bits, &dense_rhs[index], sizeof bits);
      if (bits != binding->frozen_rhs_bits[index]) {
        status = ATLAS_ORDERED_K_CONSTANT_MISMATCH;
        goto cleanup;
      }
    }
  }
  if (rank == 3)
    status = atlas_source_batched_mn_run_f32(
        binding->tile, dense_lhs, lhs_count * sizeof(float),
        dense_rhs, rhs_count * sizeof(float), batch_count,
        binding->m, binding->n, binding->k,
        binding->encode_lhs, binding->lhs_policy,
        binding->encode_rhs, binding->rhs_policy,
        output, out_count * sizeof(float), staged_result, out_count * sizeof(float),
        lhs_codes, lhs_one, rhs_codes, rhs_one, logical, out_one * 2u,
        staging, out_one * 2u, physical, ATLAS_BF16_TILE_BYTES,
        lhs_tile, binding->k * 32u, rhs_tile, binding->k * 32u,
        device_result);
  else
    status = atlas_source_mn_run_f32(
        binding->tile, dense_lhs, lhs_count * sizeof(float),
        dense_rhs, rhs_count * sizeof(float), binding->m, binding->n, binding->k,
        binding->encode_lhs, binding->lhs_policy,
        binding->encode_rhs, binding->rhs_policy,
        output, out_count * sizeof(float),
        lhs_codes, lhs_one, rhs_codes, rhs_one, logical, out_one * 2u,
        staging, out_one * 2u, physical, ATLAS_BF16_TILE_BYTES,
        lhs_tile, binding->k * 32u, rhs_tile, binding->k * 32u,
        device_result);
  if (status == ATLAS_ORDERED_K_OK) {
    if (rank == 3)
      *(atlas_f32_memref3 *)result = (atlas_f32_memref3){
          output, output, 0,
          {(int64_t)batch_count, (int64_t)binding->m, (int64_t)binding->n},
          {(int64_t)out_one, (int64_t)binding->n, 1}};
    else
      *(atlas_f32_memref2 *)result = (atlas_f32_memref2){
          output, output, 0,
          {(int64_t)binding->m, (int64_t)binding->n},
          {(int64_t)binding->n, 1}};
    output = NULL;
  }

cleanup:
  free(dense_lhs);
  free(dense_rhs);
  free(output);
  free(staged_result);
  free(lhs_codes);
  free(rhs_codes);
  free(logical);
  free(staging);
  free(physical);
  free(lhs_tile);
  free(rhs_tile);
  return status;
}

atlas_ordered_k_status atlas_source_mn_run_memref2(
    const atlas_source_mn_memref_binding *binding,
    const atlas_f32_memref2 *lhs, const atlas_f32_memref2 *rhs,
    atlas_f32_memref2 *result, atlas_host_result *device_result) {
  if (device_result) *device_result = ATLAS_HOST_INVALID;
  if (!binding || !lhs || !rhs || !result || binding->batches != 1)
    return ATLAS_ORDERED_K_INVALID;
  size_t lhs_shape[] = {binding->m, binding->k};
  size_t rhs_shape[] = {binding->k, binding->n};
  if (!valid_view(lhs->aligned, lhs->offset, lhs->sizes, lhs->strides, 2, lhs_shape) ||
      !valid_view(rhs->aligned, rhs->offset, rhs->sizes, rhs->strides, 2, rhs_shape))
    return ATLAS_ORDERED_K_INVALID;
  source_view lhs_view = {lhs->aligned, lhs->offset, {0, lhs->strides[0], lhs->strides[1]}};
  source_view rhs_view = {rhs->aligned, rhs->offset, {0, rhs->strides[0], rhs->strides[1]}};
  return run(binding, lhs_view, rhs_view, 1, result, 2, device_result);
}

atlas_ordered_k_status atlas_source_mn_run_memref3(
    const atlas_source_mn_memref_binding *binding,
    const atlas_f32_memref3 *lhs, const atlas_f32_memref3 *rhs,
    atlas_f32_memref3 *result, atlas_host_result *device_result) {
  if (device_result) *device_result = ATLAS_HOST_INVALID;
  if (!binding || !lhs || !rhs || !result) return ATLAS_ORDERED_K_INVALID;
  size_t lhs_shape[] = {binding->batches, binding->m, binding->k};
  size_t rhs_shape[] = {binding->batches, binding->k, binding->n};
  if (!valid_view(lhs->aligned, lhs->offset, lhs->sizes, lhs->strides, 3, lhs_shape) ||
      !valid_view(rhs->aligned, rhs->offset, rhs->sizes, rhs->strides, 3, rhs_shape))
    return ATLAS_ORDERED_K_INVALID;
  source_view lhs_view = {lhs->aligned, lhs->offset,
                          {lhs->strides[0], lhs->strides[1], lhs->strides[2]}};
  source_view rhs_view = {rhs->aligned, rhs->offset,
                          {rhs->strides[0], rhs->strides[1], rhs->strides[2]}};
  return run(binding, lhs_view, rhs_view, binding->batches, result, 3, device_result);
}
