#include "atlas_source_mn_memref.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static unsigned calls;
static int fail_call;
static atlas_ordered_k_tile tile;

static int encode(uint32_t bits, const void *policy, uint8_t *code) {
  (void)bits;
  (void)policy;
  *code = 0;
  return 0;
}

static void check_dense(const float *lhs, const float *rhs, size_t batches,
                        size_t m, size_t n, size_t k) {
  assert(m == 2 && n == 3 && k == 2);
  for (size_t batch = 0; batch < batches; ++batch) {
    assert(lhs[batch * 4 + 0] == (float)(1 + batch * 10));
    assert(lhs[batch * 4 + 1] == (float)(2 + batch * 10));
    assert(lhs[batch * 4 + 2] == (float)(3 + batch * 10));
    assert(lhs[batch * 4 + 3] == (float)(4 + batch * 10));
    assert(rhs[batch * 6 + 0] == (float)(5 + batch * 10));
    assert(rhs[batch * 6 + 1] == (float)(6 + batch * 10));
    assert(rhs[batch * 6 + 2] == (float)(7 + batch * 10));
    assert(rhs[batch * 6 + 3] == (float)(8 + batch * 10));
    assert(rhs[batch * 6 + 4] == (float)(9 + batch * 10));
    assert(rhs[batch * 6 + 5] == (float)(10 + batch * 10));
  }
}

atlas_ordered_k_status atlas_source_mn_run_f32(
    atlas_ordered_k_tile *selected, const float *lhs, size_t lhs_bytes,
    const float *rhs, size_t rhs_bytes, size_t m, size_t n, size_t k,
    atlas_source_mn_encode lhs_encoder, const void *lhs_policy,
    atlas_source_mn_encode rhs_encoder, const void *rhs_policy,
    float *result, size_t result_bytes,
    uint8_t *lhs_codes, size_t lhs_code_bytes,
    uint8_t *rhs_codes, size_t rhs_code_bytes,
    uint8_t *logical, size_t logical_bytes,
    uint8_t *staging, size_t staging_bytes,
    uint8_t *physical, size_t physical_bytes,
    uint8_t *lhs_tile, size_t lhs_tile_bytes,
    uint8_t *rhs_tile, size_t rhs_tile_bytes,
    atlas_host_result *device_result) {
  assert(selected == &tile && lhs_bytes == 16 && rhs_bytes == 24 &&
         result_bytes == 24 && lhs_encoder == encode && rhs_encoder == encode &&
         lhs_policy == NULL && rhs_policy == NULL && lhs_codes && rhs_codes &&
         lhs_code_bytes == 4 && rhs_code_bytes == 6 && logical && logical_bytes == 12 &&
         staging && staging_bytes == 12 && physical && physical_bytes == 2048 &&
         lhs_tile && lhs_tile_bytes == 64 && rhs_tile && rhs_tile_bytes == 64);
  check_dense(lhs, rhs, 1, m, n, k);
  ++calls;
  if (fail_call) return ATLAS_ORDERED_K_INPUT_DOMAIN;
  for (size_t index = 0; index < 6; ++index) result[index] = (float)(100 + index);
  if (device_result) *device_result = ATLAS_HOST_OK;
  return ATLAS_ORDERED_K_OK;
}

atlas_ordered_k_status atlas_source_batched_mn_run_f32(
    atlas_ordered_k_tile *selected, const float *lhs, size_t lhs_bytes,
    const float *rhs, size_t rhs_bytes, size_t batches, size_t m, size_t n, size_t k,
    atlas_source_mn_encode lhs_encoder, const void *lhs_policy,
    atlas_source_mn_encode rhs_encoder, const void *rhs_policy,
    float *result, size_t result_bytes, float *staged, size_t staged_bytes,
    uint8_t *lhs_codes, size_t lhs_code_bytes,
    uint8_t *rhs_codes, size_t rhs_code_bytes,
    uint8_t *logical, size_t logical_bytes,
    uint8_t *staging, size_t staging_bytes,
    uint8_t *physical, size_t physical_bytes,
    uint8_t *lhs_tile, size_t lhs_tile_bytes,
    uint8_t *rhs_tile, size_t rhs_tile_bytes,
    atlas_host_result *device_result) {
  assert(selected == &tile && lhs_bytes == 32 && rhs_bytes == 48 &&
         result_bytes == 48 && staged && staged_bytes == result_bytes &&
         lhs_encoder == encode && rhs_encoder == encode &&
         lhs_policy == NULL && rhs_policy == NULL && lhs_codes && rhs_codes &&
         lhs_code_bytes == 4 && rhs_code_bytes == 6 && logical && logical_bytes == 12 &&
         staging && staging_bytes == 12 && physical && physical_bytes == 2048 &&
         lhs_tile && lhs_tile_bytes == 64 && rhs_tile && rhs_tile_bytes == 64);
  check_dense(lhs, rhs, batches, m, n, k);
  ++calls;
  if (fail_call) return ATLAS_ORDERED_K_DEVICE_ERROR;
  for (size_t index = 0; index < 12; ++index) staged[index] = (float)(200 + index);
  memcpy(result, staged, result_bytes);
  if (device_result) *device_result = ATLAS_HOST_OK;
  return ATLAS_ORDERED_K_OK;
}

static void test_rank2(void) {
  float lhs_data[18] = {0}, rhs_data[20] = {0};
  lhs_data[1] = 1; lhs_data[3] = 2; lhs_data[6] = 3; lhs_data[8] = 4;
  rhs_data[2] = 5; rhs_data[4] = 6; rhs_data[6] = 7;
  rhs_data[9] = 8; rhs_data[11] = 9; rhs_data[13] = 10;
  atlas_f32_memref2 lhs = {lhs_data, lhs_data, 1, {2, 2}, {5, 2}};
  atlas_f32_memref2 rhs = {rhs_data, rhs_data, 2, {2, 3}, {7, 2}};
  atlas_f32_memref2 result = {0};
  atlas_source_mn_memref_binding binding = {
      .tile = &tile, .encode_lhs = encode, .encode_rhs = encode,
      .batches = 1, .m = 2, .n = 3, .k = 2};
  atlas_host_result device = ATLAS_HOST_INVALID;
  assert(atlas_source_mn_run_memref2(&binding, &lhs, &rhs, &result, &device) ==
         ATLAS_ORDERED_K_OK);
  assert(calls == 1 && device == ATLAS_HOST_OK);
  assert(result.allocated == result.aligned && result.offset == 0 &&
         result.sizes[0] == 2 && result.sizes[1] == 3 &&
         result.strides[0] == 3 && result.strides[1] == 1 &&
         result.aligned[0] == 100 && result.aligned[5] == 105);
  free(result.allocated);

  result = (atlas_f32_memref2){0};
  atlas_f32_memref2 before = result;
  fail_call = 1;
  assert(atlas_source_mn_run_memref2(&binding, &lhs, &rhs, &result, &device) ==
         ATLAS_ORDERED_K_INPUT_DOMAIN);
  assert(calls == 2 && !memcmp(&result, &before, sizeof result));
  fail_call = 0;
  lhs.strides[0] = INT64_MAX;
  assert(atlas_source_mn_run_memref2(&binding, &lhs, &rhs, &result, &device) ==
         ATLAS_ORDERED_K_INVALID);
  assert(calls == 2 && !memcmp(&result, &before, sizeof result));
}

static void test_rank3(void) {
  float lhs_data[50] = {0}, rhs_data[60] = {0};
  for (size_t batch = 0; batch < 2; ++batch) {
    size_t lhs_base = 1 + batch * 20;
    lhs_data[lhs_base] = (float)(1 + batch * 10);
    lhs_data[lhs_base + 2] = (float)(2 + batch * 10);
    lhs_data[lhs_base + 5] = (float)(3 + batch * 10);
    lhs_data[lhs_base + 7] = (float)(4 + batch * 10);
    size_t rhs_base = 2 + batch * 24;
    for (size_t row = 0; row < 2; ++row)
      for (size_t column = 0; column < 3; ++column)
        rhs_data[rhs_base + row * 8 + column * 2] =
            (float)(5 + batch * 10 + row * 3 + column);
  }
  atlas_f32_memref3 lhs = {lhs_data, lhs_data, 1, {2, 2, 2}, {20, 5, 2}};
  atlas_f32_memref3 rhs = {rhs_data, rhs_data, 2, {2, 2, 3}, {24, 8, 2}};
  atlas_f32_memref3 result = {0};
  atlas_source_mn_memref_binding binding = {
      .tile = &tile, .encode_lhs = encode, .encode_rhs = encode,
      .batches = 2, .m = 2, .n = 3, .k = 2};
  atlas_host_result device = ATLAS_HOST_INVALID;
  assert(atlas_source_mn_run_memref3(&binding, &lhs, &rhs, &result, &device) ==
         ATLAS_ORDERED_K_OK);
  assert(calls == 3 && device == ATLAS_HOST_OK &&
         result.allocated == result.aligned && result.offset == 0 &&
         result.sizes[0] == 2 && result.sizes[1] == 2 && result.sizes[2] == 3 &&
         result.strides[0] == 6 && result.strides[1] == 3 && result.strides[2] == 1 &&
         result.aligned[0] == 200 && result.aligned[11] == 211);
  free(result.allocated);

  uint32_t frozen[12];
  for (size_t index = 0; index < 12; ++index) {
    float value = (float)(5 + (index / 6) * 10 + index % 6);
    memcpy(&frozen[index], &value, sizeof value);
  }
  binding.frozen_rhs_bits = frozen;
  binding.frozen_rhs_elements = 12;
  result = (atlas_f32_memref3){0};
  assert(atlas_source_mn_run_memref3(&binding, &lhs, &rhs, &result, &device) ==
         ATLAS_ORDERED_K_OK);
  assert(calls == 4 && result.allocated);
  free(result.allocated);
  frozen[11] ^= 1u;
  result = (atlas_f32_memref3){0};
  atlas_f32_memref3 unchanged = result;
  assert(atlas_source_mn_run_memref3(&binding, &lhs, &rhs, &result, &device) ==
         ATLAS_ORDERED_K_CONSTANT_MISMATCH);
  assert(calls == 4 && !memcmp(&result, &unchanged, sizeof result));
  frozen[11] ^= 1u;

  result = (atlas_f32_memref3){0};
  atlas_f32_memref3 before = result;
  fail_call = 1;
  assert(atlas_source_mn_run_memref3(&binding, &lhs, &rhs, &result, &device) ==
         ATLAS_ORDERED_K_DEVICE_ERROR);
  assert(calls == 5 && !memcmp(&result, &before, sizeof result));
  fail_call = 0;
  rhs.sizes[0] = 1;
  assert(atlas_source_mn_run_memref3(&binding, &lhs, &rhs, &result, &device) ==
         ATLAS_ORDERED_K_INVALID);
  assert(calls == 5 && !memcmp(&result, &before, sizeof result));
}

int main(void) {
  test_rank2();
  test_rank3();
  return 0;
}
