#include "atlas_ordered_mn.h"
#include "atlas_source_mn.h"
#ifdef ATLAS_TEST_MN_MEMREF
#include "atlas_source_mn_memref.h"
#include <stdlib.h>
#endif
#ifdef ATLAS_USE_GENERATED_SOURCE_MN_DATA
#include "atlas_source_mn_data.h"
#define selected_pack merlin_pack_row_major_k_window
#define selected_lhs_encoder atlas_source_mn_encode_lhs
#define selected_rhs_encoder atlas_source_mn_encode_rhs
#elif defined(ATLAS_USE_GENERATED_MN_DATA)
#include "atlas_ordered_mn_data.h"
#define selected_pack merlin_pack_row_major_k_window
#define selected_lhs_encoder encode_test_f32
#define selected_rhs_encoder encode_test_f32
#else
#define selected_pack pack_window
#define selected_lhs_encoder encode_test_f32
#define selected_rhs_encoder encode_test_f32
#endif

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <string.h>

#ifdef ATLAS_TEST_MN_SOURCE_GENERATED
#define TEST_K ATLAS_SOURCE_MN_REDUCTION
enum { IMEM = 0x20000, CSR = 0x40000,
       M = ATLAS_SOURCE_MN_ROWS, N = ATLAS_SOURCE_MN_COLUMNS, K = TEST_K };
#elif defined(ATLAS_TEST_MN_FROM_HEADER)
#ifndef ATLAS_USE_GENERATED_MN_DATA
#error "Header-derived matrix test needs generated ordered M/N data"
#endif
#define TEST_K ATLAS_ORDERED_MN_REDUCTION
enum { IMEM = 0x20000, CSR = 0x40000,
       M = ATLAS_ORDERED_MN_ROWS, N = ATLAS_ORDERED_MN_COLUMNS, K = TEST_K };
#elif defined(ATLAS_TEST_MN_TAILS)
#define TEST_K 64
enum { IMEM = 0x20000, CSR = 0x40000, M = 31, N = 33, K = TEST_K };
#elif defined(ATLAS_TEST_MN_CROSSED)
#define TEST_K 31
enum { IMEM = 0x20000, CSR = 0x40000, M = 33, N = 65, K = TEST_K };
#elif defined(ATLAS_TEST_MN_THIN_K65)
#define TEST_K 65
enum { IMEM = 0x20000, CSR = 0x40000, M = 1, N = 7, K = TEST_K };
#elif defined(ATLAS_TEST_MN_THIN_K1)
#define TEST_K 1
enum { IMEM = 0x20000, CSR = 0x40000, M = 7, N = 33, K = TEST_K };
#elif defined(ATLAS_TEST_MN_KTAIL31)
#define TEST_K 31
enum { IMEM = 0x20000, CSR = 0x40000, M = 31, N = 33, K = TEST_K };
#elif defined(ATLAS_TEST_MN_KTAIL1)
#define TEST_K 1
enum { IMEM = 0x20000, CSR = 0x40000, M = 31, N = 33, K = TEST_K };
#elif defined(ATLAS_TEST_MN_KTAIL65)
#define TEST_K 65
enum { IMEM = 0x20000, CSR = 0x40000, M = 31, N = 33, K = TEST_K };
#elif defined(ATLAS_TEST_MN_K1)
#define TEST_K 32
enum { IMEM = 0x20000, CSR = 0x40000, M = 31, N = 33, K = TEST_K };
#elif defined(ATLAS_TEST_MN_SINGLE)
#define TEST_K 64
enum { IMEM = 0x20000, CSR = 0x40000, M = 1, N = 1, K = TEST_K };
#else
#define TEST_K 64
enum { IMEM = 0x20000, CSR = 0x40000, M = 64, N = 96, K = TEST_K };
#endif
enum { N_TILES = (N + 31) / 32, TILE_COUNT = ((M + 31) / 32) * N_TILES,
       INPUT_COUNT = 2 * ((K + 31) / 32) };
#ifdef ATLAS_TEST_MN_FROM_HEADER
_Static_assert(INPUT_COUNT <= 8, "fake device needs at most eight K-window inputs");
#endif
static uintptr_t inputs[8] = {
    0x90002000u, 0x90003000u, 0x90004000u, 0x90005000u,
    0x90006000u, 0x90007000u, 0x90008000u, 0x90009000u};
static uintptr_t output_address = 0x9000a000u;
static const uint8_t codes[] = {0x00, 0x30, 0xb0, 0x38, 0xb8, 0x40, 0xc0};

typedef struct {
  uint32_t imem[128];
  uint8_t input[8][1024];
  uint8_t output[2048];
  const uint8_t *activation;
  const uint8_t *weight;
  unsigned launches;
  unsigned writes;
  int running;
  int timeout_at;
  unsigned status_reads;
  int mark_batch_output;
  const uint32_t *expected_words;
  size_t expected_word_count;
  size_t expected_input_count;
#ifdef ATLAS_TEST_MN_MEMREF
  const float *source_lhs_f32;
  const float *source_rhs_f32;
  unsigned source_batch_base;
  unsigned source_batch_count;
#endif
} fake_device;

static int input_index(const fake_device *device, uintptr_t address) {
  for (size_t index = 0; index < device->expected_input_count; ++index)
    if (address == inputs[index])
      return (int)index;
  return -1;
}

static void check_staged_inputs(const fake_device *device) {
  unsigned tile = device->launches % TILE_COUNT;
  size_t m = (tile / N_TILES) * 32u;
  size_t n = (tile % N_TILES) * 32u;
  for (size_t input = 0; input < device->expected_input_count; ++input) {
    const uint8_t *source = input % 2 ? device->weight : device->activation;
    size_t row_start = input % 2 ? n : m;
    size_t k_index = input / 2u;
    size_t head = K % 32u;
    size_t head_zeros = head && k_index == 0 ? 32u - head : 0;
    size_t k_start = k_index == 0 ? 0 :
        head ? head + (k_index - 1u) * 32u : k_index * 32u;
    for (size_t row = 0; row < 32; ++row)
      for (size_t k = 0; k < 32; ++k) {
#ifdef ATLAS_TEST_MN_MEMREF
        if (device->source_lhs_f32) {
          uint8_t expected = 0;
          if (row_start + row < (input % 2 ? N : M) && k >= head_zeros) {
            unsigned batch = (device->launches - device->source_batch_base) / TILE_COUNT;
            assert(batch < device->source_batch_count);
            size_t logical_k = k_start + k - head_zeros;
            float value = input % 2 ?
                device->source_rhs_f32[batch * K * N + logical_k * N + row_start + row] :
                device->source_lhs_f32[batch * M * K + (row_start + row) * K + logical_k];
            uint32_t bits;
            memcpy(&bits, &value, sizeof bits);
            assert((input % 2 ? selected_rhs_encoder : selected_lhs_encoder)(
                       bits, 0, &expected) == 0);
          }
          assert(device->input[input][row * 32 + k] == expected);
          continue;
        }
#endif
        assert(device->input[input][row * 32 + k] ==
               (row_start + row < (input % 2 ? N : M) && k >= head_zeros
                    ? source[(row_start + row) * K + k_start + k - head_zeros] : 0));
      }
  }
}

static uint32_t read32(void *context, uintptr_t address) {
  fake_device *device = context;
  if (address == CSR + 4)
    return 0; /* Retired-instruction count, read at each start. */
  assert(address == CSR + 8);
  if (!device->running)
    return 1;
  if (device->status_reads++ == 0)
    return 1; /* The start write clears the previous halt reason. */
  if (device->timeout_at == (int)device->launches)
    return 0;
  device->running = 0;
  return 5;
}

static void write32(void *context, uintptr_t address, uint32_t word) {
  fake_device *device = context;
  if (address == CSR + 0x18) {
    assert(word == 1);
    for (size_t index = 0; index < device->expected_word_count; ++index)
      assert(device->imem[index] == device->expected_words[index]);
    check_staged_inputs(device);
    unsigned tile = device->launches % TILE_COUNT;
    for (unsigned row = 0; row < 32; ++row)
      for (unsigned col = 0; col < 32; ++col) {
        uint16_t bits = (uint16_t)(1000u * (tile + 1u) + row * 32u + col +
            (device->mark_batch_output ?
                 5000u * ((device->launches / TILE_COUNT) % 2u) : 0u));
        size_t half = col >= 16u ? 1024u : 0u;
        size_t offset = half + row * 32u + (col % 16u) * 2u;
        device->output[offset] = (uint8_t)bits;
        device->output[offset + 1] = (uint8_t)(bits >> 8);
      }
    ++device->launches;
    device->running = 1;
    device->status_reads = 0;
    return;
  }
  assert(address >= IMEM && address < IMEM + sizeof device->imem);
  device->imem[(address - IMEM) / 4] = word;
}

static int write_bytes(void *context, uintptr_t address,
                       const uint8_t *source, size_t bytes) {
  fake_device *device = context;
  int index = input_index(device, address);
  assert(index >= 0 && bytes == 1024);
  memcpy(device->input[index], source, bytes);
  ++device->writes;
  return 0;
}

static int read_bytes(void *context, uintptr_t address,
                      uint8_t *destination, size_t bytes) {
  fake_device *device = context;
  assert(address == output_address && bytes == sizeof device->output);
  memcpy(destination, device->output, bytes);
  return 0;
}

#if !defined(ATLAS_USE_GENERATED_MN_DATA) && !defined(ATLAS_USE_GENERATED_SOURCE_MN_DATA)
static int pack_window(const uint8_t *source, size_t source_bytes,
                       uint8_t *destination, size_t destination_bytes,
                       size_t rows, size_t source_columns, size_t k_start,
                       size_t tile_columns, size_t element_bytes) {
  assert(rows == 32 && source_columns == K && tile_columns == 32 &&
         element_bytes == 1 && source_bytes == 32 * K &&
         destination_bytes == 1024 && k_start + tile_columns <= K);
  for (size_t row = 0; row < rows; ++row)
    memcpy(destination + row * 32, source + row * K + k_start, 32);
  return 0;
}
#endif

static void check_output(const uint8_t *output) {
  for (unsigned row = 0; row < M; ++row)
    for (unsigned col = 0; col < N; ++col) {
      unsigned tile = (row / 32u) * N_TILES + col / 32u;
      uint16_t expected = (uint16_t)(1000u * (tile + 1u) +
                                     (row % 32u) * 32u + col % 32u);
      size_t offset = (row * N + col) * 2u;
      assert(output[offset] == (uint8_t)expected);
      assert(output[offset + 1] == (uint8_t)(expected >> 8));
    }
}

static int encode_test_f32(uint32_t bits, const void *policy, uint8_t *code) {
  (void)policy;
  switch (bits) {
    case 0x00000000u: *code = 0x00; return 0;
    case 0x3f800000u: *code = 0x38; return 0;
    case 0xbf800000u: *code = 0xb8; return 0;
    case 0x40000000u: *code = 0x40; return 0;
    case 0xc0000000u: *code = 0xc0; return 0;
    default: return 2;
  }
}

#if defined(ATLAS_TEST_MN_SOURCE_GENERATED) || defined(ATLAS_TEST_MN_TAILS) || \
    defined(ATLAS_TEST_MN_FROM_HEADER) || \
    defined(ATLAS_TEST_MN_CROSSED) || \
    defined(ATLAS_TEST_MN_THIN_K65) || defined(ATLAS_TEST_MN_THIN_K1) || \
    defined(ATLAS_TEST_MN_SINGLE) || \
    defined(ATLAS_TEST_MN_K1) || defined(ATLAS_TEST_MN_KTAIL1) || \
    defined(ATLAS_TEST_MN_KTAIL31) || \
    defined(ATLAS_TEST_MN_KTAIL65)
static uint8_t activation_tile[32 * K], weight_tile[32 * K];
#endif

static atlas_ordered_k_status run_matrix(
    atlas_ordered_k_tile *tile, const uint8_t *activation, size_t activation_bytes,
    const uint8_t *weight, size_t weight_bytes,
    size_t m_rows, size_t n_rows, size_t k_columns,
    uint8_t *result, size_t result_bytes,
    uint8_t *staging, size_t staging_bytes,
    uint8_t *physical, size_t physical_bytes,
    atlas_host_result *device_result) {
#if defined(ATLAS_TEST_MN_SOURCE_GENERATED) || defined(ATLAS_TEST_MN_TAILS) || \
    defined(ATLAS_TEST_MN_FROM_HEADER) || \
    defined(ATLAS_TEST_MN_CROSSED) || \
    defined(ATLAS_TEST_MN_THIN_K65) || defined(ATLAS_TEST_MN_THIN_K1) || \
    defined(ATLAS_TEST_MN_SINGLE) || \
    defined(ATLAS_TEST_MN_K1) || defined(ATLAS_TEST_MN_KTAIL1) || \
    defined(ATLAS_TEST_MN_KTAIL31) || \
    defined(ATLAS_TEST_MN_KTAIL65)
  return atlas_ordered_mn_run_tails(
      tile, activation, activation_bytes, weight, weight_bytes,
      m_rows, n_rows, k_columns, result, result_bytes,
      staging, staging_bytes, physical, physical_bytes,
      activation_tile, sizeof activation_tile,
      weight_tile, sizeof weight_tile, device_result);
#else
  return atlas_ordered_mn_run(
      tile, activation, activation_bytes, weight, weight_bytes,
      m_rows, n_rows, k_columns, result, result_bytes,
      staging, staging_bytes, physical, physical_bytes, device_result);
#endif
}

int main(void) {
  fake_device device = {.timeout_at = -1};
  atlas_host host;
#ifdef ATLAS_USE_GENERATED_SOURCE_MN_DATA
  _Static_assert(ATLAS_SOURCE_MN_ROWS == M && ATLAS_SOURCE_MN_COLUMNS == N &&
                 ATLAS_SOURCE_MN_REDUCTION == K &&
                 ATLAS_SOURCE_MN_TILE_COUNT == TILE_COUNT &&
                 ATLAS_SOURCE_MN_WINDOW_COUNT == INPUT_COUNT,
                 "generated source matrix geometry differs from fixture");
  const uint32_t *words = atlas_source_mn_program_words;
  const size_t word_count = sizeof atlas_source_mn_program_words /
                            sizeof atlas_source_mn_program_words[0];
  const atlas_ordered_k_window *windows = atlas_source_mn_windows;
  const atlas_host_map map = atlas_source_mn_host_map;
  output_address = ATLAS_SOURCE_MN_OUTPUT_ADDRESS;
  for (size_t index = 0; index < INPUT_COUNT; ++index)
    inputs[index] = windows[index].address;
#elif defined(ATLAS_USE_GENERATED_MN_DATA)
  _Static_assert(ATLAS_ORDERED_MN_ROWS == M && ATLAS_ORDERED_MN_COLUMNS == N &&
                 ATLAS_ORDERED_MN_REDUCTION == K &&
                 ATLAS_ORDERED_MN_TILE_COUNT == TILE_COUNT,
                 "generated matrix geometry differs from fixture");
  _Static_assert(ATLAS_ORDERED_K_WINDOW_COUNT == INPUT_COUNT,
                 "generated K windows differ from fixture");
  const uint32_t *words = atlas_ordered_k_program_words;
  const size_t word_count = ATLAS_ORDERED_K_PROGRAM_WORD_COUNT;
  const atlas_ordered_k_window *windows = atlas_ordered_k_windows;
  const atlas_host_map map = atlas_ordered_k_host_map;
  output_address = ATLAS_ORDERED_K_OUTPUT_ADDRESS;
  for (size_t index = 0; index < ATLAS_ORDERED_K_WINDOW_COUNT; ++index)
    inputs[index] = windows[index].address;
#else
  const uint32_t sample_words[] = {0x13u, 0x73u};
  const uint32_t *words = sample_words;
  const size_t word_count = 2;
  const atlas_ordered_k_window sample_windows[] = {
      {0, 32, K, 0, K % 32 ? K % 32 : 32, inputs[0]},
      {1, 32, K, 0, K % 32 ? K % 32 : 32, inputs[1]},
#if TEST_K >= 64
      {0, 32, K, K % 32 ? K % 32 : 32, 32, inputs[2]},
      {1, 32, K, K % 32 ? K % 32 : 32, 32, inputs[3]},
#endif
#if TEST_K >= 65
      {0, 32, K, 33, 32, inputs[4]}, {1, 32, K, 33, 32, inputs[5]},
#endif
  };
  const atlas_ordered_k_window *windows = sample_windows;
  const atlas_host_map map = {IMEM, CSR, sizeof device.imem};
#endif
  assert(word_count <= sizeof device.imem / sizeof device.imem[0]);
  device.expected_words = words;
  device.expected_word_count = word_count;
  device.expected_input_count = INPUT_COUNT;
  assert(atlas_host_init(&host, map,
                         (atlas_host_mmio){read32, write32, &device},
                         (atlas_host_memory){.coherent = 1}) == ATLAS_HOST_OK);
  uint8_t scratch[2048], physical[2048], staging[M * N * 2];
  uint8_t result[M * N * 2], activation[M * K], weight[N * K];
  atlas_host_buffer buffers[9];
  atlas_ordered_k_tile tile = {
      .host = &host, .memory_context = &device,
      .write = write_bytes, .read = read_bytes, .pack = selected_pack,
      .program_words = words, .program_word_count = word_count,
      .windows = windows, .window_count = INPUT_COUNT,
      .output_address = output_address, .max_polls = 2,
      .scratch = scratch, .scratch_bytes = sizeof scratch,
      .active_buffers = buffers, .active_buffer_capacity = 9,
  };
  atlas_host_result device_result;
  for (unsigned trial = 0; trial < 2; ++trial) {
    for (size_t row = 0; row < M; ++row)
      for (size_t k = 0; k < K; ++k)
        activation[row * K + k] = codes[(row + k + trial) % sizeof codes];
    for (size_t row = 0; row < N; ++row)
      for (size_t k = 0; k < K; ++k)
        weight[row * K + k] = codes[(2 * row + k + trial) % sizeof codes];
    device.activation = activation;
    device.weight = weight;
    memset(result, 0xa5, sizeof result);
    assert(run_matrix(
               &tile, activation, sizeof activation, weight, sizeof weight,
               M, N, K, result, sizeof result, staging, sizeof staging,
               physical, sizeof physical, &device_result) == ATLAS_ORDERED_K_OK);
    assert(device_result == ATLAS_HOST_OK);
    assert(device.launches == TILE_COUNT * (trial + 1u));
    assert(device.writes == device.expected_input_count * device.launches);
    check_output(result);
  }
  weight[sizeof weight - 1] = 0xff;
  memset(result, 0xa5, sizeof result);
  assert(run_matrix(
             &tile, activation, sizeof activation, weight, sizeof weight,
             M, N, K, result, sizeof result, staging, sizeof staging,
             physical, sizeof physical, &device_result) == ATLAS_ORDERED_K_INPUT_DOMAIN);
  assert(device.launches == 2u * TILE_COUNT &&
         result[0] == 0xa5 && result[sizeof result - 1] == 0xa5);
  weight[sizeof weight - 1] = 0x38;
  assert(run_matrix(
             &tile, activation, sizeof activation, weight, sizeof weight,
             M + 1u, N, K, result, sizeof result, staging, sizeof staging,
             physical, sizeof physical, &device_result) == ATLAS_ORDERED_K_INVALID);
  assert(run_matrix(
             &tile, activation, sizeof activation, weight, sizeof weight,
             M, N, K, result, sizeof result, result, sizeof result,
             physical, sizeof physical, &device_result) == ATLAS_ORDERED_K_INVALID);
  assert(device.launches == 2u * TILE_COUNT);

  float source_lhs[M * K], source_rhs[K * N], source_result[M * N];
  uint8_t lhs_codes[M * K], rhs_codes[N * K];
  uint8_t logical_bf16[M * N * 2];
  uint8_t source_lhs_tile[32 * K], source_rhs_tile[32 * K];
  static const float source_values[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
  for (size_t row = 0; row < M; ++row)
    for (size_t k = 0; k < K; ++k)
      source_lhs[row * K + k] = source_values[(row + 2 * k) % 5];
#if defined(ATLAS_USE_GENERATED_SOURCE_MN_DATA) && ATLAS_SOURCE_MN_FROZEN_RHS
  memcpy(source_rhs, atlas_source_mn_frozen_rhs_bits, sizeof source_rhs);
#else
  for (size_t k = 0; k < K; ++k)
    for (size_t col = 0; col < N; ++col)
      source_rhs[k * N + col] = source_values[(3 * k + col) % 5];
#endif
  device.activation = lhs_codes;
  device.weight = rhs_codes;
  memset(source_result, 0xa5, sizeof source_result);
  assert(atlas_source_mn_run_f32(
             &tile, source_lhs, sizeof source_lhs,
             source_rhs, sizeof source_rhs, M, N, K,
             selected_lhs_encoder, 0, selected_rhs_encoder, 0,
             source_result, sizeof source_result,
             lhs_codes, sizeof lhs_codes, rhs_codes, sizeof rhs_codes,
             logical_bf16, sizeof logical_bf16,
             staging, sizeof staging, physical, sizeof physical,
             source_lhs_tile, sizeof source_lhs_tile,
             source_rhs_tile, sizeof source_rhs_tile,
             &device_result) == ATLAS_ORDERED_K_OK);
  assert(device_result == ATLAS_HOST_OK);
  assert(device.launches == 3u * TILE_COUNT);
  unsigned expected_source_launches = 3u * TILE_COUNT;
  for (size_t row = 0; row < M; ++row)
    for (size_t k = 0; k < K; ++k) {
      uint8_t expected;
      uint32_t bits;
      memcpy(&bits, &source_lhs[row * K + k], sizeof bits);
      assert(encode_test_f32(bits, 0, &expected) == 0);
      assert(lhs_codes[row * K + k] == expected);
    }
  for (size_t k = 0; k < K; ++k)
    for (size_t col = 0; col < N; ++col) {
      uint8_t expected;
      uint32_t bits;
      memcpy(&bits, &source_rhs[k * N + col], sizeof bits);
      assert(encode_test_f32(bits, 0, &expected) == 0);
      assert(rhs_codes[col * K + k] == expected);
    }
  for (size_t row = 0; row < M; ++row)
    for (size_t col = 0; col < N; ++col) {
      unsigned tile_index = (row / 32u) * N_TILES + col / 32u;
      uint32_t expected = (uint32_t)(1000u * (tile_index + 1u) +
                                     (row % 32u) * 32u + col % 32u) << 16;
      uint32_t bits;
      memcpy(&bits, &source_result[row * N + col], sizeof bits);
      assert(bits == expected);
    }
  source_rhs[K * N - 1] = -0.0f;
  memset(source_result, 0xa5, sizeof source_result);
  int zero_status = atlas_source_mn_run_f32(
             &tile, source_lhs, sizeof source_lhs,
             source_rhs, sizeof source_rhs, M, N, K,
             selected_lhs_encoder, 0, selected_rhs_encoder, 0,
             source_result, sizeof source_result,
             lhs_codes, sizeof lhs_codes, rhs_codes, sizeof rhs_codes,
             logical_bf16, sizeof logical_bf16,
             staging, sizeof staging, physical, sizeof physical,
             source_lhs_tile, sizeof source_lhs_tile,
             source_rhs_tile, sizeof source_rhs_tile,
             &device_result);
#ifdef ATLAS_TEST_MN_POSITIVE_ZERO
  assert(zero_status == ATLAS_ORDERED_K_OK);
  assert(device_result == ATLAS_HOST_OK);
  assert(rhs_codes[(N - 1u) * K + K - 1u] == 0);
  expected_source_launches += TILE_COUNT;
  assert(device.launches == expected_source_launches);
#else
  assert(zero_status == ATLAS_ORDERED_K_INPUT_DOMAIN);
  assert(device.launches == expected_source_launches);
  assert(device_result == ATLAS_HOST_INVALID);
  for (size_t i = 0; i < sizeof source_result; ++i)
    assert(((const uint8_t *)source_result)[i] == 0xa5);
#endif
  source_rhs[K * N - 1] = 0.0f;
  assert(atlas_source_mn_run_f32(
             &tile, source_lhs, sizeof source_lhs,
             source_rhs, sizeof source_rhs, M, N, K,
             selected_lhs_encoder, 0, selected_rhs_encoder, 0,
             source_result, sizeof source_result,
             lhs_codes, sizeof lhs_codes, rhs_codes, sizeof rhs_codes,
             logical_bf16, sizeof logical_bf16,
             staging, sizeof staging, physical, sizeof physical,
             source_lhs_tile, sizeof source_lhs_tile,
             source_rhs_tile, sizeof source_rhs_tile - 1,
             &device_result) == ATLAS_ORDERED_K_INVALID);
  assert(device.launches == expected_source_launches);

  float batch_lhs[2 * M * K], batch_rhs[2 * K * N];
  float batch_result[2 * M * N], batch_staging[2 * M * N];
  for (size_t batch = 0; batch < 2; ++batch) {
    for (size_t row = 0; row < M; ++row)
      for (size_t k = 0; k < K; ++k)
        batch_lhs[batch * M * K + row * K + k] =
            source_values[(row + 2 * k + batch) % 5];
#if defined(ATLAS_USE_GENERATED_SOURCE_MN_DATA) && ATLAS_SOURCE_MN_FROZEN_RHS
    memcpy(&batch_rhs[batch * K * N],
           &atlas_source_mn_frozen_rhs_bits[
               (ATLAS_SOURCE_MN_BATCHES > 1 ? batch : 0) * K * N],
           K * N * sizeof(float));
#else
    for (size_t k = 0; k < K; ++k)
      for (size_t col = 0; col < N; ++col)
        batch_rhs[batch * K * N + k * N + col] =
            source_values[(3 * k + col + batch) % 5];
#endif
  }
  device.mark_batch_output = 1;
  unsigned first_batch_marker = (device.launches / TILE_COUNT) % 2u;
  memset(batch_result, 0xa5, sizeof batch_result);
  assert(atlas_source_batched_mn_run_f32(
             &tile, batch_lhs, sizeof batch_lhs, batch_rhs, sizeof batch_rhs,
             2, M, N, K, selected_lhs_encoder, 0, selected_rhs_encoder, 0,
             batch_result, sizeof batch_result,
             batch_staging, sizeof batch_staging,
             lhs_codes, sizeof lhs_codes, rhs_codes, sizeof rhs_codes,
             logical_bf16, sizeof logical_bf16, staging, sizeof staging,
             physical, sizeof physical,
             source_lhs_tile, sizeof source_lhs_tile,
             source_rhs_tile, sizeof source_rhs_tile,
             &device_result) == ATLAS_ORDERED_K_OK);
  assert(device_result == ATLAS_HOST_OK);
  expected_source_launches += 2u * TILE_COUNT;
  assert(device.launches == expected_source_launches);
  for (size_t batch = 0; batch < 2; ++batch)
    for (size_t row = 0; row < M; ++row)
      for (size_t col = 0; col < N; ++col) {
        unsigned tile_index = (row / 32u) * N_TILES + col / 32u;
        uint32_t expected = (uint32_t)(1000u * (tile_index + 1u) +
            (row % 32u) * 32u + col % 32u +
            5000u * ((first_batch_marker + batch) % 2u)) << 16;
        uint32_t actual;
        memcpy(&actual, &batch_result[batch * M * N + row * N + col], sizeof actual);
        assert(actual == expected);
      }

  /* A late invalid operand must preserve the complete result, including
   * the first batch that already ran on the device. */
  uint32_t nan_bits = 0x7fc00000u;
  uint8_t unused_code;
  int encode_failure = selected_rhs_encoder(nan_bits, 0, &unused_code);
  assert(encode_failure == 1 || encode_failure == 2);
  atlas_ordered_k_status expected_failure = encode_failure == 2 ?
      ATLAS_ORDERED_K_INPUT_DOMAIN : ATLAS_ORDERED_K_INVALID;
  memcpy(&batch_rhs[2 * K * N - 1], &nan_bits, sizeof nan_bits);
  memset(batch_result, 0xa5, sizeof batch_result);
  assert(atlas_source_batched_mn_run_f32(
             &tile, batch_lhs, sizeof batch_lhs, batch_rhs, sizeof batch_rhs,
             2, M, N, K, selected_lhs_encoder, 0, selected_rhs_encoder, 0,
             batch_result, sizeof batch_result,
             batch_staging, sizeof batch_staging,
             lhs_codes, sizeof lhs_codes, rhs_codes, sizeof rhs_codes,
             logical_bf16, sizeof logical_bf16, staging, sizeof staging,
             physical, sizeof physical,
             source_lhs_tile, sizeof source_lhs_tile,
             source_rhs_tile, sizeof source_rhs_tile,
             &device_result) == expected_failure);
  expected_source_launches += TILE_COUNT;
  assert(device.launches == expected_source_launches);
  assert(device_result == ATLAS_HOST_INVALID);
  for (size_t i = 0; i < sizeof batch_result; ++i)
    assert(((const uint8_t *)batch_result)[i] == 0xa5);
  memcpy(&batch_rhs[2 * K * N - 1], &source_rhs[K * N - 1], sizeof(float));
  assert(atlas_source_batched_mn_run_f32(
             &tile, batch_lhs, sizeof batch_lhs, batch_rhs, sizeof batch_rhs,
             2, M, N, K, selected_lhs_encoder, 0, selected_rhs_encoder, 0,
             batch_result, sizeof batch_result,
             batch_result, sizeof batch_result,
             lhs_codes, sizeof lhs_codes, rhs_codes, sizeof rhs_codes,
             logical_bf16, sizeof logical_bf16, staging, sizeof staging,
             physical, sizeof physical,
             source_lhs_tile, sizeof source_lhs_tile,
             source_rhs_tile, sizeof source_rhs_tile,
             &device_result) == ATLAS_ORDERED_K_INVALID);
  assert(device.launches == expected_source_launches);
#ifdef ATLAS_TEST_MN_MEMREF
  device.mark_batch_output = 0;
  device.source_lhs_f32 = source_lhs;
  device.source_rhs_f32 = source_rhs;
  device.source_batch_base = device.launches;
  device.source_batch_count = 1;
  atlas_source_mn_memref_binding memref_binding = {
      .tile = &tile, .encode_lhs = selected_lhs_encoder,
      .encode_rhs = selected_rhs_encoder, .batches = 1, .m = M, .n = N, .k = K};
  atlas_f32_memref2 lhs_ref = {
      source_lhs, source_lhs, 0, {M, K}, {K, 1}};
  atlas_f32_memref2 rhs_ref = {
      source_rhs, source_rhs, 0, {K, N}, {N, 1}};
  atlas_f32_memref2 result_ref = {0};
  assert(atlas_source_mn_run_memref2(&memref_binding, &lhs_ref, &rhs_ref,
                                     &result_ref, &device_result) == ATLAS_ORDERED_K_OK);
  expected_source_launches += TILE_COUNT;
  assert(device.launches == expected_source_launches && result_ref.allocated &&
         result_ref.sizes[0] == M && result_ref.sizes[1] == N);
  free(result_ref.allocated);
#ifdef ATLAS_USE_GENERATED_SOURCE_MN_DATA
#if ATLAS_SOURCE_MN_BATCHES > 1
  device.mark_batch_output = 1;
  device.source_lhs_f32 = batch_lhs;
  device.source_rhs_f32 = batch_rhs;
  device.source_batch_base = device.launches;
  device.source_batch_count = 2;
  memref_binding.batches = 2;
  atlas_f32_memref3 batch_lhs_ref = {
      batch_lhs, batch_lhs, 0, {2, M, K}, {M * K, K, 1}};
  atlas_f32_memref3 batch_rhs_ref = {
      batch_rhs, batch_rhs, 0, {2, K, N}, {K * N, N, 1}};
  atlas_f32_memref3 batch_result_ref = {0};
  assert(atlas_source_mn_run_memref3(&memref_binding, &batch_lhs_ref,
                                     &batch_rhs_ref, &batch_result_ref,
                                     &device_result) == ATLAS_ORDERED_K_OK);
  expected_source_launches += 2u * TILE_COUNT;
  assert(device.launches == expected_source_launches && batch_result_ref.allocated &&
         batch_result_ref.sizes[0] == 2 && batch_result_ref.sizes[1] == M &&
         batch_result_ref.sizes[2] == N);
  free(batch_result_ref.allocated);
#endif
#endif
  device.source_lhs_f32 = 0;
  device.source_rhs_f32 = 0;
#endif
  device.mark_batch_output = 0;
  device.activation = activation;
  device.weight = weight;
#if defined(ATLAS_TEST_MN_SOURCE_GENERATED) || defined(ATLAS_TEST_MN_TAILS) || \
    defined(ATLAS_TEST_MN_FROM_HEADER) || \
    defined(ATLAS_TEST_MN_CROSSED) || \
    defined(ATLAS_TEST_MN_THIN_K65) || defined(ATLAS_TEST_MN_THIN_K1) || \
    defined(ATLAS_TEST_MN_SINGLE) || \
    defined(ATLAS_TEST_MN_K1) || defined(ATLAS_TEST_MN_KTAIL1) || \
    defined(ATLAS_TEST_MN_KTAIL31) || \
    defined(ATLAS_TEST_MN_KTAIL65)
  assert(atlas_ordered_mn_run_tails(
             &tile, activation, sizeof activation, weight, sizeof weight,
             M, N, K, result, sizeof result, staging, sizeof staging,
             physical, sizeof physical,
             activation, sizeof activation_tile,
             weight_tile, sizeof weight_tile,
             &device_result) == ATLAS_ORDERED_K_INVALID);
  assert(atlas_ordered_mn_run_tails(
             &tile, activation, sizeof activation, weight, sizeof weight,
             M, N, K, result, sizeof result, staging, sizeof staging,
             physical, sizeof physical,
             activation_tile, sizeof activation_tile - 1,
             weight_tile, sizeof weight_tile,
             &device_result) == ATLAS_ORDERED_K_INVALID);
  assert(device.launches == expected_source_launches);
#endif
  device.timeout_at = (int)(device.launches + 1u);
  device.activation = lhs_codes;
  device.weight = rhs_codes;
  memset(source_result, 0xa5, sizeof source_result);
  assert(atlas_source_mn_run_f32(
             &tile, source_lhs, sizeof source_lhs,
             source_rhs, sizeof source_rhs, M, N, K,
             selected_lhs_encoder, 0, selected_rhs_encoder, 0,
             source_result, sizeof source_result,
             lhs_codes, sizeof lhs_codes, rhs_codes, sizeof rhs_codes,
             logical_bf16, sizeof logical_bf16,
             staging, sizeof staging, physical, sizeof physical,
             source_lhs_tile, sizeof source_lhs_tile,
             source_rhs_tile, sizeof source_rhs_tile,
             &device_result) == ATLAS_ORDERED_K_DEVICE_ERROR);
  assert(device_result == ATLAS_HOST_TIMEOUT && host.active);
  for (size_t i = 0; i < sizeof source_result; ++i)
    assert(((const uint8_t *)source_result)[i] == 0xa5);
  return 0;
}
