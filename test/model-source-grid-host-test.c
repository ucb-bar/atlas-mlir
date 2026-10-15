/* Execute an outlined model entry against a deterministic Atlas MMIO stand-in.
 * The stand-in checks staged inputs and instruction words. It can replay
 * separately checked AtlasCore result bytes to test model readback. */
#include "atlas_model_bindings.h"
#include "atlas_model_entry.h"
#include "atlas_source_mn_data.h"

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
  M = ATLAS_SOURCE_MN_ROWS,
  N = ATLAS_SOURCE_MN_COLUMNS,
  K = ATLAS_SOURCE_MN_REDUCTION,
  B = ATLAS_SOURCE_MN_BATCHES,
  N_TILES = (N + 31) / 32,
  TILE_COUNT = ((M + 31) / 32) * N_TILES,
  ALL_TILES = B * TILE_COUNT,
};

_Static_assert(B > 0, "source grid needs a batch");

typedef struct {
  uint32_t imem[sizeof atlas_source_mn_program_words /
                sizeof atlas_source_mn_program_words[0]];
  uint8_t input[ATLAS_SOURCE_MN_WINDOW_COUNT][1024];
  uint8_t output[2048];
  const float *lhs;
  const float *rhs;
  unsigned launches;
  unsigned writes;
  unsigned polls;
  int running;
} fake_device;

static uint32_t f32_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof bits);
  return bits;
}

static uint8_t test_fp8_code(float value) {
  switch (f32_bits(value)) {
    case 0x00000000u: return 0x00u;
    case 0x3f800000u: return 0x38u;
    case 0xbf800000u: return 0xb8u;
    case 0x40000000u: return 0x40u;
    case 0xc0000000u: return 0xc0u;
    default: assert(0 && "unexpected source value in public host test"); return 0;
  }
}

static int input_index(uintptr_t address) {
  for (size_t i = 0; i < ATLAS_SOURCE_MN_WINDOW_COUNT; ++i)
    if (address == atlas_source_mn_windows[i].address) return (int)i;
  return -1;
}

static void check_inputs(const fake_device *device) {
  unsigned tile = device->launches % TILE_COUNT;
  unsigned batch = device->launches / TILE_COUNT % B;
  unsigned m_start = (tile / N_TILES) * 32u;
  unsigned n_start = (tile % N_TILES) * 32u;
  unsigned first_k = K % 32u ? K % 32u : 32u;
  for (size_t input = 0; input < ATLAS_SOURCE_MN_WINDOW_COUNT; ++input) {
    unsigned segment = input / 2u;
    unsigned k_start = segment ? first_k + (segment - 1u) * 32u : 0u;
    unsigned left_pad = segment ? 0u : 32u - first_k;
    for (unsigned row = 0; row < 32u; ++row)
      for (unsigned column = 0; column < 32u; ++column) {
        uint8_t expected = 0;
        unsigned source_row = (input & 1u ? n_start : m_start) + row;
        if (source_row < (input & 1u ? N : M) && column >= left_pad) {
          unsigned source_k = k_start + column - left_pad;
          float source = input & 1u
              ? device->rhs[batch * K * N + source_k * N + source_row]
              : device->lhs[batch * M * K + source_row * K + source_k];
          expected = test_fp8_code(source);
        }
        assert(device->input[input][row * 32u + column] == expected);
      }
  }
}

static void save_input_windows(const fake_device *device) {
  const char *directory = getenv("ATLAS_SOURCE_MN_CAPTURE_DIR");
  if (!directory || !*directory) return;
  unsigned trial = device->launches / ALL_TILES;
  unsigned tile = device->launches % ALL_TILES;
  for (size_t input = 0; input < ATLAS_SOURCE_MN_WINDOW_COUNT; ++input) {
    char path[4096];
    int length = snprintf(path, sizeof path, "%s/trial-%u-tile-%u-input-%zu.bin",
                          directory, trial, tile, input);
    assert(length > 0 && (size_t)length < sizeof path);
    FILE *file = fopen(path, "wb");
    assert(file);
    assert(fwrite(device->input[input], 1, 1024u, file) == 1024u);
    assert(fclose(file) == 0);
  }
}

static int load_rtl_output(fake_device *device) {
  const char *directory = getenv("ATLAS_SOURCE_MN_RTL_OUTPUT_DIR");
  if (!directory || !*directory) return 0;
  unsigned trial = device->launches / ALL_TILES;
  unsigned matrix_tile = device->launches % TILE_COUNT;
  unsigned n_start = matrix_tile % N_TILES * 32u;
  char path[4096];
#if ATLAS_SOURCE_MN_BATCHES == 1
  int length = snprintf(path, sizeof path, "%s/real_host_trial%u_n%u.bin",
                        directory, trial, n_start);
#else
  unsigned batch = device->launches / TILE_COUNT % B;
  unsigned m_start = matrix_tile / N_TILES * 32u;
  int length = snprintf(path, sizeof path,
                        "%s/real_host_trial%u_b%u_m%u_n%u.bin",
                        directory, trial, batch, m_start, n_start);
#endif
  assert(length > 0 && (size_t)length < sizeof path);
  FILE *file = fopen(path, "rb");
  assert(file);
  assert(fread(device->output, 1, sizeof device->output, file) ==
         sizeof device->output);
  assert(fgetc(file) == EOF);
  assert(fclose(file) == 0);
  return 1;
}

static uint32_t read32(void *context, uintptr_t address) {
  fake_device *device = context;
  assert(address == atlas_source_mn_host_map.csr_base + 8u);
  if (!device->running || device->polls++ == 0) return 1;
  device->running = 0;
  return 5;
}

static void write32(void *context, uintptr_t address, uint32_t value) {
  fake_device *device = context;
  if (address == atlas_source_mn_host_map.csr_base + 0x18u) {
    assert(value == 1);
    for (size_t i = 0; i < sizeof device->imem / sizeof device->imem[0]; ++i)
      assert(device->imem[i] == atlas_source_mn_program_words[i]);
    check_inputs(device);
    save_input_windows(device);
    if (!load_rtl_output(device)) {
      unsigned tile = device->launches % ALL_TILES;
      for (unsigned row = 0; row < 32u; ++row)
        for (unsigned column = 0; column < 32u; ++column) {
          uint16_t bits = (uint16_t)(1000u * (tile + 1u) + row * 32u + column);
          size_t offset = (column >= 16u ? 1024u : 0u) + row * 32u +
                          (column % 16u) * 2u;
          device->output[offset] = (uint8_t)bits;
          device->output[offset + 1u] = (uint8_t)(bits >> 8);
        }
    }
    device->launches++;
    device->polls = 0;
    device->running = 1;
    return;
  }
  assert(address >= atlas_source_mn_host_map.imem_base);
  size_t index = (address - atlas_source_mn_host_map.imem_base) / 4u;
  assert(index < sizeof device->imem / sizeof device->imem[0]);
  device->imem[index] = value;
}

static int write_bytes(void *context, uintptr_t address,
                       const uint8_t *source, size_t bytes) {
  fake_device *device = context;
  int index = input_index(address);
  assert(index >= 0 && bytes == 1024u);
  memcpy(device->input[index], source, bytes);
  device->writes++;
  return 0;
}

static int read_bytes(void *context, uintptr_t address,
                      uint8_t *destination, size_t bytes) {
  fake_device *device = context;
  assert(address == ATLAS_SOURCE_MN_OUTPUT_ADDRESS && bytes == 2048u);
  memcpy(destination, device->output, bytes);
  return 0;
}

static void check_result(const float *result, const float *lhs, const float *rhs,
                         int64_t offset, const int64_t *strides) {
  const char *rtl_output = getenv("ATLAS_SOURCE_MN_RTL_OUTPUT_DIR");
  for (size_t batch = 0; batch < B; ++batch)
    for (size_t row = 0; row < M; ++row)
      for (size_t column = 0; column < N; ++column) {
        uint32_t expected;
        if (rtl_output && *rtl_output) {
          float sum = 0.0f;
          for (size_t k = 0; k < K; ++k)
            sum += lhs[batch * M * K + row * K + k] *
                   rhs[batch * K * N + k * N + column];
          expected = f32_bits(sum);
        } else {
          unsigned tile = column / 32u;
          tile += (unsigned)row / 32u * N_TILES + (unsigned)batch * TILE_COUNT;
          uint16_t bf16 = (uint16_t)(1000u * (tile + 1u) +
                                     (row % 32u) * 32u + column % 32u);
          expected = (uint32_t)bf16 << 16;
        }
        float value = result[offset + batch * strides[0] + row * strides[1] +
                             column * strides[2]];
        assert(f32_bits(value) == expected);
      }
}

int main(void) {
  fake_device device = {0};
  atlas_host host;
  static float lhs[B * M * K], rhs[B * K * N], output[B * M * N];
  device.lhs = lhs;
  device.rhs = rhs;
  assert(atlas_bind_model_tiles(
      &host, (atlas_host_mmio){read32, write32, &device},
      (atlas_host_memory){.coherent = 1},
      (atlas_tile_memory_io){write_bytes, read_bytes, &device}, 4) == 0);
  for (unsigned trial = 0; trial < 2u; ++trial) {
    static const float values[] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
    for (size_t i = 0; i < B * M * K; ++i)
      lhs[i] = values[(i + trial) % 5u];
#if ATLAS_SOURCE_MN_FROZEN_RHS
    memcpy(rhs, atlas_source_mn_frozen_rhs_bits, sizeof rhs);
#else
    for (size_t batch = 0; batch < B; ++batch)
      for (size_t i = 0; i < K * N; ++i)
        rhs[batch * K * N + i] = (float)(batch + 1u);
#endif
#if ATLAS_SOURCE_MN_BATCHES == 1
    atlas_f32_memref2 inputs[2] = {
        {lhs, lhs, 0, {M, K}, {K, 1}},
        {rhs, rhs, 0, {K, N}, {N, 1}},
    };
    atlas_f32_memref2 result = {output, output, 0, {M, N}, {N, 1}};
    assert(atlas_invoke_checked_entry(inputs, 2, &result, 1) == 0);
    int64_t strides[] = {0, result.strides[0], result.strides[1]};
    check_result(result.aligned, lhs, rhs, result.offset, strides);
#else
    atlas_f32_memref3 left = {lhs, lhs, 0, {B, M, K}, {M * K, K, 1}};
    atlas_f32_memref3 right = {rhs, rhs, 0, {B, K, N}, {K * N, N, 1}};
    atlas_f32_memref3 result = {output, output, 0, {B, M, N}, {M * N, N, 1}};
    atlas_model_entry_call call = {&left, &right, &result};
    assert(atlas_invoke_checked_entry(&call) == 0);
    check_result(result.aligned, lhs, rhs, result.offset, result.strides);
#endif
    assert(device.launches == (trial + 1u) * ALL_TILES);
    assert(device.writes == device.launches * ATLAS_SOURCE_MN_WINDOW_COUNT);
    if (result.allocated != output) free(result.allocated);
  }
  return 0;
}
