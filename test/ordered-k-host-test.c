#include "atlas_ordered_k.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <string.h>

#if defined(ATLAS_USE_MERLIN_K_PACKER)
#include "row_major_k_window.h"
#define selected_pack merlin_pack_row_major_k_window
#else
#define selected_pack pack_window
#endif

#if defined(ATLAS_USE_GENERATED_K_DATA)
#include "atlas_ordered_k_data.h"
_Static_assert(ATLAS_ORDERED_K_WINDOW_COUNT >= 4u &&
               ATLAS_ORDERED_K_WINDOW_COUNT <= 8u,
               "test supports two to four K tiles");
#endif

enum { IMEM = 0x20000, CSR = 0x40000, MAX_WINDOWS = 8,
       MAX_SOURCE_BYTES = 4096 };

static uintptr_t input_addresses[MAX_WINDOWS];
static size_t input_count;
static uintptr_t output_address;

typedef struct {
  uint32_t words[128];
  uint8_t inputs[MAX_WINDOWS][1024];
  uint8_t output[2048];
  unsigned writes;
  unsigned starts;
  unsigned status_reads;
  int running;
  int timeout;
  int read_fails;
} fake_device;

static int input_index(uintptr_t address) {
  for (size_t index = 0; index < input_count; ++index)
    if (address == input_addresses[index])
      return (int)index;
  return -1;
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
  if (device->timeout)
    return 0;
  device->running = 0;
  return 5;
}

static void write32(void *context, uintptr_t address, uint32_t value) {
  fake_device *device = context;
  if (address == CSR + 0x18) {
    assert(value == 1);
    ++device->starts;
    device->running = 1;
    device->status_reads = 0;
    for (size_t index = 0; index < sizeof(device->output); ++index)
      device->output[index] = (uint8_t)(index ^ 0x5a);
    return;
  }
  assert(address >= IMEM && address < IMEM + sizeof(device->words));
  device->words[(address - IMEM) / 4] = value;
}

static int write_bytes(void *context, uintptr_t address,
                       const uint8_t *source, size_t bytes) {
  fake_device *device = context;
  int index = input_index(address);
  assert(index >= 0 && bytes == 1024);
  memcpy(device->inputs[index], source, bytes);
  ++device->writes;
  return 0;
}

static int read_bytes(void *context, uintptr_t address,
                      uint8_t *destination, size_t bytes) {
  fake_device *device = context;
  assert(address == output_address && bytes == sizeof(device->output));
  if (device->read_fails)
    return 1;
  memcpy(destination, device->output, bytes);
  return 0;
}

#if !defined(ATLAS_USE_MERLIN_K_PACKER)
static int pack_window(const uint8_t *source, size_t source_bytes,
                       uint8_t *destination, size_t destination_bytes,
                       size_t rows, size_t source_columns, size_t k_start,
                       size_t tile_columns, size_t element_bytes) {
  assert(rows == 32 && source_columns == 32 * (input_count / 2) &&
         tile_columns == 32 && element_bytes == 1 &&
         source_bytes == 32 * source_columns && destination_bytes == 1024);
  for (size_t row = 0; row < rows; ++row)
    for (size_t column = 0; column < tile_columns; ++column)
      destination[row * tile_columns + column] =
          source[row * source_columns + k_start + column];
  return 0;
}
#endif

int main(void) {
  for (unsigned code = 0; code < 256; ++code) {
    int expected = code == 0 || (code >= 8 && code <= 119) ||
                   (code >= 136 && code <= 247);
    assert(atlas_ordered_k_admit_code((uint8_t)code) == expected);
  }
  fake_device device = {0};
  atlas_host host;
#if defined(ATLAS_USE_GENERATED_K_DATA)
  assert(atlas_ordered_k_host_map.imem_base == IMEM &&
         atlas_ordered_k_host_map.csr_base == CSR);
  const atlas_host_map map = atlas_ordered_k_host_map;
  const uint32_t *words = atlas_ordered_k_program_words;
  const size_t word_count = ATLAS_ORDERED_K_PROGRAM_WORD_COUNT;
  atlas_ordered_k_window windows[ATLAS_ORDERED_K_WINDOW_COUNT];
  memcpy(windows, atlas_ordered_k_windows, sizeof(windows));
  input_count = ATLAS_ORDERED_K_WINDOW_COUNT;
  output_address = ATLAS_ORDERED_K_OUTPUT_ADDRESS;
#else
  const atlas_host_map map = {IMEM, CSR, sizeof(device.words)};
  const uint32_t sample_words[] = {0x13, 0x73, 0x8067};
  const uint32_t *words = sample_words;
  const size_t word_count = 3;
  atlas_ordered_k_window windows[] = {
      {0, 32, 64, 0, 32, 0x90002000},
      {1, 32, 64, 0, 32, 0x90003000},
      {0, 32, 64, 32, 32, 0x90004000},
      {1, 32, 64, 32, 32, 0x90005000},
  };
  input_count = 4;
  output_address = 0x90023000;
#endif
  assert(word_count <= sizeof(device.words) / sizeof(device.words[0]));
  for (size_t index = 0; index < input_count; ++index)
    input_addresses[index] = windows[index].address;
  assert(atlas_host_init(
      &host, map,
      (atlas_host_mmio){read32, write32, &device},
      (atlas_host_memory){.coherent = 1}) == ATLAS_HOST_OK);
  uint8_t activation[MAX_SOURCE_BYTES], weight[MAX_SOURCE_BYTES];
  uint8_t output[2048], scratch[2048];
  atlas_host_buffer buffers[MAX_WINDOWS + 1];
  size_t source_columns = 32 * (input_count / 2);
  size_t logical_bytes = 32 * source_columns;
  for (size_t row = 0; row < 32; ++row)
    for (size_t k = 0; k < source_columns; ++k) {
      activation[row * source_columns + k] = (k / 32) & 1 ? 0x40 : 0x38;
      weight[row * source_columns + k] = (k / 32) & 1 ? 0x38 : 0xb8;
    }
#if defined(ATLAS_ORDERED_K_SOURCE_F32_DIAGNOSTIC)
  float source_f32[MAX_SOURCE_BYTES];
  uint8_t expected_codes[MAX_SOURCE_BYTES];
  for (unsigned side = 0; side < 2; ++side) {
    uint8_t *codes = side ? weight : activation;
    memcpy(expected_codes, codes, logical_bytes);
    for (size_t index = 0; index < logical_bytes; ++index)
      source_f32[index] = codes[index] == 0x40 ? 2.0f
                          : codes[index] == 0xb8 ? -1.0f : 1.0f;
    memset(codes, 0xa5, logical_bytes);
    if (!side) {
      source_f32[0] = -0.0f;
      assert(atlas_ordered_k_encode_source_f32(source_f32, codes)
             == MERLIN_F32_U8_OUT_OF_DOMAIN);
      assert(codes[0] == 0xa5 && codes[logical_bytes - 1] == 0xa5);
      source_f32[0] = 1.0f;
    }
    assert(atlas_ordered_k_encode_source_f32(source_f32, codes)
           == MERLIN_F32_U8_OK);
    assert(memcmp(codes, expected_codes, logical_bytes) == 0);
  }
#endif
  memset(output, 0xa5, sizeof(output));
#if defined(ATLAS_USE_GENERATED_K_DATA)
  atlas_ordered_k_tile tile;
  assert(atlas_ordered_k_bind(&tile, &host, &device, write_bytes, read_bytes,
      2, scratch, sizeof(scratch), buffers, input_count + 1) == 0);
  assert(tile.program_words == atlas_ordered_k_program_words &&
         tile.windows == atlas_ordered_k_windows);
#else
  atlas_ordered_k_tile tile = {
      .host = &host, .memory_context = &device,
      .write = write_bytes, .read = read_bytes, .pack = selected_pack,
      .program_words = words, .program_word_count = word_count,
      .windows = windows, .window_count = input_count,
      .output_address = output_address, .max_polls = 2,
      .scratch = scratch, .scratch_bytes = sizeof(scratch),
      .active_buffers = buffers, .active_buffer_capacity = input_count + 1,
  };
#endif
  atlas_host_result host_result;
  assert(atlas_ordered_k_run(&tile, activation, logical_bytes,
      weight, logical_bytes, output, sizeof(output), &host_result)
      == ATLAS_ORDERED_K_OK);
  assert(host_result == ATLAS_HOST_OK && device.writes == input_count && device.starts == 1);
  assert(device.words[2] == words[2]);
  for (size_t index = 0; index < input_count; ++index)
    for (size_t row = 0; row < 32; ++row)
      for (size_t k = 0; k < 32; ++k) {
        const uint8_t *source = index & 1 ? weight : activation;
        assert(device.inputs[index][row * 32 + k] ==
               source[row * source_columns + 32 * (index / 2) + k]);
      }
  for (size_t index = 0; index < sizeof(output); ++index)
    assert(output[index] == (uint8_t)(index ^ 0x5a));
#if defined(ATLAS_USE_GENERATED_K_DATA)
  float logical[1024];
  assert(atlas_ordered_k_readout_f32(output, logical) == 0);
  for (size_t row = 0; row < 32; ++row)
    for (size_t column = 0; column < 32; ++column) {
      size_t physical = column < 16 ? row * 16 + column
                                    : 512 + row * 16 + column - 16;
      uint32_t bits;
      memcpy(&bits, &logical[row * 32 + column], sizeof(bits));
      assert(bits == ((uint32_t)output[2 * physical] << 16 |
                      (uint32_t)output[2 * physical + 1] << 24));
    }
#endif

  unsigned writes = device.writes, starts = device.starts;
  memset(output, 0xa5, sizeof(output));
  activation[0] = 0x80; /* Negative zero is outside this MXU0 input domain. */
  assert(atlas_ordered_k_run(&tile, activation, logical_bytes,
      weight, logical_bytes, output, sizeof(output), &host_result)
      == ATLAS_ORDERED_K_INPUT_DOMAIN);
  assert(device.writes == writes && device.starts == starts && output[0] == 0xa5);
  activation[0] = 0x78; /* Exponent 15 remains outside the diagnostic runtime subset. */
  assert(atlas_ordered_k_run(&tile, activation, logical_bytes,
      weight, logical_bytes, output, sizeof(output), &host_result)
      == ATLAS_ORDERED_K_INPUT_DOMAIN);
  assert(device.writes == writes && device.starts == starts && output[0] == 0xa5);
  activation[0] = 0x38;

  tile.windows = windows;
  windows[2].k_start = 16;
  assert(atlas_ordered_k_run(&tile, activation, logical_bytes,
      weight, logical_bytes, output, sizeof(output), &host_result)
      == ATLAS_ORDERED_K_INVALID);
  assert(device.writes == writes && device.starts == starts && output[0] == 0xa5);
  windows[2].k_start = 32;

  device.timeout = 1;
  assert(atlas_ordered_k_run(&tile, activation, logical_bytes,
      weight, logical_bytes, output, sizeof(output), &host_result)
      == ATLAS_ORDERED_K_DEVICE_ERROR);
  assert(host_result == ATLAS_HOST_TIMEOUT && host.active && output[0] == 0xa5);
  assert(atlas_ordered_k_run(&tile, activation, logical_bytes,
      weight, logical_bytes, output, sizeof(output), &host_result)
      == ATLAS_ORDERED_K_INVALID);
  device.timeout = 0;
  assert(atlas_host_poll(&host) == ATLAS_HOST_OK);

  device.read_fails = 1;
  assert(atlas_ordered_k_run(&tile, activation, logical_bytes,
      weight, logical_bytes, output, sizeof(output), &host_result)
      == ATLAS_ORDERED_K_MEMORY_ERROR);
  assert(output[0] == 0xa5);
  puts("atlas-ordered-k-host: PASS");
  return 0;
}
