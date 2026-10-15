#include "atlas_graph_memref.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { BASE = 0x90000000u, MEMORY_BYTES = 4096 };

typedef struct {
  uint8_t memory[MEMORY_BYTES];
  unsigned state;
  unsigned starts;
  unsigned writes;
  unsigned reads;
  int fail_read;
} fake_device;

static uint32_t read32(void *context, uintptr_t address) {
  fake_device *device = context;
  if (address != 0x2008u) return 0;
  if (device->state == 1u) {
    device->state = 2u;
    return 0;
  }
  if (device->state == 2u) device->state = 0u;
  return 5u; /* halted, ECALL */
}

static void write32(void *context, uintptr_t address, uint32_t value) {
  fake_device *device = context;
  if (address != 0x2018u || value != 1u) return;
  ++device->starts;
  device->state = 1u;
  const uint16_t first[] = {0x3f00u, 0x3fc0u, 0x3f80u, 0x4000u};
  const uint16_t second[] = {0x4040u, 0x40a0u, 0x4080u, 0x40c0u};
  memcpy(device->memory + 0x400, first, sizeof first);
  memcpy(device->memory + 0x500, second, sizeof second);
}

static int write_memory(void *context, uintptr_t address,
                        const uint8_t *source, size_t bytes) {
  fake_device *device = context;
  if (address < BASE || address - BASE > MEMORY_BYTES ||
      bytes > MEMORY_BYTES - (address - BASE)) return 1;
  memcpy(device->memory + address - BASE, source, bytes);
  ++device->writes;
  return 0;
}

static int read_memory(void *context, uintptr_t address,
                       uint8_t *destination, size_t bytes) {
  fake_device *device = context;
  if (device->fail_read || address < BASE || address - BASE > MEMORY_BYTES ||
      bytes > MEMORY_BYTES - (address - BASE)) return 1;
  memcpy(destination, device->memory + address - BASE, bytes);
  ++device->reads;
  return 0;
}

int main(void) {
  static const uint8_t codes[] = {0x30u, 0x38u};
  static const uint32_t midpoints[] = {0x3f400000u};
  static const merlin_f32_u8_codebook codebook = {
      codes, midpoints, 2u, 0x3e800000u,
      {1u, (1u << 16) | (1u << 24), 0, 0, 0, 0, 0, 0}, 0u};
  static const uint32_t program[] = {0x00000013u, 0x00000073u};
  static const atlas_graph_input_slot input_slots[] = {
      {BASE, 2, 2, 4, &codebook}, {BASE + 0x100, 2, 2, 4, &codebook}};
  static const uint8_t constant[] = {0x80u, 0x3fu};
  static const atlas_graph_constant_slot constant_slots[] = {
      {BASE + 0x200, constant, sizeof constant}};
  static const atlas_graph_output_slot output_slots[] = {
      {BASE + 0x400, 2, 2, 1, 8}, {BASE + 0x500, 2, 2, 1, 8}};
  fake_device device = {0};
  atlas_host host;
  atlas_host_map map = {0x1000u, 0x2000u, 1024u};
  atlas_host_mmio mmio = {read32, write32, &device};
  atlas_host_memory memory = {1, NULL, NULL};
  assert(atlas_host_init(&host, map, mmio, memory) == ATLAS_HOST_OK);
  atlas_host_buffer active[5];
  atlas_graph_memref_binding binding = {
      .host = &host,
      .io = {write_memory, read_memory, &device},
      .program_words = program, .program_word_count = 2,
      .inputs = input_slots, .input_count = 2,
      .constants = constant_slots, .constant_count = 1,
      .outputs = output_slots, .output_count = 2,
      .max_polls = 4, .active_buffers = active, .active_buffer_capacity = 5};
  float first_source[] = {0.5f, 1.0f, 0.5f, 1.0f};
  float second_source[] = {0.0f, 1.0f, 1.0f, 0.0f};
  atlas_f32_memref2 first = {first_source, first_source, 0, {2, 2}, {2, 1}};
  atlas_f32_memref2 second = {second_source, second_source, 0, {2, 2}, {2, 1}};
  const atlas_f32_memref2 *arguments[] = {&first, &second};
  atlas_f32_memref2 results[2] = {0};
  atlas_host_result device_result = ATLAS_HOST_INVALID;
  assert(atlas_graph_run_memrefs(&binding, arguments, 2, results, 2,
                                 &device_result) == ATLAS_MODEL_TILE_OK);
  assert(device_result == ATLAS_HOST_OK && device.starts == 1u);
  assert(device.writes == 3u && device.reads == 2u);
  assert(memcmp(device.memory, "\x30\x38\x30\x38", 4) == 0);
  assert(memcmp(device.memory + 0x100, "\x00\x38\x38\x00", 4) == 0);
  assert(memcmp(device.memory + 0x200, constant, sizeof constant) == 0);
  const float expected_first[] = {0.5f, 1.0f, 1.5f, 2.0f};
  const float expected_second[] = {3.0f, 4.0f, 5.0f, 6.0f};
  assert(memcmp(results[0].aligned, expected_first, sizeof expected_first) == 0);
  assert(memcmp(results[1].aligned, expected_second, sizeof expected_second) == 0);
  free(results[0].allocated);
  free(results[1].allocated);

  atlas_f32_memref2 unchanged[2] = {{0}, {0}};
  unchanged[0].offset = 17;
  unchanged[1].offset = 23;
  unsigned before = device.starts;
  first_source[0] = NAN;
  assert(atlas_graph_run_memrefs(&binding, arguments, 2, unchanged, 2,
                                 &device_result) == ATLAS_MODEL_TILE_INPUT_DOMAIN);
  assert(device.starts == before && unchanged[0].offset == 17 &&
         unchanged[1].offset == 23);
  first_source[0] = 0.5f;

  device.fail_read = 1;
  assert(atlas_graph_run_memrefs(&binding, arguments, 2, unchanged, 2,
                                 &device_result) == ATLAS_MODEL_TILE_MEMORY_ERROR);
  assert(unchanged[0].offset == 17 && unchanged[1].offset == 23);
  device.fail_read = 0;

  atlas_graph_output_slot overlapped[] = {output_slots[0], output_slots[1]};
  overlapped[0].address = input_slots[0].address;
  binding.outputs = overlapped;
  before = device.starts;
  assert(atlas_graph_run_memrefs(&binding, arguments, 2, unchanged, 2,
                                 &device_result) == ATLAS_MODEL_TILE_INVALID);
  assert(device.starts == before);
  return 0;
}
