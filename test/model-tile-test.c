#include "atlas_model_tile.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { IMEM = 0x20000, CSR = 0x40000, LHS = 0x1000, RHS = 0x1400,
       OUTPUT = 0x2000 };

typedef struct {
  uint8_t dram[0x3000];
  uint32_t imem[4];
  unsigned starts;
  unsigned status_reads;
  unsigned writes;
  int fail_read;
  int stall;
} fake_device;

static uint32_t read32(void *context, uintptr_t address) {
  fake_device *device = context;
  assert(address == CSR + 8);
  if (device->starts && device->status_reads++ == 0)
    return 1; /* The start write clears the previous halt reason. */
  return device->starts >= 3 && device->stall ? 0 : 5;
}

static void write32(void *context, uintptr_t address, uint32_t value) {
  fake_device *device = context;
  if (address == CSR + 0x18) {
    assert(value == 1);
    ++device->starts;
    device->status_reads = 0;
    for (unsigned row = 0; row < ATLAS_MODEL_TILE_EDGE; ++row) {
      for (unsigned column = 0; column < ATLAS_MODEL_TILE_EDGE; ++column) {
        unsigned half = column >= 16;
        unsigned physical = half * 512 + row * 16 + column % 16;
        uint16_t code = half ? 0x4000 : 0x3f80;
        device->dram[OUTPUT + 2 * physical] = (uint8_t)code;
        device->dram[OUTPUT + 2 * physical + 1] = (uint8_t)(code >> 8);
      }
    }
    return;
  }
  assert(address >= IMEM && address < IMEM + sizeof(device->imem));
  device->imem[(address - IMEM) / 4] = value;
}

static int write_bytes(void *context, uintptr_t address,
                       const uint8_t *source, size_t bytes) {
  fake_device *device = context;
  assert(address + bytes <= sizeof(device->dram));
  memcpy(device->dram + address, source, bytes);
  ++device->writes;
  return 0;
}

static int read_bytes(void *context, uintptr_t address,
                      uint8_t *destination, size_t bytes) {
  fake_device *device = context;
  assert(address + bytes <= sizeof(device->dram));
  if (device->fail_read)
    return 1;
  memcpy(destination, device->dram + address, bytes);
  return 0;
}

static int pack(const float *source, uint8_t *destination) {
  for (unsigned index = 0; index < ATLAS_MODEL_TILE_FP8_BYTES; ++index) {
    if (source[index] < 0 || source[index] > 255)
      return 1;
    destination[index] = (uint8_t)source[index];
  }
  return 0;
}

static int unpack(const volatile uint8_t *source, float *destination) {
  for (unsigned row = 0; row < ATLAS_MODEL_TILE_EDGE; ++row)
    for (unsigned column = 0; column < ATLAS_MODEL_TILE_EDGE; ++column) {
      unsigned physical = (column / 16) * 512 + row * 16 + column % 16;
      uint32_t bits = ((uint32_t)source[2 * physical] |
                       ((uint32_t)source[2 * physical + 1] << 8)) << 16;
      memcpy(&destination[row * ATLAS_MODEL_TILE_EDGE + column], &bits, 4);
    }
  return 0;
}

static atlas_f32_memref2 descriptor(float *data) {
  return (atlas_f32_memref2){data, data, 0,
      {ATLAS_MODEL_TILE_EDGE, ATLAS_MODEL_TILE_EDGE}, {ATLAS_MODEL_TILE_EDGE, 1}};
}

int main(void) {
  fake_device device = {0};
  atlas_host host;
  atlas_host_map map = {IMEM, CSR, sizeof(device.imem)};
  atlas_host_mmio mmio = {read32, write32, &device};
  atlas_host_memory memory = {1, 0, 0};
  assert(atlas_host_init(&host, map, mmio, memory) == ATLAS_HOST_OK);
  uint32_t words[] = {0x13, 0x73, 0x8067};
  uint8_t constant[ATLAS_MODEL_TILE_FP8_BYTES];
  for (unsigned row = 0; row < ATLAS_MODEL_TILE_EDGE; ++row)
    for (unsigned column = 0; column < ATLAS_MODEL_TILE_EDGE; ++column)
      constant[row * ATLAS_MODEL_TILE_EDGE + column] = (uint8_t)(7 + column);
  _Alignas(float) uint8_t scratch[ATLAS_MODEL_TILE_SCRATCH_BYTES];
  atlas_model_tile binding = {
      .host = &host,
      .io = {write_bytes, read_bytes, &device},
      .program_words = words,
      .program_word_count = 3,
      .input_addresses = {LHS, RHS},
      .output_address = OUTPUT,
      .constant_bytes = constant,
      .constant_bytes_length = sizeof(constant),
      .constant_operand = 1,
      .transpose_input = {0, 1},
      .pack = {pack, pack},
      .unpack = unpack,
      .max_polls = 2,
      .scratch = scratch,
      .scratch_bytes = sizeof(scratch),
  };
  float activation[ATLAS_MODEL_TILE_FP8_BYTES];
  float weight[ATLAS_MODEL_TILE_FP8_BYTES];
  for (unsigned index = 0; index < ATLAS_MODEL_TILE_FP8_BYTES; ++index) {
    activation[index] = 3;
    weight[index] = (float)(7 + index / ATLAS_MODEL_TILE_EDGE);
  }
  atlas_f32_memref2 lhs = descriptor(activation), rhs = descriptor(weight);
  atlas_f32_memref2 output = {0};
  atlas_host_result device_status;
  assert(atlas_model_tile_run(&binding, &lhs, &rhs, &output, &device_status) == ATLAS_MODEL_TILE_OK);
  assert(device_status == ATLAS_HOST_OK && device.starts == 1 && device.writes == 2);
  assert(device.imem[2] == words[2] && device.dram[LHS] == 3 && device.dram[RHS] == 7);
  assert(device.dram[RHS + 1] == 8 && device.dram[RHS + ATLAS_MODEL_TILE_EDGE] == 7);
  assert(output.aligned[0] == 1 && output.aligned[16] == 2);
  free(output.allocated);

  weight[0] = 8;
  output = (atlas_f32_memref2){0};
  assert(atlas_model_tile_run(&binding, &lhs, &rhs, &output, &device_status) ==
         ATLAS_MODEL_TILE_CONSTANT_MISMATCH);
  assert(!output.allocated && device.starts == 1 && device.writes == 2);
  weight[0] = 7;

  lhs.sizes[0] = 31;
  assert(atlas_model_tile_run(&binding, &lhs, &rhs, &output, &device_status) == ATLAS_MODEL_TILE_INVALID);
  lhs.sizes[0] = 32;
  binding.constant_bytes_length--;
  assert(atlas_model_tile_run(&binding, &lhs, &rhs, &output, &device_status) == ATLAS_MODEL_TILE_INVALID);
  binding.constant_bytes_length++;
  binding.input_addresses[1] = LHS + 512;
  assert(atlas_model_tile_run(&binding, &lhs, &rhs, &output, &device_status) == ATLAS_MODEL_TILE_INVALID);
  binding.input_addresses[1] = RHS;

  binding.input_addresses[0] = UINT32_MAX - 1023;
  binding.input_addresses[1] = UINT32_MAX - 1535;
  assert(atlas_model_tile_run(&binding, &lhs, &rhs, &output, &device_status) == ATLAS_MODEL_TILE_INVALID);
  binding.input_addresses[0] = LHS;
  binding.input_addresses[1] = RHS;

  device.fail_read = 1;
  assert(atlas_model_tile_run(&binding, &lhs, &rhs, &output, &device_status) ==
         ATLAS_MODEL_TILE_MEMORY_ERROR);
  assert(!output.allocated && device.starts == 2);
  device.fail_read = 0;

  device.stall = 1;
  assert(atlas_model_tile_run(&binding, &lhs, &rhs, &output, &device_status) ==
         ATLAS_MODEL_TILE_DEVICE_ERROR);
  assert(device_status == ATLAS_HOST_TIMEOUT && host.active);
  assert(atlas_model_tile_run(&binding, &lhs, &rhs, &output, &device_status) == ATLAS_MODEL_TILE_INVALID);
  device.stall = 0;
  assert(atlas_host_poll(&host) == ATLAS_HOST_OK && !host.active);

  binding.constant_operand = 2;
  binding.constant_bytes = NULL;
  binding.constant_bytes_length = 0;
  weight[0] = 8;
  assert(atlas_model_tile_run(&binding, &lhs, &rhs, &output, &device_status) == ATLAS_MODEL_TILE_OK);
  assert(device.dram[RHS] == 8 && device.starts == 4);
  free(output.allocated);

  puts("atlas-model-tile: PASS");
  return 0;
}
