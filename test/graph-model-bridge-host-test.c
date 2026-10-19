#include "atlas_graph_model_wrapper.h"
#include "atlas_graph_data.h"
#include "atlas_graph_host_entry.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { IMEM_BASE = 0x1000u, CSR_BASE = 0x2000u };

__attribute__((noreturn)) void atlas_model_tile_fail(
    atlas_model_tile_status status, atlas_host_result device_result) {
  (void)status;
  (void)device_result;
  abort();
}

typedef struct {
  uint8_t inputs[ATLAS_GRAPH_INPUT_COUNT][1024];
  uint8_t constants[ATLAS_GRAPH_CONSTANT_COUNT][2048];
  uint8_t outputs[ATLAS_GRAPH_OUTPUT_COUNT][2048];
  size_t words;
  size_t writes;
  size_t reads;
  size_t starts;
  unsigned running;
} fake_device;

static uint32_t read32(void *context, uintptr_t address) {
  fake_device *device = context;
  if (address != CSR_BASE + 0x08u) return 0;
  if (device->running == 1u) {
    device->running = 2u;
    return 0;
  }
  if (device->running == 2u) device->running = 0u;
  return 5u;
}

static void write32(void *context, uintptr_t address, uint32_t word) {
  fake_device *device = context;
  if (address >= IMEM_BASE &&
      address < IMEM_BASE + ATLAS_GRAPH_PROGRAM_WORD_COUNT * 4u) {
    size_t index = (address - IMEM_BASE) / 4u;
    assert(word == atlas_graph_program_words[index]);
    ++device->words;
  } else if (address == CSR_BASE + 0x18u) {
    assert(word == 1u);
    assert(device->words == (device->starts + 1u) * ATLAS_GRAPH_PROGRAM_WORD_COUNT);
    ++device->starts;
    device->running = 1u;
  } else {
    abort();
  }
}

static int write_memory(void *context, uintptr_t address,
                        const uint8_t *source, size_t bytes) {
  fake_device *device = context;
  for (size_t index = 0; index < ATLAS_GRAPH_INPUT_COUNT; ++index) {
    if (address == atlas_graph_inputs[index].address && bytes == 1024u) {
      memcpy(device->inputs[index], source, bytes);
      ++device->writes;
      return 0;
    }
  }
  for (size_t index = 0; index < ATLAS_GRAPH_CONSTANT_COUNT; ++index) {
    if (address == atlas_graph_constants[index].address &&
        bytes == atlas_graph_constants[index].payload_bytes) {
      memcpy(device->constants[index], source, bytes);
      ++device->writes;
      return 0;
    }
  }
  return 1;
}

static int read_memory(void *context, uintptr_t address,
                       uint8_t *destination, size_t bytes) {
  fake_device *device = context;
  for (size_t index = 0; index < ATLAS_GRAPH_OUTPUT_COUNT; ++index) {
    if (address == atlas_graph_outputs[index].address && bytes == 2048u) {
      memcpy(destination, device->outputs[index], bytes);
      ++device->reads;
      return 0;
    }
  }
  return 1;
}

/* These physical bytes emulate device completion. This test checks host ABI
 * and buffer handling; the selected AtlasCore suite checks the computation. */
static void set_outputs(fake_device *device) {
  static const uint16_t values[] = {0x3f80u, 0x4000u, 0x4040u};
  for (size_t output = 0; output < ATLAS_GRAPH_OUTPUT_COUNT; ++output) {
    assert(output < sizeof(values) / sizeof(values[0]));
    uint16_t bits = values[output];
    for (size_t row = 0; row < 32u; ++row) {
      for (size_t column = 0; column < 32u; ++column) {
        size_t offset = (column / 16u) * 1024u + row * 32u +
                        (column % 16u) * 2u;
        device->outputs[output][offset] = (uint8_t)bits;
        device->outputs[output][offset + 1u] = (uint8_t)(bits >> 8);
      }
    }
  }
}

int main(void) {
  assert(ATLAS_GRAPH_INPUT_COUNT == 6u);
  assert(ATLAS_GRAPH_OUTPUT_COUNT == 2u || ATLAS_GRAPH_OUTPUT_COUNT == 3u);
  fake_device *device = calloc(1u, sizeof(*device));
  assert(device);
  set_outputs(device);
  atlas_host host;
  atlas_host_map map = {IMEM_BASE, CSR_BASE, 4096u};
  atlas_host_mmio mmio = {read32, write32, device};
  atlas_host_memory memory = {1, NULL, NULL};
  assert(atlas_host_init(&host, map, mmio, memory) == ATLAS_HOST_OK);
  atlas_tile_memory_io io = {write_memory, read_memory, device};
  assert(ATLAS_GRAPH_BINDING_FUNCTION(&host, io, 4u) == 0);

  float inputs[6][1024];
  float result_storage[ATLAS_GRAPH_OUTPUT_COUNT][1024];
  atlas_f32_memref2 arguments[6];
  atlas_f32_memref2 results[ATLAS_GRAPH_OUTPUT_COUNT];
  for (size_t index = 0; index < 6u; ++index) {
    for (size_t cell = 0; cell < 1024u; ++cell)
      inputs[index][cell] = index & 1u ? 1.0f : 0.5f;
    arguments[index] = (atlas_f32_memref2){
        inputs[index], inputs[index], 0, {32, 32}, {32, 1}};
  }
  for (size_t index = 0; index < ATLAS_GRAPH_OUTPUT_COUNT; ++index)
    results[index] = (atlas_f32_memref2){
        result_storage[index], result_storage[index], 0, {32, 32}, {32, 1}};
  for (size_t invocation = 0; invocation < 2u; ++invocation) {
    atlas_graph_invoke_host(arguments, results);
    assert(device->starts == invocation + 1u);
    assert(device->writes == (invocation + 1u) *
        (ATLAS_GRAPH_INPUT_COUNT + ATLAS_GRAPH_CONSTANT_COUNT));
    assert(device->reads == (invocation + 1u) * ATLAS_GRAPH_OUTPUT_COUNT);
    for (size_t output = 0; output < ATLAS_GRAPH_OUTPUT_COUNT; ++output) {
      assert(results[output].sizes[0] == 32 && results[output].sizes[1] == 32);
      for (size_t cell = 0; cell < 1024u; ++cell)
        assert(result_storage[output][cell] == (float)(output + 1u));
    }
    for (size_t index = 0; index < 6u; ++index) {
      uint8_t code = 0;
      uint32_t bits = index & 1u ? 0x3f800000u : 0x3f000000u;
      assert(merlin_f32_u8_encode_bits(bits, atlas_graph_inputs[index].codebook,
                                        &code) == MERLIN_F32_U8_OK);
      for (size_t cell = 0; cell < 1024u; ++cell) {
        uint8_t expected = code;
        if (invocation == 1u && index == 5u && cell == 0u)
          assert(merlin_f32_u8_encode_bits(0x3f000000u,
              atlas_graph_inputs[index].codebook, &expected) == MERLIN_F32_U8_OK);
        if (device->inputs[index][cell] != expected) {
          fprintf(stderr, "input %zu cell %zu got %u expected %u\n", index,
                  cell, device->inputs[index][cell], expected);
          abort();
        }
      }
    }
    for (size_t index = 0; index < ATLAS_GRAPH_CONSTANT_COUNT; ++index)
      assert(memcmp(device->constants[index], atlas_graph_constants[index].bytes,
                    atlas_graph_constants[index].payload_bytes) == 0);
    inputs[5][0] = invocation ? 1.0f : 0.5f;
  }
  free(device);
  return 0;
}
