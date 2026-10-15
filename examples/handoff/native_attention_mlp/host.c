#include "atlas_host.h"
#include "atlas_program.h"
#include "atlas_captured_data.h"

#include <stdint.h>

#if !ATLAS_PROGRAM_BITSTREAM_BOUND && !defined(ATLAS_DIAGNOSTIC_ALLOW_UNBOUND)
#error "Bind the selected host map to an FPGA bitstream before deployment"
#endif

#if !defined(ATLAS_PLATFORM_CACHE_SYNC) && !defined(ATLAS_DIAGNOSTIC_ASSUME_COHERENT)
#error "Supply a qualified cache-sync callback or select diagnostic coherence"
#endif

#ifndef ATLAS_MAX_POLLS
#error "Set ATLAS_MAX_POLLS from a platform timing measurement"
#endif

#if ATLAS_CAPTURED_INPUT_COUNT != 6 || ATLAS_CAPTURED_OUTPUT_COUNT != 2
#error "This directed check requires six FP8 inputs and two ordered BF16 outputs"
#endif

/* The standalone ELF has no libc. Baremetal IDE supplies these via glossy. */
#ifdef ATLAS_STANDALONE_MEMOPS
void *memset(void *destination, int value, size_t count) {
  volatile uint8_t *bytes = (volatile uint8_t *)destination;
  for (size_t index = 0; index < count; ++index)
    bytes[index] = (uint8_t)value;
  return destination;
}

void *memcpy(void *destination, const void *source, size_t count) {
  volatile uint8_t *out = (volatile uint8_t *)destination;
  const volatile uint8_t *in = (const volatile uint8_t *)source;
  for (size_t index = 0; index < count; ++index)
    out[index] = in[index];
  return destination;
}
#endif

volatile uint32_t atlas_handoff_status;
volatile uint32_t atlas_handoff_mismatches[2];
volatile uint32_t atlas_handoff_first_output[2];
volatile uint32_t atlas_handoff_second_output[2];

#ifdef ATLAS_PLATFORM_CACHE_SYNC
extern int atlas_platform_cache_sync(void *context,
                                     const atlas_host_buffer *buffer,
                                     int before_launch);
#endif

static const uintptr_t input_addresses[] = {
    ATLAS_CAPTURED_INPUT_ADDRESS_0, ATLAS_CAPTURED_INPUT_ADDRESS_1,
    ATLAS_CAPTURED_INPUT_ADDRESS_2, ATLAS_CAPTURED_INPUT_ADDRESS_3,
    ATLAS_CAPTURED_INPUT_ADDRESS_4, ATLAS_CAPTURED_INPUT_ADDRESS_5,
};

static void prepare_input(unsigned run) {
#if ATLAS_CAPTURED_CONSTANT_COUNT > 0
  for (size_t blob = 0; blob < ATLAS_CAPTURED_CONSTANT_COUNT; ++blob) {
    const atlas_init_blob *item = &atlas_captured_constants[blob];
    volatile uint8_t *destination = (volatile uint8_t *)item->address;
    for (size_t cell = 0; cell < item->size; ++cell)
      destination[cell] = item->data[cell];
  }
#endif
  for (size_t input = 0; input < 6; ++input) {
    volatile uint8_t *destination = (volatile uint8_t *)input_addresses[input];
    for (size_t cell = 0; cell < ATLAS_CAPTURED_INPUT_BYTES; ++cell) {
      uint8_t value;
      if (input == 0)
        value = 0x38;  /* Token tile. */
      else if (input == 1 || input == 2)
        value = 0x10;  /* Q/K projection weights. */
      else if (input == 3)
        value = run == 0 || (cell / 32) % 2 == 0 ? 0x10 : 0x08;
      else
        value = (cell / 32 == cell % 32) ? 0x38 : 0;
      destination[cell] = value;
    }
  }
  for (size_t output = 0; output < ATLAS_CAPTURED_OUTPUT_COUNT; ++output) {
    volatile uint8_t *destination =
        (volatile uint8_t *)atlas_captured_outputs[output].address;
    for (size_t cell = 0; cell < atlas_captured_outputs[output].size; ++cell)
      destination[cell] = 0xa5;
  }
  volatile uint8_t *guard = (volatile uint8_t *)(
      atlas_captured_outputs[1].address + atlas_captured_outputs[1].size);
  for (size_t cell = 0; cell < 32; ++cell)
    guard[cell] = 0x5a;
}

static uint32_t inspect_output(unsigned run) {
  uint32_t mismatches = 0;
  for (size_t result = 0; result < ATLAS_CAPTURED_OUTPUT_COUNT; ++result) {
    volatile const uint16_t *output =
        (volatile const uint16_t *)atlas_captured_outputs[result].address;
    if (result == 0)
      atlas_handoff_first_output[run] = output[0];
    else
      atlas_handoff_second_output[run] = output[0];
    for (size_t cell = 0; cell < 1024; ++cell) {
      const size_t feature = (cell / 512) * 16 + cell % 16;
      const uint16_t expected = run == 0 || feature % 2 == 0
                                    ? 0x3f80u : 0x3f00u;
      mismatches += output[cell] != expected;
    }
  }
  volatile const uint8_t *guard = (volatile const uint8_t *)(
      atlas_captured_outputs[1].address + atlas_captured_outputs[1].size);
  for (size_t cell = 0; cell < 32; ++cell)
    mismatches += guard[cell] != 0x5a;
  atlas_handoff_mismatches[run] = mismatches;
  return mismatches;
}

int main(void) {
  atlas_handoff_status = 0xffffffffu;
#ifdef ATLAS_PLATFORM_CACHE_SYNC
  atlas_host_memory memory = {0, atlas_platform_cache_sync, 0};
#else
  /* Diagnostic build only; this does not qualify Shuttle cache coherence. */
  atlas_host_memory memory = {1, 0, 0};
#endif
  atlas_host_mmio mmio = {atlas_host_direct_read32,
                          atlas_host_direct_write32, 0};
  atlas_host host;
  atlas_host_result result = atlas_host_init(&host, atlas_program_map, mmio,
                                             memory);
  if (result != ATLAS_HOST_OK)
    return atlas_handoff_status = 0x100u + result;
  result = atlas_host_load(&host, atlas_program_words, ATLAS_PROGRAM_WORD_COUNT);
  if (result != ATLAS_HOST_OK)
    return atlas_handoff_status = 0x200u + result;

  atlas_host_buffer buffers[4] = {
      {ATLAS_CAPTURED_INPUT_REGION_ADDRESS, ATLAS_CAPTURED_INPUT_REGION_BYTES,
       ATLAS_HOST_DEVICE_READ},
      {ATLAS_CAPTURED_OUTPUT_ADDRESS_0, ATLAS_CAPTURED_OUTPUT_BYTES,
       ATLAS_HOST_DEVICE_WRITE},
      {ATLAS_CAPTURED_OUTPUT_ADDRESS_1, ATLAS_CAPTURED_OUTPUT_BYTES,
       ATLAS_HOST_DEVICE_WRITE},
      {ATLAS_CAPTURED_OUTPUT_ADDRESS_1 + ATLAS_CAPTURED_OUTPUT_BYTES, 32u,
       ATLAS_HOST_DEVICE_WRITE},
  };
  for (unsigned run = 0; run < 2; ++run) {
    prepare_input(run);
    result = atlas_host_start(&host, buffers,
                              sizeof(buffers) / sizeof(buffers[0]));
    if (result != ATLAS_HOST_OK)
      return atlas_handoff_status = 0x300u + 16u * run + result;
    result = atlas_host_wait(&host, ATLAS_MAX_POLLS);
    if (result != ATLAS_HOST_OK)
      return atlas_handoff_status = 0x400u + 16u * run + result;
    if (inspect_output(run))
      return atlas_handoff_status = 0x500u + run;
  }
  return atlas_handoff_status = 0;
}
