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

#if ATLAS_CAPTURED_OUTPUT_COUNT < 1 || ATLAS_CAPTURED_OUTPUT_COUNT > 2
#error "This directed host example supports one or two result tiles"
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

/* A debugger can read these after the bare-metal entry loop. */
volatile uint32_t atlas_handoff_status;
volatile uint32_t atlas_handoff_mismatches[3];
volatile uint32_t atlas_handoff_first_output[3];
volatile uint32_t atlas_handoff_second_output[3];

#ifdef ATLAS_PLATFORM_CACHE_SYNC
extern int atlas_platform_cache_sync(void *context,
                                     const atlas_host_buffer *buffer,
                                     int before_launch);
#endif

static void copy_constants(void) {
  for (size_t blob = 0; blob < ATLAS_CAPTURED_CONSTANT_COUNT; ++blob) {
    const atlas_init_blob *item = &atlas_captured_constants[blob];
    volatile uint8_t *destination = (volatile uint8_t *)item->address;
    for (size_t index = 0; index < item->size; ++index)
      destination[index] = item->data[index];
  }
}

static uintptr_t output_guard_address(void) {
  const atlas_output_slot *last =
      &atlas_captured_outputs[ATLAS_CAPTURED_OUTPUT_COUNT - 1];
  return last->address + last->size;
}

static void prepare_input(unsigned run) {
  volatile uint8_t *input = (volatile uint8_t *)ATLAS_CAPTURED_INPUT_ADDRESS;
  for (size_t index = 0; index < ATLAS_CAPTURED_INPUT_BYTES; ++index) {
    if (run == 2)
      input[index] = (index % 32u) % 2u ? 0x38u : 0x30u;
    else
      input[index] = run == 0 ? 0x30u : 0x38u;
  }
  for (size_t result = 0; result < ATLAS_CAPTURED_OUTPUT_COUNT; ++result) {
    volatile uint8_t *output =
        (volatile uint8_t *)atlas_captured_outputs[result].address;
    for (size_t index = 0; index < atlas_captured_outputs[result].size; ++index)
      output[index] = 0xa5;
  }
  volatile uint8_t *guard = (volatile uint8_t *)output_guard_address();
  for (size_t index = 0; index < 32; ++index)
    guard[index] = 0x5a;
}

static uint16_t expected_cell(unsigned run, size_t result,
                             size_t physical_index) {
  if (ATLAS_CAPTURED_OUTPUT_COUNT == 2 && result == 0)
    return run == 0 ? 0x3e00u : 0x3e80u;
  if (run == 0)
    return 0x3d80u;
  if (run == 1)
    return 0x3e00u;
  /* BF16 register pairs store columns 0-15 before columns 16-31. */
  size_t column = (physical_index / 512u) * 16u + physical_index % 16u;
  return column % 2u ? 0x3d80u : 0x3e00u;
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
    for (size_t index = 0; index < 1024; ++index)
      mismatches += output[index] != expected_cell(run, result, index);
  }
  volatile const uint8_t *guard =
      (volatile const uint8_t *)output_guard_address();
  for (size_t index = 0; index < 32; ++index)
    mismatches += guard[index] != 0x5a;
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

  copy_constants();
  result = atlas_host_load(&host, atlas_program_words,
                           ATLAS_PROGRAM_WORD_COUNT);
  if (result != ATLAS_HOST_OK)
    return atlas_handoff_status = 0x200u + result;

  atlas_host_buffer buffers[ATLAS_CAPTURED_OUTPUT_COUNT + 2];
  buffers[0] = (atlas_host_buffer){ATLAS_CAPTURED_INPUT_REGION_ADDRESS,
                                  ATLAS_CAPTURED_INPUT_REGION_BYTES,
                                  ATLAS_HOST_DEVICE_READ};
  for (size_t index = 0; index < ATLAS_CAPTURED_OUTPUT_COUNT; ++index)
    buffers[index + 1] = (atlas_host_buffer){atlas_captured_outputs[index].address,
                                             atlas_captured_outputs[index].size,
                                             ATLAS_HOST_DEVICE_WRITE};
  buffers[ATLAS_CAPTURED_OUTPUT_COUNT + 1] =
      (atlas_host_buffer){output_guard_address(), 32u, ATLAS_HOST_DEVICE_WRITE};
  const unsigned run_count = ATLAS_CAPTURED_OUTPUT_COUNT == 1 ? 3u : 2u;
  for (unsigned run = 0; run < run_count; ++run) {
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
