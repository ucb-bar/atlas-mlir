/* One-shot host program for an atlasc build: stage the constants and the
 * model's example inputs, launch the Atlas program once, and keep the raw
 * BF16 output tiles where a debugger or simulator harness can read them. */
#include "atlas_host.h"
#include "atlas_program.h"
#include "atlas_captured_data.h"
#include "atlas_run_inputs.h"

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

#define ATLAS_RUN_GUARD_BYTES 32u

/* The standalone ELF has no libc. */
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

/* 0 after a completed launch with an intact guard; otherwise the failing
 * step (0x100 init, 0x200 load, 0x300 start, 0x400 wait, 0x500 guard) plus
 * the atlas_host_result. */
volatile uint32_t atlas_run_status;
/* Each output in its DRAM layout: BF16 columns 0-15, then 16-31. */
volatile uint16_t atlas_run_outputs[ATLAS_CAPTURED_OUTPUT_COUNT]
                                   [ATLAS_CAPTURED_OUTPUT_BYTES / 2];

#ifdef ATLAS_PLATFORM_CACHE_SYNC
extern int atlas_platform_cache_sync(void *context,
                                     const atlas_host_buffer *buffer,
                                     int before_launch);
#endif

static void copy_blobs(const atlas_init_blob *blobs, size_t count) {
  for (size_t blob = 0; blob < count; ++blob) {
    volatile uint8_t *destination = (volatile uint8_t *)blobs[blob].address;
    for (size_t index = 0; index < blobs[blob].size; ++index)
      destination[index] = blobs[blob].data[index];
  }
}

static uintptr_t output_guard_address(void) {
  const atlas_output_slot *last =
      &atlas_captured_outputs[ATLAS_CAPTURED_OUTPUT_COUNT - 1];
  return last->address + last->size;
}

static void poison_outputs(void) {
  for (size_t result = 0; result < ATLAS_CAPTURED_OUTPUT_COUNT; ++result) {
    volatile uint8_t *output =
        (volatile uint8_t *)atlas_captured_outputs[result].address;
    for (size_t index = 0; index < atlas_captured_outputs[result].size; ++index)
      output[index] = 0xa5;
  }
  volatile uint8_t *guard = (volatile uint8_t *)output_guard_address();
  for (size_t index = 0; index < ATLAS_RUN_GUARD_BYTES; ++index)
    guard[index] = 0x5a;
}

static int guard_intact(void) {
  volatile const uint8_t *guard =
      (volatile const uint8_t *)output_guard_address();
  for (size_t index = 0; index < ATLAS_RUN_GUARD_BYTES; ++index)
    if (guard[index] != 0x5a)
      return 0;
  return 1;
}

static void keep_outputs(void) {
  for (size_t result = 0; result < ATLAS_CAPTURED_OUTPUT_COUNT; ++result) {
    volatile const uint16_t *output =
        (volatile const uint16_t *)atlas_captured_outputs[result].address;
    for (size_t index = 0; index < ATLAS_CAPTURED_OUTPUT_BYTES / 2; ++index)
      atlas_run_outputs[result][index] = output[index];
  }
}

int main(void) {
  atlas_run_status = 0xffffffffu;
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
    return atlas_run_status = 0x100u + result;

  copy_blobs(atlas_captured_constants, ATLAS_CAPTURED_CONSTANT_COUNT);
  copy_blobs(atlas_run_inputs, ATLAS_RUN_INPUT_COUNT);
  poison_outputs();
  result = atlas_host_load(&host, atlas_program_words,
                           ATLAS_PROGRAM_WORD_COUNT);
  if (result != ATLAS_HOST_OK)
    return atlas_run_status = 0x200u + result;

  atlas_host_buffer buffers[ATLAS_CAPTURED_OUTPUT_COUNT + 2];
  buffers[0] = (atlas_host_buffer){ATLAS_CAPTURED_INPUT_REGION_ADDRESS,
                                  ATLAS_CAPTURED_INPUT_REGION_BYTES,
                                  ATLAS_HOST_DEVICE_READ};
  for (size_t index = 0; index < ATLAS_CAPTURED_OUTPUT_COUNT; ++index)
    buffers[index + 1] = (atlas_host_buffer){atlas_captured_outputs[index].address,
                                             atlas_captured_outputs[index].size,
                                             ATLAS_HOST_DEVICE_WRITE};
  buffers[ATLAS_CAPTURED_OUTPUT_COUNT + 1] = (atlas_host_buffer){
      output_guard_address(), ATLAS_RUN_GUARD_BYTES, ATLAS_HOST_DEVICE_WRITE};
  result = atlas_host_start(&host, buffers,
                            sizeof(buffers) / sizeof(buffers[0]));
  if (result != ATLAS_HOST_OK)
    return atlas_run_status = 0x300u + result;
  result = atlas_host_wait(&host, ATLAS_MAX_POLLS);
  if (result != ATLAS_HOST_OK)
    return atlas_run_status = 0x400u + result;
  if (!guard_intact())
    return atlas_run_status = 0x500u;
  keep_outputs();
  return atlas_run_status = 0;
}
