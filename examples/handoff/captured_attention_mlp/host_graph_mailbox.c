#include "atlas_graph_model_wrapper.h"
#include "atlas_graph_data.h"
#include "atlas_graph_host_entry.h"

#include <stddef.h>
#include <stdint.h>

#if !defined(ATLAS_DIAGNOSTIC_ALLOW_UNBOUND) && !defined(ATLAS_PLATFORM_MAP_BOUND)
#error "Bind the Atlas MMIO and DRAM map to the selected SoC build"
#endif
#if !defined(ATLAS_DIAGNOSTIC_ASSUME_COHERENT) && !defined(ATLAS_PLATFORM_CACHE_SYNC)
#error "Supply platform cache maintenance before using Atlas DMA"
#endif
#ifndef ATLAS_MAX_POLLS
#error "Set a poll bound for the selected platform"
#endif

_Static_assert(ATLAS_GRAPH_INPUT_COUNT == 6u &&
                   ATLAS_GRAPH_OUTPUT_COUNT >= 1u &&
                   ATLAS_GRAPH_OUTPUT_COUNT <= 16u,
               "This mailbox example needs six arguments and bounded results");

enum { ATLAS_GRAPH_CELLS = 1024u, ATLAS_GRAPH_HEAP_BYTES = 131072u };
enum { ATLAS_GRAPH_BUSY = 0xffffffffu, ATLAS_GRAPH_GUARD = 0x76a31c5bu };

typedef struct {
  volatile uint32_t request_id;
  volatile uint32_t response_id;
  volatile uint32_t status;
  volatile uint32_t reserved;
  uint32_t guard_before;
  float inputs[ATLAS_GRAPH_INPUT_COUNT][ATLAS_GRAPH_CELLS];
  float outputs[ATLAS_GRAPH_OUTPUT_COUNT][ATLAS_GRAPH_CELLS];
  uint32_t guard_after;
} atlas_graph_mailbox_type;

_Static_assert(offsetof(atlas_graph_mailbox_type, request_id) == 0u &&
                   offsetof(atlas_graph_mailbox_type, response_id) == 4u &&
                   offsetof(atlas_graph_mailbox_type, status) == 8u &&
                   offsetof(atlas_graph_mailbox_type, inputs) == 20u &&
                   offsetof(atlas_graph_mailbox_type, outputs) == 24596u &&
                   offsetof(atlas_graph_mailbox_type, guard_after) ==
                       24596u + 4096u * ATLAS_GRAPH_OUTPUT_COUNT &&
                   sizeof(atlas_graph_mailbox_type) ==
                       24600u + 4096u * ATLAS_GRAPH_OUTPUT_COUNT,
               "The runner-facing mailbox layout changed");

/* The runner writes all six f32 argument tiles, then request_id. It waits for
 * response_id before changing any input. The first five arguments are model
 * weights for this capture; the sixth is the invocation's token tile. */
__attribute__((aligned(64))) atlas_graph_mailbox_type atlas_graph_mailbox;
volatile uint32_t atlas_graph_heap_peak;
volatile uint32_t atlas_graph_last_device_status;

/* This fixed-size allocator backs the MLIR output temporaries and graph
 * staging. All allocations are released together after an invocation. */
static unsigned char atlas_graph_heap[ATLAS_GRAPH_HEAP_BYTES]
    __attribute__((aligned(16)));
static size_t atlas_graph_heap_used;

void *malloc(size_t bytes) {
  if (bytes > SIZE_MAX - 15u || atlas_graph_heap_used > SIZE_MAX - 15u)
    return 0;
  size_t start = (atlas_graph_heap_used + 15u) & ~(size_t)15u;
  if (start > ATLAS_GRAPH_HEAP_BYTES || bytes > ATLAS_GRAPH_HEAP_BYTES - start)
    return 0;
  atlas_graph_heap_used = start + bytes;
  if (atlas_graph_heap_peak < atlas_graph_heap_used)
    atlas_graph_heap_peak = (uint32_t)atlas_graph_heap_used;
  return atlas_graph_heap + start;
}

void free(void *address) { (void)address; }

void *memcpy(void *destination, const void *source, size_t bytes) {
  unsigned char *out = destination;
  const unsigned char *in = source;
  for (size_t index = 0; index < bytes; ++index) out[index] = in[index];
  return destination;
}

void *memset(void *destination, int value, size_t bytes) {
  unsigned char *out = destination;
  for (size_t index = 0; index < bytes; ++index) out[index] = (unsigned char)value;
  return destination;
}

static int direct_write(void *context, uintptr_t address,
                        const uint8_t *source, size_t bytes) {
  (void)context;
  if (!address || address > UINT32_MAX || !source || !bytes ||
      bytes - 1u > UINT32_MAX - address)
    return 1;
  volatile uint8_t *destination = (volatile uint8_t *)address;
  for (size_t index = 0; index < bytes; ++index) destination[index] = source[index];
  __sync_synchronize();
  return 0;
}

static int direct_read(void *context, uintptr_t address,
                       uint8_t *destination, size_t bytes) {
  (void)context;
  if (!address || address > UINT32_MAX || !destination || !bytes ||
      bytes - 1u > UINT32_MAX - address)
    return 1;
  __sync_synchronize();
  const volatile uint8_t *source = (const volatile uint8_t *)address;
  for (size_t index = 0; index < bytes; ++index) destination[index] = source[index];
  return 0;
}

#ifdef ATLAS_PLATFORM_CACHE_SYNC
extern int atlas_platform_cache_sync(void *context, const atlas_host_buffer *buffer,
                                     int before_launch);
#endif
#ifdef ATLAS_PLATFORM_MAP_BOUND
extern const atlas_host_map atlas_selected_host_map;
#endif

__attribute__((noreturn)) void atlas_model_tile_fail(
    atlas_model_tile_status status, atlas_host_result device_result) {
  atlas_graph_last_device_status = (uint32_t)device_result;
  atlas_graph_mailbox.status = 0x10000u + ((uint32_t)status << 8) +
                               (uint32_t)device_result;
  __sync_synchronize();
  atlas_graph_mailbox.response_id = atlas_graph_mailbox.request_id;
  for (;;) __asm__ volatile("wfi");
}

int main(void) {
  atlas_host host;
  atlas_host_mmio mmio = {
      atlas_host_direct_read32, atlas_host_direct_write32, 0};
#ifdef ATLAS_PLATFORM_CACHE_SYNC
  atlas_host_memory memory = {0, atlas_platform_cache_sync, 0};
#else
  atlas_host_memory memory = {1, 0, 0};
#endif
#ifdef ATLAS_PLATFORM_MAP_BOUND
  atlas_host_map map = atlas_selected_host_map;
#else
  atlas_host_map map = {0x00020000u, 0x00040000u, 131072u};
#endif
  atlas_graph_mailbox.guard_before = ATLAS_GRAPH_GUARD;
  atlas_graph_mailbox.guard_after = ATLAS_GRAPH_GUARD;
  atlas_graph_mailbox.status = 0;
  atlas_graph_mailbox.response_id = 0;
  atlas_host_result result = atlas_host_init(&host, map, mmio, memory);
  if (result != ATLAS_HOST_OK) {
    atlas_graph_mailbox.status = 0x20000u + (uint32_t)result;
    return 1;
  }
  atlas_tile_memory_io io = {direct_write, direct_read, 0};
  if (ATLAS_GRAPH_BINDING_FUNCTION(&host, io, ATLAS_MAX_POLLS)) {
    atlas_graph_mailbox.status = 0x30000u;
    return 1;
  }

  for (;;) {
    uint32_t request = atlas_graph_mailbox.request_id;
    if (request == 0 || request == atlas_graph_mailbox.response_id) {
      /* The runner does not provide an interrupt for a mailbox write. */
      __asm__ volatile("nop" ::: "memory");
      continue;
    }
    __sync_synchronize();
    atlas_graph_mailbox.status = ATLAS_GRAPH_BUSY;
    atlas_graph_heap_used = 0;
    atlas_f32_memref2 arguments[ATLAS_GRAPH_INPUT_COUNT];
    atlas_f32_memref2 outputs[ATLAS_GRAPH_OUTPUT_COUNT];
    for (size_t index = 0; index < ATLAS_GRAPH_INPUT_COUNT; ++index) {
      float *data = atlas_graph_mailbox.inputs[index];
      arguments[index] = (atlas_f32_memref2){
          data, data, 0, {32, 32}, {32, 1}};
    }
    for (size_t index = 0; index < ATLAS_GRAPH_OUTPUT_COUNT; ++index) {
      float *data = atlas_graph_mailbox.outputs[index];
      outputs[index] = (atlas_f32_memref2){
          data, data, 0, {32, 32}, {32, 1}};
    }
    atlas_graph_invoke_host(arguments, outputs);
    atlas_graph_mailbox.status =
        atlas_graph_mailbox.guard_before == ATLAS_GRAPH_GUARD &&
        atlas_graph_mailbox.guard_after == ATLAS_GRAPH_GUARD ? 0u : 0x40000u;
    __sync_synchronize();
    atlas_graph_mailbox.response_id = request;
  }
}
