#include "atlas_host.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

enum { IMEM = 0x20000, CSR = 0x40000 };

typedef struct {
  uint32_t words[8];
  unsigned status_reads;
  unsigned starts;
  unsigned imem_writes;
  unsigned cache_before;
  unsigned cache_after;
  int halt_reason;
  int cache_fails;
  int stale_ecall_once;
  int finish_before_poll;
  uint32_t retired;
} fake_device;

static uint32_t read32(void *context, uintptr_t address) {
  fake_device *device = context;
  if (address == CSR + 4)
    return device->retired;
  assert(address == CSR + 8);
  if (!device->starts)
    return 1;
  ++device->status_reads;
  if (device->finish_before_poll)
    return 5; /* Finished before the first poll; only the count moved. */
  if (device->status_reads == 1 && device->stale_ecall_once) {
    device->stale_ecall_once = 0;
    return 5; /* Previous invocation, before this start is acknowledged. */
  }
  if (device->status_reads == 1)
    return 1; /* Stale halt just after the start write. */
  ++device->retired;
  if (device->status_reads == 2)
    return 0;
  return 1 | ((uint32_t)device->halt_reason << 1);
}

static void write32(void *context, uintptr_t address, uint32_t value) {
  fake_device *device = context;
  if (address == CSR + 0x18) {
    assert(value == 1);
    ++device->starts;
    device->status_reads = 0;
    if (device->finish_before_poll)
      device->retired += 3; /* Runs to its ECALL before any poll. */
    return;
  }
  assert(address >= IMEM && address < IMEM + sizeof(device->words));
  assert((address & 3) == 0);
  device->words[(address - IMEM) / 4] = value;
  ++device->imem_writes;
}

static int cache_sync(void *context, const atlas_host_buffer *buffer,
                      int before_launch) {
  fake_device *device = context;
  assert(buffer->size_bytes == 32);
  if (before_launch)
    ++device->cache_before;
  else
    ++device->cache_after;
  return device->cache_fails;
}

static atlas_host setup(fake_device *device) {
  atlas_host host;
  atlas_host_map map = {IMEM, CSR, sizeof(device->words)};
  atlas_host_mmio mmio = {read32, write32, device};
  atlas_host_memory memory = {0, cache_sync, device};
  assert(atlas_host_init(&host, map, mmio, memory) == ATLAS_HOST_OK);
  return host;
}

static uint64_t tick(void *context) {
  uint64_t *counter = context;
  return ++*counter;
}

int main(void) {
  fake_device device = {.halt_reason = 2};
  atlas_host host = setup(&device);
  uint32_t program[] = {0x13, 0x73, 0x8067};
  atlas_host_buffer buffers[] = {
      {0x80002000, 32, ATLAS_HOST_DEVICE_READ},
      {0x80003000, 32, ATLAS_HOST_DEVICE_WRITE},
  };
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_NOT_READY);
  assert(atlas_host_load(&host, program, 3) == ATLAS_HOST_OK);
  assert(device.words[2] == program[2]);
  assert(atlas_host_load(&host, program, 9) == ATLAS_HOST_INVALID);
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_NOT_READY);
  assert(atlas_host_load(&host, program, 3) == ATLAS_HOST_OK);
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_OK);
  assert(device.cache_before == 2);
  assert(atlas_host_load(&host, program, 3) == ATLAS_HOST_BUSY);
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_BUSY);

  /* The host can prepare another tile while the accelerator runs. The caller
   * keeps the launched buffer descriptors alive until completion. */
  unsigned next_tile_prepared = 1;
  assert(next_tile_prepared && atlas_host_wait(&host, 2) == ATLAS_HOST_TIMEOUT);
  assert(host.active);
  assert(atlas_host_wait(&host, 1) == ATLAS_HOST_OK);
  assert(device.cache_after == 1);
  assert(atlas_host_poll(&host) == ATLAS_HOST_NOT_READY);

  device.stale_ecall_once = 1;
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_OK);
  assert(atlas_host_wait(&host, 1) == ATLAS_HOST_TIMEOUT);
  assert(host.active && device.cache_after == 1);
  assert(atlas_host_wait(&host, 2) == ATLAS_HOST_OK);
  assert(device.starts == 2 && device.cache_after == 2);

  /* A launch that halts before the first poll is never seen running; its
   * moved retired-instruction count still identifies its ECALL. */
  device.finish_before_poll = 1;
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_OK);
  assert(atlas_host_wait(&host, 2) == ATLAS_HOST_OK);
  assert(device.cache_after == 3);
  device.finish_before_poll = 0;

  device.cache_fails = 1;
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_CACHE_ERROR);
  device.cache_fails = 0;
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_OK);
  device.cache_fails = 1;
  assert(atlas_host_wait(&host, 3) == ATLAS_HOST_CACHE_ERROR);
  assert(host.active);
  device.cache_fails = 0;
  assert(atlas_host_poll(&host) == ATLAS_HOST_OK);
  assert(!host.active && device.cache_after == 5);

  device.halt_reason = 1;
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_OK);
  assert(atlas_host_wait(&host, 3) == ATLAS_HOST_BAD_HALT);
  assert(device.cache_after == 5);

  buffers[0].address = UINT32_MAX - 15;
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_INVALID);
  buffers[0].address = 0x80002000;
  device.cache_fails = 1;
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_CACHE_ERROR);
  assert(!host.active && device.starts == 5);

  atlas_host_memory unqualified = {0, 0, 0};
  atlas_host_mmio mmio = {read32, write32, &device};
  atlas_host_map map = {IMEM, CSR, 32};
  assert(atlas_host_init(&host, map, mmio, unqualified) == ATLAS_HOST_INVALID);
  fake_device reused = {.halt_reason = 2};
  host = setup(&reused);
  uint64_t ticks = 0;
  assert(atlas_host_set_clock(&host, tick, &ticks) == ATLAS_HOST_OK);
  static const uint32_t a[] = {0x13, 0x73}, duplicate[] = {0x13, 0x73}, b[] = {0x93, 0x73};
  assert(atlas_host_ensure_loaded(&host, a, 2) == ATLAS_HOST_OK);
  assert(atlas_host_ensure_loaded(&host, duplicate, 2) == ATLAS_HOST_OK);
  assert(reused.imem_writes == 2 && host.image_reuses == 1);
  assert(host.timings.load_ticks == 1);
  assert(atlas_host_ensure_loaded(&host, b, 2) == ATLAS_HOST_OK);
  assert(atlas_host_ensure_loaded(&host, a, 2) == ATLAS_HOST_OK);
  assert(reused.imem_writes == 6 && host.image_loads == 3);
  assert(atlas_host_ensure_loaded(&host, a, 9) == ATLAS_HOST_INVALID);
  assert(!host.program_loaded && !host.resident_words);
  assert(atlas_host_ensure_loaded(&host, a, 2) == ATLAS_HOST_OK);
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_OK);
  assert(atlas_host_ensure_loaded(&host, a, 2) == ATLAS_HOST_BUSY);
  assert(atlas_host_wait(&host, 1) == ATLAS_HOST_TIMEOUT);
  assert(atlas_host_invalidate_program(&host) == ATLAS_HOST_BUSY);
  /* The platform has reset the fake hardware before acknowledging it. */
  reused.starts = 0;
  reused.status_reads = 0;
  assert(atlas_host_reset_completed(&host) == ATLAS_HOST_OK);
  assert(!host.active && !host.program_loaded && !host.resident_words);
  assert(atlas_host_ensure_loaded(&host, a, 2) == ATLAS_HOST_OK);
  assert(reused.imem_writes == 10);
  assert(atlas_host_invalidate_program(&host) == ATLAS_HOST_OK);
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_NOT_READY);
  puts("atlas-host-launch: PASS");
  return 0;
}
