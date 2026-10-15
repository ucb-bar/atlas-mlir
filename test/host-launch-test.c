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
  unsigned cache_before;
  unsigned cache_after;
  int halt_reason;
  int cache_fails;
  int stale_ecall_once;
} fake_device;

static uint32_t read32(void *context, uintptr_t address) {
  fake_device *device = context;
  assert(address == CSR + 8);
  if (!device->starts)
    return 1;
  ++device->status_reads;
  if (device->status_reads == 1 && device->stale_ecall_once) {
    device->stale_ecall_once = 0;
    return 5; /* Previous invocation, before this start is acknowledged. */
  }
  if (device->status_reads == 1)
    return 1; /* Stale halt just after the start write. */
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
    return;
  }
  assert(address >= IMEM && address < IMEM + sizeof(device->words));
  assert((address & 3) == 0);
  device->words[(address - IMEM) / 4] = value;
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

  device.cache_fails = 1;
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_CACHE_ERROR);
  device.cache_fails = 0;
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_OK);
  device.cache_fails = 1;
  assert(atlas_host_wait(&host, 3) == ATLAS_HOST_CACHE_ERROR);
  assert(host.active);
  device.cache_fails = 0;
  assert(atlas_host_poll(&host) == ATLAS_HOST_OK);
  assert(!host.active && device.cache_after == 4);

  device.halt_reason = 1;
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_OK);
  assert(atlas_host_wait(&host, 3) == ATLAS_HOST_BAD_HALT);
  assert(device.cache_after == 4);

  buffers[0].address = UINT32_MAX - 15;
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_INVALID);
  buffers[0].address = 0x80002000;
  device.cache_fails = 1;
  assert(atlas_host_start(&host, buffers, 2) == ATLAS_HOST_CACHE_ERROR);
  assert(!host.active && device.starts == 4);

  atlas_host_memory unqualified = {0, 0, 0};
  atlas_host_mmio mmio = {read32, write32, &device};
  atlas_host_map map = {IMEM, CSR, 32};
  assert(atlas_host_init(&host, map, mmio, unqualified) == ATLAS_HOST_INVALID);
  puts("atlas-host-launch: PASS");
  return 0;
}
