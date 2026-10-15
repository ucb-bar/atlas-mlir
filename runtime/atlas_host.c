#include "atlas_host.h"

#include <limits.h>

enum { ATLAS_CSR_STATUS = 0x08, ATLAS_CSR_EXEC_CONTROL = 0x18 };
enum { ATLAS_HALTED = 1, ATLAS_HALT_REASON_SHIFT = 1, ATLAS_ECALL = 2 };

uint32_t atlas_host_direct_read32(void *context, uintptr_t address) {
  (void)context;
  __sync_synchronize();
  return *(volatile const uint32_t *)address;
}

void atlas_host_direct_write32(void *context, uintptr_t address, uint32_t word) {
  (void)context;
  *(volatile uint32_t *)address = word;
  __sync_synchronize();
}

atlas_host_result atlas_host_init(atlas_host *host, atlas_host_map map,
                                  atlas_host_mmio mmio, atlas_host_memory memory) {
  if (!host || !mmio.read32 || !mmio.write32 ||
      (map.imem_base & 3) || (map.csr_base & 3) ||
      map.imem_capacity_bytes < 4 || (map.imem_capacity_bytes & 3) ||
      map.imem_capacity_bytes > UINTPTR_MAX - map.imem_base ||
      map.csr_base > UINTPTR_MAX - ATLAS_CSR_EXEC_CONTROL ||
      (!memory.coherent && !memory.sync))
    return ATLAS_HOST_INVALID;
  *host = (atlas_host){0};
  host->map = map;
  host->mmio = mmio;
  host->memory = memory;
  return ATLAS_HOST_OK;
}

atlas_host_result atlas_host_load(atlas_host *host, const uint32_t *words,
                                  size_t word_count) {
  if (!host)
    return ATLAS_HOST_INVALID;
  if (host->active)
    return ATLAS_HOST_BUSY;
  /* A failed replacement must not leave the previous program launchable. */
  host->program_loaded = 0;
  if (!words || !word_count ||
      word_count > host->map.imem_capacity_bytes / sizeof(uint32_t))
    return ATLAS_HOST_INVALID;
  if (!(host->mmio.read32(host->mmio.context,
                          host->map.csr_base + ATLAS_CSR_STATUS) & ATLAS_HALTED))
    return ATLAS_HOST_NOT_READY;
  for (size_t index = 0; index < word_count; ++index)
    host->mmio.write32(host->mmio.context, host->map.imem_base + 4 * index,
                       words[index]);
  host->program_loaded = 1;
  return ATLAS_HOST_OK;
}

atlas_host_result atlas_host_start(atlas_host *host,
                                   const atlas_host_buffer *buffers,
                                   size_t buffer_count) {
  if (!host || (buffer_count && !buffers))
    return ATLAS_HOST_INVALID;
  if (host->active)
    return ATLAS_HOST_BUSY;
  if (!host->program_loaded)
    return ATLAS_HOST_NOT_READY;
  if (!(host->mmio.read32(host->mmio.context,
                          host->map.csr_base + ATLAS_CSR_STATUS) & ATLAS_HALTED))
    return ATLAS_HOST_NOT_READY;
  for (size_t index = 0; index < buffer_count; ++index) {
    const atlas_host_buffer *buffer = &buffers[index];
    if (!buffer->size_bytes || buffer->address > UINT32_MAX ||
        buffer->size_bytes - 1 > UINT32_MAX - buffer->address ||
        buffer->access > ATLAS_HOST_DEVICE_READ_WRITE)
      return ATLAS_HOST_INVALID;
  }
  if (!host->memory.coherent) {
    for (size_t index = 0; index < buffer_count; ++index)
      if (host->memory.sync(host->memory.context, &buffers[index], 1))
        return ATLAS_HOST_CACHE_ERROR;
  }
  host->active_buffers = buffers;
  host->active_buffer_count = buffer_count;
  host->active = 1;
  host->launch_acknowledged = 0;
  host->mmio.write32(host->mmio.context,
                     host->map.csr_base + ATLAS_CSR_EXEC_CONTROL, 1);
  return ATLAS_HOST_OK;
}

atlas_host_result atlas_host_poll(atlas_host *host) {
  if (!host || !host->active)
    return ATLAS_HOST_NOT_READY;
  uint32_t status = host->mmio.read32(host->mmio.context,
                                      host->map.csr_base + ATLAS_CSR_STATUS);
  if (!(status & ATLAS_HALTED)) {
    host->launch_acknowledged = 1;
    return ATLAS_HOST_BUSY;
  }
  unsigned reason = (status >> ATLAS_HALT_REASON_SHIFT) & 3;
  if (!reason) {
    host->launch_acknowledged = 1;
    return ATLAS_HOST_BUSY;
  }
  /* A previous ECALL may remain visible until the start write reaches CSRFile.
   * That write clears the reason before the selected core begins execution. */
  if (!host->launch_acknowledged)
    return ATLAS_HOST_BUSY;
  if (reason != ATLAS_ECALL) {
    host->active = 0;
    host->launch_acknowledged = 0;
    host->active_buffers = 0;
    host->active_buffer_count = 0;
    return ATLAS_HOST_BAD_HALT;
  }
  if (!host->memory.coherent) {
    for (size_t index = 0; index < host->active_buffer_count; ++index)
      if (host->active_buffers[index].access != ATLAS_HOST_DEVICE_READ &&
          host->memory.sync(host->memory.context, &host->active_buffers[index], 0))
        return ATLAS_HOST_CACHE_ERROR;
  }
  host->active = 0;
  host->launch_acknowledged = 0;
  host->active_buffers = 0;
  host->active_buffer_count = 0;
  return ATLAS_HOST_OK;
}

atlas_host_result atlas_host_wait(atlas_host *host, size_t max_polls) {
  if (!host || !host->active)
    return ATLAS_HOST_NOT_READY;
  for (size_t poll = 0; poll < max_polls; ++poll) {
    atlas_host_result result = atlas_host_poll(host);
    if (result != ATLAS_HOST_BUSY)
      return result;
  }
  return ATLAS_HOST_TIMEOUT;
}
