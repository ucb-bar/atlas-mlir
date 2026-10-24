#include "atlas_host.h"

#include <limits.h>

enum { ATLAS_HALTED = 1, ATLAS_HALT_REASON_SHIFT = 1, ATLAS_ECALL = 2 };

/* Compiler atomics may emit only FENCE rw,rw on RV64. Atlas MMIO needs the
 * device I/O domains as well as host memory ordered at publication/consumption. */
static void atlas_host_fence(void) {
#if defined(__riscv)
  __asm__ __volatile__("fence iorw, iorw" ::: "memory");
#else
  __sync_synchronize();
#endif
}

uint32_t atlas_host_direct_read32(void *context, uintptr_t address) {
  (void)context;
  atlas_host_fence();
  return *(volatile const uint32_t *)address;
}

void atlas_host_direct_write32(void *context, uintptr_t address, uint32_t word) {
  (void)context;
  *(volatile uint32_t *)address = word;
  atlas_host_fence();
}

atlas_host_result atlas_host_init(atlas_host *host, atlas_host_map map, atlas_host_mmio mmio,
                                  atlas_host_memory memory) {
  if (!host || !mmio.read32 || !mmio.write32 || (map.imem_base & 3) || (map.csr_base & 3) ||
      map.imem_capacity_bytes < 4 || (map.imem_capacity_bytes & 3) ||
      map.imem_capacity_bytes > UINTPTR_MAX - map.imem_base || map.csr_base > UINTPTR_MAX - ATLAS_CSR_EXEC_CONTROL ||
      (!memory.coherent && !memory.sync))
    return ATLAS_HOST_INVALID;
  *host = (atlas_host){0};
  host->map = map;
  host->mmio = mmio;
  host->memory = memory;
  return ATLAS_HOST_OK;
}

atlas_host_result atlas_host_load(atlas_host *host, const uint32_t *words, size_t word_count) {
  if (!host)
    return ATLAS_HOST_INVALID;
  if (host->active)
    return ATLAS_HOST_BUSY;
  /* A failed replacement must not leave the previous program launchable. */
  host->program_loaded = 0;
  host->raw_completion = 0;
  if (!words || !word_count || word_count > host->map.imem_capacity_bytes / sizeof(uint32_t))
    return ATLAS_HOST_INVALID;
  if (!(host->mmio.read32(host->mmio.context, host->map.csr_base + ATLAS_CSR_STATUS) & ATLAS_HALTED))
    return ATLAS_HOST_NOT_READY;
  for (size_t index = 0; index < word_count; ++index)
    host->mmio.write32(host->mmio.context, host->map.imem_base + 4 * index, words[index]);
  host->program_loaded = 1;
  return ATLAS_HOST_OK;
}

/* No allocation or tensor wrapper: retain original PCs and extend the terminal
 * epilogue in place. These standard CSR encodings match Atlas scalar decode. */
atlas_host_result atlas_host_load_raw(atlas_host *host, const uint32_t *words, size_t word_count) {
  static const uint32_t epilogue[] = {
      0x00000013u,              /* NOP: also separates a preceding DELAY from CSR read. */
      0xc1102ff3u,              /* CSRRS x31, DBG1, x0 */
      0x00000013u, 0xc10f9073u, /* CSRRW x0, DBG0, x31 */
      0x00000013u, 0x00000073u,
  };
  if (!host)
    return ATLAS_HOST_INVALID;
  if (host->active)
    return ATLAS_HOST_BUSY;
  host->program_loaded = 0;
  host->raw_completion = 0;
  size_t capacity = host->map.imem_capacity_bytes / sizeof(uint32_t);
  if (!words || !word_count || capacity < 6 || word_count > capacity - 5 || words[word_count - 1] != 0x00000073u)
    return ATLAS_HOST_INVALID;
  for (size_t index = 0; index < word_count - 1; ++index) {
    uint32_t word = words[index];
    if (word == 0x00000073u || word == 0x00100073u)
      return ATLAS_HOST_INVALID;
    if ((word & 0x7fu) == 0x73u) {
      unsigned csr = word >> 20, kind = (word >> 12) & 7u;
      unsigned source = (word >> 15) & 31u;
      int writes = kind == 1 || kind == 5 || (kind && source);
      if ((csr == 0xc10u || csr == 0xc11u) && writes)
        return ATLAS_HOST_INVALID;
    }
  }
  if (!(host->mmio.read32(host->mmio.context, host->map.csr_base + ATLAS_CSR_STATUS) & ATLAS_HALTED))
    return ATLAS_HOST_NOT_READY;
  for (size_t index = 0; index < word_count - 1; ++index)
    host->mmio.write32(host->mmio.context, host->map.imem_base + 4 * index, words[index]);
  for (size_t index = 0; index < sizeof epilogue / sizeof epilogue[0]; ++index)
    host->mmio.write32(host->mmio.context, host->map.imem_base + 4 * (word_count - 1 + index), epilogue[index]);
  host->program_loaded = 1;
  host->raw_completion = 1;
  return ATLAS_HOST_OK;
}

atlas_host_result atlas_host_start(atlas_host *host, const atlas_host_buffer *buffers, size_t buffer_count) {
  if (!host || (buffer_count && !buffers))
    return ATLAS_HOST_INVALID;
  if (host->active)
    return ATLAS_HOST_BUSY;
  if (!host->program_loaded)
    return ATLAS_HOST_NOT_READY;
  if (!(host->mmio.read32(host->mmio.context, host->map.csr_base + ATLAS_CSR_STATUS) & ATLAS_HALTED))
    return ATLAS_HOST_NOT_READY;
  for (size_t index = 0; index < buffer_count; ++index) {
    const atlas_host_buffer *buffer = &buffers[index];
    uintptr_t limit = host->raw_completion ? UINTPTR_MAX : UINT32_MAX;
    if (!buffer->size_bytes || buffer->address > limit || buffer->size_bytes - 1 > limit - buffer->address ||
        buffer->access < ATLAS_HOST_DEVICE_READ || buffer->access > ATLAS_HOST_DEVICE_READ_WRITE)
      return ATLAS_HOST_INVALID;
  }
  if (host->raw_completion && host->invocation_sequence == UINT32_MAX)
    return ATLAS_HOST_NOT_READY;
  if (!host->memory.coherent) {
    for (size_t index = 0; index < buffer_count; ++index)
      if (host->memory.sync(host->memory.context, &buffers[index], 1))
        return ATLAS_HOST_CACHE_ERROR;
  }
  host->active_buffers = buffers;
  host->active_buffer_count = buffer_count;
  host->active = 1;
  host->launch_acknowledged = 0;
  if (host->raw_completion) {
    ++host->invocation_sequence;
    host->mmio.write32(host->mmio.context, host->map.csr_base + ATLAS_CSR_DBG0, 0);
    host->mmio.write32(host->mmio.context, host->map.csr_base + ATLAS_CSR_DBG1, host->invocation_sequence);
  } else {
    /* Atlas is halted, so this count holds until the start write takes effect. */
    host->start_retired = host->mmio.read32(host->mmio.context, host->map.csr_base + ATLAS_CSR_RETIRED);
  }
  atlas_host_fence();
  host->mmio.write32(host->mmio.context, host->map.csr_base + ATLAS_CSR_EXEC_CONTROL, 1);
  return ATLAS_HOST_OK;
}

atlas_host_result atlas_host_poll(atlas_host *host) {
  if (!host || !host->active)
    return ATLAS_HOST_NOT_READY;
  /* Marker first: a status sample preceding publication could be the old
   * invocation's ECALL while the new invocation is still in its epilogue. */
  uint32_t marker =
      host->raw_completion ? host->mmio.read32(host->mmio.context, host->map.csr_base + ATLAS_CSR_DBG0) : 0;
  uint32_t status = host->mmio.read32(host->mmio.context, host->map.csr_base + ATLAS_CSR_STATUS);
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
   * Raw streams publish their invocation marker before ECALL. Legacy streams
   * acknowledge retirement, then consume the following status read. */
  if (host->raw_completion) {
    if (marker != host->invocation_sequence) {
      /* A terminal fault is diagnosable after start was observed; absent
       * that evidence an old halt remains indistinguishable from a new one. */
      if (!host->launch_acknowledged || reason == ATLAS_ECALL)
        return ATLAS_HOST_BUSY;
    }
  } else if (!host->launch_acknowledged) {
    host->launch_acknowledged =
        (host->mmio.read32(host->mmio.context, host->map.csr_base + ATLAS_CSR_RETIRED) != host->start_retired);
    return ATLAS_HOST_BUSY;
  }
  if (reason != ATLAS_ECALL) {
    /* A scalar fault is not a drain: raw ownership remains retained. */
    if (host->raw_completion)
      return ATLAS_HOST_BAD_HALT;
    host->active = 0;
    host->launch_acknowledged = 0;
    host->active_buffers = 0;
    host->active_buffer_count = 0;
    return ATLAS_HOST_BAD_HALT;
  }
  atlas_host_fence();
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
