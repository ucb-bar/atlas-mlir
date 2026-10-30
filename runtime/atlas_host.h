#ifndef ATLAS_HOST_H
#define ATLAS_HOST_H

#include "atlas_host_layout.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Host-visible addresses; supply these from the selected SoC build. */
typedef struct {
  uintptr_t imem_base;
  uintptr_t csr_base;
  size_t imem_capacity_bytes;
} atlas_host_map;

typedef uint32_t (*atlas_host_read32)(void *context, uintptr_t address);
typedef void (*atlas_host_write32)(void *context, uintptr_t address, uint32_t word);

typedef struct {
  atlas_host_read32 read32;
  atlas_host_write32 write32;
  void *context;
} atlas_host_mmio;

typedef enum { ATLAS_HOST_DEVICE_READ, ATLAS_HOST_DEVICE_WRITE, ATLAS_HOST_DEVICE_READ_WRITE } atlas_host_access;

typedef struct {
  uintptr_t address;
  size_t size_bytes;
  atlas_host_access access;
} atlas_host_buffer;

/* The platform owns cache maintenance. For a coherent system, set coherent=1.
 * Otherwise each hook must complete its work before it returns. */
typedef int (*atlas_host_cache_sync)(void *context, const atlas_host_buffer *buffer, int before_launch);
typedef struct {
  int coherent;
  atlas_host_cache_sync sync;
  void *context;
} atlas_host_memory;

typedef enum {
  ATLAS_HOST_OK = 0,
  ATLAS_HOST_INVALID,
  ATLAS_HOST_BUSY,
  ATLAS_HOST_NOT_READY,
  ATLAS_HOST_TIMEOUT,
  ATLAS_HOST_BAD_HALT,
  ATLAS_HOST_CACHE_ERROR
} atlas_host_result;

typedef struct {
  atlas_host_map map;
  atlas_host_mmio mmio;
  atlas_host_memory memory;
  const atlas_host_buffer *active_buffers;
  size_t active_buffer_count;
  int program_loaded;
  int active;
  int launch_acknowledged;
  uint32_t start_retired;
  int raw_completion;
  uint32_t invocation_sequence;
} atlas_host;

/* These calls only manage a reset-entry Atlas instruction stream. The stream
 * must initialize its own registers, wait for its DMA writes, and ECALL. It
 * must retire at least one instruction and must not write CSR 0xC01. */
atlas_host_result atlas_host_init(atlas_host *host, atlas_host_map map, atlas_host_mmio mmio, atlas_host_memory memory);
atlas_host_result atlas_host_load(atlas_host *host, const uint32_t *words, size_t word_count);
/* Raw reset-entry streams have no graph shape restriction. The original stream
 * must end at its only ECALL, drain every engine before that ECALL, and never
 * write DBG0/DBG1. The loader replaces the final ECALL with a mailbox epilogue;
 * x31 is clobbered only after original work finishes. Capacity needs five extra
 * instruction words. All control-flow paths must reach that terminal epilogue.
 * DBG1 is host-owned during execution; DBG0 is device-owned. The caller supplies
 * known quiescent hardware before init: STOP alone does not establish this.
 *
 * Raw starts accept complete uintptr_t buffer ranges. This does not relocate
 * instructions: their DMA high-half/base and low32 ranges must match buffers. */
atlas_host_result atlas_host_load_raw(atlas_host *host, const uint32_t *words, size_t word_count);

atlas_host_result atlas_host_start(atlas_host *host, const atlas_host_buffer *buffers, size_t buffer_count);
/* Descriptors remain alive and unchanged until successful completion. Timeout,
 * post-launch cache failure and raw bad halt retain ownership; fault recovery requires
 * externally established quiescence followed by init. Raw launches
 * use nonwrapping invocation sequences; exhaustion requires known quiescence
 * and reinitialization. A marker is sampled before status, so successful
 * completion never depends on observing a running pulse. */
atlas_host_result atlas_host_poll(atlas_host *host);
atlas_host_result atlas_host_wait(atlas_host *host, size_t max_polls);

/* MMIO callbacks for a bare-metal host with directly mapped Atlas registers. */
uint32_t atlas_host_direct_read32(void *context, uintptr_t address);
void atlas_host_direct_write32(void *context, uintptr_t address, uint32_t word);

#ifdef __cplusplus
}
#endif
#endif
