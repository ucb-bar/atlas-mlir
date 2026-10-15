#ifndef ATLAS_HOST_H
#define ATLAS_HOST_H

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

typedef enum {
  ATLAS_HOST_DEVICE_READ,
  ATLAS_HOST_DEVICE_WRITE,
  ATLAS_HOST_DEVICE_READ_WRITE
} atlas_host_access;

typedef struct {
  uintptr_t address;
  size_t size_bytes;
  atlas_host_access access;
} atlas_host_buffer;

/* The platform owns cache maintenance. For a coherent system, set coherent=1.
 * Otherwise each hook must complete its work before it returns. */
typedef int (*atlas_host_cache_sync)(void *context, const atlas_host_buffer *buffer,
                                      int before_launch);
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
} atlas_host;

/* These calls only manage a reset-entry Atlas instruction stream. The stream
 * must initialize its own registers, wait for its DMA writes, and ECALL. */
atlas_host_result atlas_host_init(atlas_host *host, atlas_host_map map,
                                  atlas_host_mmio mmio, atlas_host_memory memory);
atlas_host_result atlas_host_load(atlas_host *host, const uint32_t *words,
                                  size_t word_count);
atlas_host_result atlas_host_start(atlas_host *host,
                                   const atlas_host_buffer *buffers,
                                   size_t buffer_count);
atlas_host_result atlas_host_poll(atlas_host *host);
atlas_host_result atlas_host_wait(atlas_host *host, size_t max_polls);

/* MMIO callbacks for a bare-metal host with directly mapped Atlas registers. */
uint32_t atlas_host_direct_read32(void *context, uintptr_t address);
void atlas_host_direct_write32(void *context, uintptr_t address, uint32_t word);

#ifdef __cplusplus
}
#endif
#endif
