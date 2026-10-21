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

/* A monotonic platform clock: for example rdcycle on the host, or nanoseconds
 * in a native harness. Timings include callbacks and failed/timed-out attempts.
 * The owner records the clock unit and execution platform with measurements. */
typedef uint64_t (*atlas_host_read_clock)(void *context);
typedef struct {
  uint64_t load_ticks, start_ticks, wait_ticks;
} atlas_host_timings;

typedef enum {
  ATLAS_HOST_OK = 0,
  ATLAS_HOST_INVALID,
  ATLAS_HOST_BUSY,
  ATLAS_HOST_NOT_READY,
  ATLAS_HOST_TIMEOUT,
  ATLAS_HOST_BAD_HALT,
  ATLAS_HOST_CACHE_ERROR
} atlas_host_result;

/* Synchronous access to the program's device-visible DRAM ranges. */
typedef int (*atlas_tile_write)(void *context, uintptr_t address,
                                const uint8_t *source, size_t bytes);
typedef int (*atlas_tile_read)(void *context, uintptr_t address,
                               uint8_t *destination, size_t bytes);

typedef struct {
  atlas_tile_write write;
  atlas_tile_read read;
  void *context;
} atlas_tile_memory_io;


typedef struct {
  atlas_host_map map;
  atlas_host_mmio mmio;
  atlas_host_memory memory;
  const atlas_host_buffer *active_buffers;
  size_t active_buffer_count;
  int program_loaded;
  /* Immutable host copy, valid until invalidation; one context per device. */
  const uint32_t *resident_words;
  size_t resident_word_count;
  /* Cumulative counters; timings are measured by the platform's clock. */
  uint64_t image_loads, image_reuses, imem_words_written, launches;
  atlas_host_read_clock clock;
  void *clock_context;
  atlas_host_timings timings;
  int active;
  int launch_acknowledged;
  uint32_t start_retired;
} atlas_host;

/* These calls only manage a reset-entry Atlas instruction stream. The stream
 * must initialize its own registers, wait for its DMA writes, and ECALL. It
 * must retire at least one instruction and must not write CSR 0xC01. */
atlas_host_result atlas_host_init(atlas_host *host, atlas_host_map map,
                                  atlas_host_mmio mmio, atlas_host_memory memory);
/* Set or disable the clock while idle; reset cumulative timing totals. */
atlas_host_result atlas_host_set_clock(atlas_host *host,
                                      atlas_host_read_clock clock, void *context);
atlas_host_result atlas_host_load(atlas_host *host, const uint32_t *words,
                                  size_t word_count);
/* Retain a reset-entry image across invocations. The array must be immutable
 * and remain alive until replacement or invalidation. Equality compares all
 * words, so separately exported identical arrays can share residency. */
atlas_host_result atlas_host_ensure_loaded(atlas_host *host,
                                         const uint32_t *words, size_t word_count);
/* Call before external IMEM writes, while idle. A platform reset that has
 * drained DMA must instead be acknowledged with atlas_host_reset_completed. */
atlas_host_result atlas_host_invalidate_program(atlas_host *host);
atlas_host_result atlas_host_reset_completed(atlas_host *host);
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
