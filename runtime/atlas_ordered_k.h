#ifndef ATLAS_ORDERED_K_H
#define ATLAS_ORDERED_K_H

#include "atlas_host.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Raw FP8 MXU0 input windows, in alternating activation/weight order. */
typedef struct {
  unsigned source_index;
  size_t rows;
  size_t source_columns;
  size_t k_start;
  size_t tile_columns;
  uintptr_t address;
} atlas_ordered_k_window;

typedef int (*atlas_ordered_k_pack)(
    const uint8_t *source, size_t source_bytes,
    uint8_t *destination, size_t destination_bytes,
    size_t rows, size_t source_columns, size_t k_start,
    size_t tile_columns, size_t element_bytes);
typedef int (*atlas_ordered_k_write)(void *context, uintptr_t address,
                                     const uint8_t *source, size_t bytes);
typedef int (*atlas_ordered_k_read)(void *context, uintptr_t address,
                                    uint8_t *destination, size_t bytes);

typedef struct {
  atlas_host *host;
  void *memory_context;
  atlas_ordered_k_write write;
  atlas_ordered_k_read read;
  atlas_ordered_k_pack pack;
  const uint32_t *program_words;
  size_t program_word_count;
  const atlas_ordered_k_window *windows;
  size_t window_count;
  uintptr_t output_address;
  size_t max_polls;
  uint8_t *scratch;
  size_t scratch_bytes;
  /* Retained by atlas_host while a timed-out launch is active. */
  atlas_host_buffer *active_buffers;
  size_t active_buffer_capacity;
} atlas_ordered_k_tile;

typedef enum {
  ATLAS_ORDERED_K_OK = 0,
  ATLAS_ORDERED_K_INVALID,
  ATLAS_ORDERED_K_INPUT_DOMAIN,
  ATLAS_ORDERED_K_MEMORY_ERROR,
  ATLAS_ORDERED_K_DEVICE_ERROR,
  ATLAS_ORDERED_K_CONSTANT_MISMATCH
} atlas_ordered_k_status;

/* Diagnostic raw-E4M3 admission used by this launcher. This is a runtime
 * subset of the compiler's named input domain, not a hardware capability
 * verdict for codes outside it. */
int atlas_ordered_k_admit_code(uint8_t code);

/* Launch one verified reset-entry program with one or more K windows.
 * The logical FP8 sources are row-major A[32,K] and W[32,K]. Output is the
 * physical 2 KiB BF16 readout;
 * the caller owns its logical column-layout conversion. On failure output
 * remains unchanged. A short first window is left-padded with positive zero;
 * later windows contain the remaining source columns in order. This diagnostic
 * runtime admits positive zero and signed
 * E4M3 codes with exponent fields 1 through 14. A timeout retains
 * host->active and active_buffers. */
atlas_ordered_k_status atlas_ordered_k_run(
    atlas_ordered_k_tile *tile,
    const uint8_t *activation, size_t activation_bytes,
    const uint8_t *weight, size_t weight_bytes,
    uint8_t *physical_output, size_t physical_output_bytes,
    atlas_host_result *device_result);

#ifdef __cplusplus
}
#endif
#endif
