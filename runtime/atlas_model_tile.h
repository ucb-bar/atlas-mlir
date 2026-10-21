#ifndef ATLAS_MODEL_TILE_H
#define ATLAS_MODEL_TILE_H

#include "atlas_host.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { ATLAS_MODEL_TILE_EDGE = 32, ATLAS_MODEL_TILE_FP8_BYTES = 1024,
       ATLAS_MODEL_TILE_BF16_BYTES = 2048, ATLAS_MODEL_TILE_SCRATCH_BYTES = 8192 };

/* MLIR's ranked two-dimensional f32 memref descriptor on a 64-bit host. */
typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
} atlas_f32_memref2;

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
} atlas_f32_memref1;

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t sizes[3];
  int64_t strides[3];
} atlas_f32_memref3;

typedef int (*atlas_tile_pack)(const float *source, uint8_t *destination);
typedef int (*atlas_tile_unpack)(const volatile uint8_t *source,
                                 float *destination);

typedef struct {
  atlas_host *host;
  atlas_tile_memory_io io;
  const uint32_t *program_words;
  size_t program_word_count;
  uintptr_t input_addresses[2];
  uintptr_t output_address;
  const uint8_t *constant_bytes;
  size_t constant_bytes_length;
  /* 0/1 checks that operand against the frozen bytes; 2 means both operands
   * are runtime values. This checks target-program equivalence, not original
   * f32 model equivalence. */
  unsigned constant_operand;
  /* Transpose a source memref before packing it into target row-major order. */
  unsigned transpose_input[2];
  atlas_tile_pack pack[2];
  atlas_tile_unpack unpack;
  size_t max_polls;
  void *scratch;
  size_t scratch_bytes;
  /* atlas_host retains this array through an outstanding timed-out launch. */
  atlas_host_buffer active_buffers[3];
} atlas_model_tile;

typedef enum {
  ATLAS_MODEL_TILE_OK = 0,
  ATLAS_MODEL_TILE_INVALID,
  ATLAS_MODEL_TILE_INPUT_DOMAIN,
  ATLAS_MODEL_TILE_CONSTANT_MISMATCH,
  ATLAS_MODEL_TILE_MEMORY_ERROR,
  ATLAS_MODEL_TILE_DEVICE_ERROR,
  ATLAS_MODEL_TILE_ALLOC_ERROR
} atlas_model_tile_status;

/* The MLIR C-interface callback returns void, so it cannot return a tile
 * failure to its caller. Platforms may define this non-returning hook to
 * record the status and halt. The runtime's weak default traps. Call the
 * checked tile function directly when an in-process status is required. */
__attribute__((noreturn)) void atlas_model_tile_fail(
    atlas_model_tile_status status, atlas_host_result device_result);

/* Synchronous one-tile dispatch. On failure, `result` is unchanged. A device
 * timeout leaves host->active set; the owner must finish or reset that launch
 * before reusing the binding or its scratch and buffer descriptors. */
atlas_model_tile_status atlas_model_tile_run(atlas_model_tile *binding,
    const atlas_f32_memref2 *lhs, const atlas_f32_memref2 *rhs,
    atlas_f32_memref2 *result, atlas_host_result *device_result);

/* One source matmul contained in a physical tile. The K values occupy the
 * rightmost physical columns, matching the selected ordered-reduction tile.
 * The remaining cells are positive zero. On success, the caller owns the
 * newly allocated M-by-N result; on failure its descriptor is unchanged. */
atlas_model_tile_status atlas_model_tile_run_window(atlas_model_tile *binding,
    const atlas_f32_memref2 *lhs, const atlas_f32_memref2 *rhs,
    atlas_f32_memref2 *result, size_t m, size_t n, size_t k,
    atlas_host_result *device_result);

/* Direct mapped DRAM callbacks for a qualified bare-metal platform. */
int atlas_model_tile_direct_write(void *context, uintptr_t address,
                                  const uint8_t *source, size_t bytes);
int atlas_model_tile_direct_read(void *context, uintptr_t address,
                                 uint8_t *destination, size_t bytes);

#ifdef __cplusplus
}
#endif
#endif
