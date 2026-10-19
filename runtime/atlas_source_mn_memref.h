#ifndef ATLAS_SOURCE_MN_MEMREF_H
#define ATLAS_SOURCE_MN_MEMREF_H

#include "atlas_model_tile.h"
#include "atlas_source_mn.h"

/* MLIR C-interface adapter for one source-bound matmul region. The selected
 * tile, conversion functions, and geometry come from a verified native grid.
 * Each call reads strided f32 inputs and publishes a dense owned result only
 * after every Atlas launch succeeds. The caller frees result->allocated. */
typedef struct {
  atlas_ordered_k_tile *tile;
  atlas_source_mn_encode encode_lhs;
  atlas_source_mn_encode encode_rhs;
  const void *lhs_policy;
  const void *rhs_policy;
  size_t batches;
  size_t m;
  size_t n;
  size_t k;
  /* Optional exact source bits for a declared frozen RHS. A mismatch fails
   * before any device launch, including when a model passes changed weights. */
  const uint32_t *frozen_rhs_bits;
  size_t frozen_rhs_elements;
} atlas_source_mn_memref_binding;

atlas_ordered_k_status atlas_source_mn_run_memref2(
    const atlas_source_mn_memref_binding *binding,
    const atlas_f32_memref2 *lhs, const atlas_f32_memref2 *rhs,
    atlas_f32_memref2 *result, atlas_host_result *device_result);

atlas_ordered_k_status atlas_source_mn_run_memref3(
    const atlas_source_mn_memref_binding *binding,
    const atlas_f32_memref3 *lhs, const atlas_f32_memref3 *rhs,
    atlas_f32_memref3 *result, atlas_host_result *device_result);

#endif
