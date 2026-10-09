#ifndef ATLAS_DMA_MEMORY_VERIFICATION_H
#define ATLAS_DMA_MEMORY_VERIFICATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::atlas {
struct AtlasVerificationContext;

// Recomputes captured DMA and vector memory ranges from the emitted stream.
// This does not use the scheduler's dependency graph or memory footprints.
// The generated schedule verifier establishes launch/wait and control policy.
// DRAM disjointness is qualified only for the bounded zero-upper ABI; the
// module does not establish a negotiated TileLink address width.
LogicalResult verifyAtlasGeneratedDMAMemory(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasGeneratedDMAMemory(ModuleOp module);

} // namespace mlir::atlas

#endif
