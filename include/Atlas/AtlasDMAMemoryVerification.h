#ifndef ATLAS_DMA_MEMORY_VERIFICATION_H
#define ATLAS_DMA_MEMORY_VERIFICATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::atlas {
struct AtlasVerificationContext;

// Checks pending DMA ranges against VMEM/DRAM accesses recomputed from the
// emitted stream (zero-upper DRAM ABI only). See docs/dialect-reference.md.
LogicalResult verifyAtlasGeneratedDMAMemory(const AtlasVerificationContext &ctx);

} // namespace mlir::atlas

#endif
