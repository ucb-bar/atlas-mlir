#ifndef ATLAS_DMA_ALLOCATION_VERIFICATION_H
#define ATLAS_DMA_ALLOCATION_VERIFICATION_H

#include "Atlas/AtlasVirtualAllocation.h"

namespace mlir::atlas {

struct VirtualDMAAssignment {
  Value transfer;
  DMATransferPlacement placement;
};

// Checks DMA placements against live, SSA-verified source: completeness, unique
// transfer ids, geometry and block-local channel/window ownership.
// See docs/dialect-reference.md.
LogicalResult verifyAtlasDMAAllocation(
    func::FuncOp function, llvm::ArrayRef<VirtualDMAAssignment> assignments);

} // namespace mlir::atlas

#endif
