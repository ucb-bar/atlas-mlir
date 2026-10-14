#ifndef ATLAS_DMA_ALLOCATION_VERIFICATION_H
#define ATLAS_DMA_ALLOCATION_VERIFICATION_H

#include "Atlas/AtlasVirtualAllocation.h"

namespace mlir::atlas {

// `transfer` is the state result of the DMA operation the placement serves.
struct VirtualDMAAssignment {
  Value transfer;
  DMATransferPlacement placement;
};

// Checks DMA placements against live, SSA-verified source: completeness, unique
// transfer ids and geometry. Channel and staging reuse is legal; DMA wait
// insertion completes the earlier transfer first.
// See docs/dialect-reference.md.
LogicalResult verifyAtlasDMAAllocation(
    func::FuncOp function, llvm::ArrayRef<VirtualDMAAssignment> assignments);

} // namespace mlir::atlas

#endif
