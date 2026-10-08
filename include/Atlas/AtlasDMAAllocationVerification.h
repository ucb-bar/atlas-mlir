#ifndef ATLAS_DMA_ALLOCATION_VERIFICATION_H
#define ATLAS_DMA_ALLOCATION_VERIFICATION_H

#include "Atlas/AtlasVirtualAllocation.h"

namespace mlir::atlas {

struct VirtualDMAAssignment {
  Value transfer;
  DMATransferPlacement placement;
};

// Requires live source IR with verified SSA and typed virtual DMA operations.
// Admission and state-flow verification remain separate. Independently
// checks assignment completeness, unique nonnegative i32 transfer ids, geometry,
// and logical channel/window ownership
// from each launch through its matching block-local completion. Does not prove
// helper live-range preservation, emitted correspondence, or completion of the
// post-await VLOAD reads that copy staging into tensor registers.
LogicalResult verifyAtlasDMAAllocation(
    func::FuncOp function, llvm::ArrayRef<VirtualDMAAssignment> assignments);

} // namespace mlir::atlas

#endif
