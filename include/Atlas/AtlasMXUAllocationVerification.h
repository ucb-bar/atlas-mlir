#ifndef ATLAS_MXU_ALLOCATION_VERIFICATION_H
#define ATLAS_MXU_ALLOCATION_VERIFICATION_H

#include "Atlas/AtlasVirtualAllocation.h"

namespace mlir::atlas {

struct VirtualMXUAssignment {
  Value handle;
  MXUPlacement placement;
};

// Requires live source IR with verified SSA and typed virtual MXU operations.
// Admission and state-flow verification remain separate. Independently
// checks complete placements, unit/slot geometry, block-local logical ownership,
// and in-place accumulator versions from source uses. Last virtual weight use
// and accumulator readout release logical ownership only: this does not prove
// physical engine read/write completion or correspondence with emitted commands.
LogicalResult verifyAtlasMXUAllocation(
    func::FuncOp function, llvm::ArrayRef<VirtualMXUAssignment> assignments,
    const FixedResourcePlacement &fixed);

} // namespace mlir::atlas

#endif
