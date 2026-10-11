#ifndef ATLAS_MXU_ALLOCATION_VERIFICATION_H
#define ATLAS_MXU_ALLOCATION_VERIFICATION_H

#include "Atlas/AtlasVirtualAllocation.h"

namespace mlir::atlas {

struct VirtualMXUAssignment {
  Value handle;
  MXUPlacement placement;
};

// Checks MXU placements against live, SSA-verified source: completeness,
// geometry and logical ownership from source-derived use counts.
// See docs/dialect-reference.md.
LogicalResult verifyAtlasMXUAllocation(
    func::FuncOp function, llvm::ArrayRef<VirtualMXUAssignment> assignments,
    const FixedResourcePlacement &fixed);

} // namespace mlir::atlas

#endif
