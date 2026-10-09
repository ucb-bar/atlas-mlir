#ifndef ATLAS_MXU_CONTRACT_VERIFICATION_H
#define ATLAS_MXU_CONTRACT_VERIFICATION_H

#include "Atlas/AtlasMXUAllocationVerification.h"
#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::atlas {
struct AtlasVerificationContext;

// Builds MXU command records in source order from live source, checked
// register allocation and MXU placements.
FailureOr<ArrayAttr> buildAtlasMXUContract(
    func::FuncOp function, llvm::ArrayRef<VirtualRegisterAssignment> registers,
    llvm::ArrayRef<VirtualMXUAssignment> mxuAssignments,
    const FixedResourcePlacement &fixed);

// Checks a generated artifact's MXU commands, versions and FP8 scales against
// its contract. See docs/dialect-reference.md.
LogicalResult verifyAtlasGeneratedMXUContract(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasGeneratedMXUContract(ModuleOp module);

} // namespace mlir::atlas

#endif
