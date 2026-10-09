#ifndef ATLAS_MXU_CONTRACT_VERIFICATION_H
#define ATLAS_MXU_CONTRACT_VERIFICATION_H

#include "Atlas/AtlasMXUAllocationVerification.h"
#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::atlas {
struct AtlasVerificationContext;

// Requires live verified source and independently checked register allocation.
// Tensor assignments are validated locally; MXU placement is checked again.
// Command ids follow source order, including all three legacy expansion steps.
FailureOr<ArrayAttr> buildAtlasMXUContract(
    func::FuncOp function, llvm::ArrayRef<VirtualRegisterAssignment> registers,
    llvm::ArrayRef<VirtualMXUAssignment> mxuAssignments,
    const FixedResourcePlacement &fixed);

// Checks exact command fields, logical versions and SELI-defined FP8 scales.
// SELD writes to an FP8 readout's scale register require separate completion
// evidence and are unsupported, even when later SELI instructions restore it.
// Source-order record ownership and emitted physical versions apply the shared
// MXUOwnership transitions to use counts and issued commands derived here,
// never to allocator decisions. Fresh producers are required on every
// reachable CFG path, including loops. Source/emitted CFG correspondence,
// tensor contents and physical engine completion remain separate obligations.
// Requires a generated artifact (AtlasGeneratedArtifact.h); its MXU contract
// may be empty only when the stream issues no MXU command.
LogicalResult verifyAtlasGeneratedMXUContract(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasGeneratedMXUContract(ModuleOp module);

} // namespace mlir::atlas

#endif
