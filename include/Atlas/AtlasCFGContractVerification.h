#ifndef ATLAS_CFG_CONTRACT_VERIFICATION_H
#define ATLAS_CFG_CONTRACT_VERIFICATION_H

#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::atlas {
struct AtlasVerificationContext;
// Builds the CFG/SSA correspondence contract from live source and scalar/tensor
// register placements, never from planned words.
FailureOr<DictionaryAttr> buildAtlasCFGContract(
    func::FuncOp function, llvm::ArrayRef<VirtualRegisterAssignment> registers);

// Checks a generated artifact's emitted paths against its CFG contract.
// See docs/dialect-reference.md.
LogicalResult verifyAtlasGeneratedCFGContract(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasGeneratedCFGContract(ModuleOp module);
} // namespace mlir::atlas
#endif
