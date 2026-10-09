#ifndef ATLAS_CFG_CONTRACT_VERIFICATION_H
#define ATLAS_CFG_CONTRACT_VERIFICATION_H

#include "Atlas/AtlasRegisterAllocationVerification.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::atlas {
// Stable identities follow source block, argument and operation order. The
// builder reads live admitted SSA and placement claims, never planned words.
// Registers include scalar and tensor values; other resource handles are omitted.
FailureOr<DictionaryAttr> buildAtlasCFGContract(
    func::FuncOp function, llvm::ArrayRef<VirtualRegisterAssignment> registers);

// Checks source block visits, conditional polarity/targets, scalar definitions,
// tensor origins, simultaneous edge copies and source operation execution.
// Scalar expressions collapse to checked source identities at definitions, so
// the proof is inductive across source loops rather than bounded unrolling.
// PACK's marked internal helper is opaque here: its memory effects, iteration
// count and finite termination require the separate helper/buffer obligation.
// Numerical tensor semantics and physical completion remain separate checks.
// Resource-contract-v3 requires CFG metadata; older artifacts without it retain
// their existing verification scope.
LogicalResult verifyAtlasGeneratedCFGContract(ModuleOp module);
} // namespace mlir::atlas
#endif
