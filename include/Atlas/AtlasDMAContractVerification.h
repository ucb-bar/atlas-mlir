#ifndef ATLAS_DMA_CONTRACT_VERIFICATION_H
#define ATLAS_DMA_CONTRACT_VERIFICATION_H

#include "Atlas/AtlasDMAAllocationVerification.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::atlas {
struct AtlasVerificationContext;

// Builds explicit-transfer records, sorted by transfer id, from live source and
// independently validated placements.
FailureOr<ArrayAttr> buildAtlasDMAContract(
    func::FuncOp function, llvm::ArrayRef<VirtualDMAAssignment> assignments);

// Checks a generated artifact's captured tagged DMA operands against its
// contract, after the structural stage. See docs/dialect-reference.md.
LogicalResult verifyAtlasGeneratedDMAContract(const AtlasVerificationContext &ctx);

// Checks pending DMA ranges against VMEM/DRAM accesses recomputed from the
// emitted stream (zero-upper DRAM ABI only). See docs/dialect-reference.md.
LogicalResult verifyAtlasGeneratedDMAMemory(const AtlasVerificationContext &ctx);

} // namespace mlir::atlas

#endif
