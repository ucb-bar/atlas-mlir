#ifndef ATLAS_BUFFER_CONTRACT_VERIFICATION_H
#define ATLAS_BUFFER_CONTRACT_VERIFICATION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::atlas {
struct AtlasVerificationContext;
// Derives which source writer's words every VMEM reader must observe from the
// CFG and tile contracts, and each PACK's scale code from live source.
FailureOr<DictionaryAttr> buildAtlasBufferContract(
    func::FuncOp function, DictionaryAttr cfgContract, ArrayAttr tileContract);

// Checks along issued paths that each reader observes those words, executing
// PACK's counted copy loop. Word origins are static command ids, and loop
// headers keep only first-entry words, so a raw conversion store is current
// only because the CFG and tile checks force every PACK visit to issue its own
// VSTORE, copy and VLOAD. The ModuleOp entry relies on the earlier tile
// tag/address binding, DMA-memory pending-range exclusion with block-local
// intervals, the structural rule against scalar stores while a DMA is pending,
// and the timing check for the asynchronous LW. See docs/dialect-reference.md.
LogicalResult verifyAtlasGeneratedBufferContract(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasGeneratedBufferContract(ModuleOp module);
} // namespace mlir::atlas
#endif
