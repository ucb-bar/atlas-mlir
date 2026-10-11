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

// Runs issued paths and PACK's counted copy loop to check readers see those
// words by static command id. CFG and tile checks keep raw PACK stores current;
// the ModuleOp entry presumes the tile, DMA-memory, structural, timing checks.
LogicalResult verifyAtlasGeneratedBufferContract(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasGeneratedBufferContract(ModuleOp module);
} // namespace mlir::atlas
#endif
