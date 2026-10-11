#ifndef ATLAS_SOURCE_MEMORY_EFFECT_CONTRACT_H
#define ATLAS_SOURCE_MEMORY_EFFECT_CONTRACT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::atlas {
struct AtlasVerificationContext;
// Derives source DRAM effects and their write-ordering predecessors from live
// source and the CFG and tile contracts, never from planned or issued order.
FailureOr<DictionaryAttr> buildAtlasSourceMemoryEffectContract(
    func::FuncOp function, DictionaryAttr cfgContract, ArrayAttr tileContract);

// Checks along issued paths that each predecessor's WAIT precedes its
// successor's launch within a source visit. See docs/dialect-reference.md.
LogicalResult verifyAtlasGeneratedSourceMemoryEffectContract(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasGeneratedSourceMemoryEffectContract(ModuleOp module);
} // namespace mlir::atlas
#endif
