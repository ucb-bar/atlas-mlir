#ifndef ATLAS_SOURCE_MEMORY_EFFECT_CONTRACT_H
#define ATLAS_SOURCE_MEMORY_EFFECT_CONTRACT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::atlas {
struct AtlasVerificationContext;
// Derives every source DRAM transfer, explicit and implicit, with its proven
// byte span from the live source and the independently built CFG and tile
// contracts. Overlapping effects of one source block visit become ordered
// predecessors whenever either writes. No planned or issued order is read.
FailureOr<DictionaryAttr> buildAtlasSourceMemoryEffectContract(
    func::FuncOp function, DictionaryAttr cfgContract, ArrayAttr tileContract);

// Requires separately checked CFG correspondence, tile geometry and DMA
// lifecycle. Proves along issued paths that each predecessor's WAIT precedes
// the successor's launch within the same source visit, that a visit drains at
// its source edges and exits, and leaves read/read overlap and disjoint spans
// unordered. WAIT is semantic completion; timing providers prove physical
// completion. Tensor numerics and buffer preservation remain separate.
// Requires a generated artifact (AtlasGeneratedArtifact.h), which always
// carries this contract.
LogicalResult verifyAtlasGeneratedSourceMemoryEffectContract(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasGeneratedSourceMemoryEffectContract(ModuleOp module);
} // namespace mlir::atlas
#endif
