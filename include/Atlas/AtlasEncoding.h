#ifndef ATLAS_ENCODING_H
#define ATLAS_ENCODING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir::atlas {

// One selected-RTL 32-bit word for a verified machine operation.
FailureOr<uint32_t> encodeMachineWord(Operation *op);

// Validates a flat ordered Atlas stream and collects all words atomically.
// llvmBlock additionally requires statically in-block direct branch targets and
// rejects unresolved JALR, since one inline assembly block has no external Atlas
// program/ABI to resolve such jumps.
LogicalResult collectAtlasWords(ModuleOp module,
                                llvm::SmallVectorImpl<uint32_t> &words,
                                bool llvmBlock);

} // namespace mlir::atlas

#endif
