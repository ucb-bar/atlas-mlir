#ifndef ATLAS_ENCODING_H
#define ATLAS_ENCODING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir::atlas {

// One selected-RTL 32-bit word for a verified machine operation.
FailureOr<uint32_t> encodeMachineWord(Operation *op);

// Validates the flat state chain and delay-slot adjacency, then appends every
// word atomically. It performs no artifact checks.
LogicalResult encodeAtlasWords(ModuleOp module,
                               llvm::SmallVectorImpl<uint32_t> &words);
// Number of encodeAtlasWords calls in this process, for decode-once tests.
unsigned atlasWordEncodeCount();

// Verification boundary: the MLIR verifier, timing-state check and, on one
// AtlasVerificationContext, the generated or timed checks; appends atomically.
// llvmBlock additionally requires statically in-block direct branch targets and
// rejects unresolved JALR, since one inline assembly block has no external Atlas
// program/ABI to resolve such jumps.
LogicalResult verifyAtlasArtifact(ModuleOp module, bool llvmBlock,
                                  llvm::SmallVectorImpl<uint32_t> &words);

} // namespace mlir::atlas

#endif
