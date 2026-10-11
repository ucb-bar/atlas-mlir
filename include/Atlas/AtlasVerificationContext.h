#ifndef ATLAS_VERIFICATION_CONTEXT_H
#define ATLAS_VERIFICATION_CONTEXT_H

#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir::atlas {

// Facts shared by every check at one verification boundary. The context holds
// raw operations and is stale after any module mutation; build a new one.
struct AtlasVerificationContext {
  ModuleOp module;
  bool generated = false;
  llvm::SmallVector<Operation *> ops; // program order, atlas.start excluded
  llvm::DenseMap<Operation *, size_t> pcOf;
  bool hasDMA = false, hasResourceTags = false, hasCFGContract = false, timed = false;
  llvm::SmallVector<uint32_t> words; // the decode's encoding, else empty
  std::optional<AtlasStream> stream;
  std::vector<std::optional<uint32_t>> dmaUpperEntry; // present with stream
  std::vector<timing::RegValues> scaleEntry;          // present with stream

  // The union of the generated checkers' and the timing check's decode gates.
  // Untimed non-generated streams are never decoded, so they may keep JALR.
  bool needsDecodedStream() const {
    return timed || (generated && (hasDMA || hasResourceTags || hasCFGContract));
  }

  AtlasVerificationContext() = default;
  AtlasVerificationContext(AtlasVerificationContext &&) = default;
  AtlasVerificationContext &operator=(AtlasVerificationContext &&) = default;
  AtlasVerificationContext(const AtlasVerificationContext &) = delete;
  AtlasVerificationContext &operator=(const AtlasVerificationContext &) = delete;
};

// After the MLIR verifier and verifyAtlasTimingState, decodes (so encodes) at
// most once. `generated` first runs the structural stage (DMA pairing,
// protected intervals, x0-NOP redirect slots); requireStream forces decoding.
FailureOr<AtlasVerificationContext>
buildAtlasVerificationContext(ModuleOp module, bool generated,
                              bool requireStream = false);

// The exact `addi x0, x0, 0` that generated redirect delay slots require.
inline bool isExactAtlasNop(Operation *op) {
  auto nop = dyn_cast_or_null<ALUImmOp>(op);
  return nop && nop.getKind() == "addi" && nop.getDst() == 0 &&
         nop.getSrc() == 0 && nop.getImmediate() == 0;
}

} // namespace mlir::atlas

#endif
