#ifndef ATLAS_STREAM_H
#define ATLAS_STREAM_H

#include "Atlas/AtlasTiming.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::atlas {

// A flat Atlas stream without delays, in basic blocks. A block ending in a
// branch or jump ends with its delay slot.
struct AtlasStream {
  llvm::SmallVector<Operation *> ops;
  std::vector<timing::Instr> instrs;
  llvm::DenseMap<Operation *, Operation *> targetOf; // nullptr: stream end
  std::vector<size_t> starts;
  std::vector<llvm::SmallVector<size_t, 2>> succs;
  std::vector<timing::RegValues> entry;

  size_t blockEnd(size_t block) const;
  bool endsInBranch(size_t block) const;
  bool endsInHalt(size_t block) const;
  bool fallsThrough(size_t block) const;
};

struct DelayInsertion {
  std::vector<uint32_t> delays;
  bool guard = false; // end on a NOP: a halt does not wait for a delay
  std::string reason;
  const char *guardReason = "a halt does not wait for a delay";
};

// `allowDelays` admits explicit DELAYs, as in a final stream being verified.
FailureOr<AtlasStream> readAtlasStream(ModuleOp module,
                                       bool allowDelays = false);
FailureOr<timing::Instr> atlasInstruction(Operation *op);
// Rejects illegal delay slots and DMA hazards that no delay can cover. With a
// selected target, DMA must also be waited within its block.
LogicalResult checkAtlasStream(const AtlasStream &stream,
                               const timing::TargetTiming &target = {});
// Selected RTL timing verifies each basic block from an idle engine state, so
// every block must be reachable and end in ECALL, in a branch or JAL whose
// target is an instruction and whose delay slot is ADDI or LUI, or by
// falling into the next block.
LogicalResult checkSelectedControlFlow(ModuleOp module,
                                       const AtlasStream &stream);
// Rewrites the module as the ops in `order`, each after its insertion, and
// re-aims branches at the new first op of their target block. `order` keeps
// every block's ops at that block's positions.
LogicalResult writeAtlasStream(ModuleOp module, const AtlasStream &stream,
                               llvm::ArrayRef<size_t> order,
                               llvm::ArrayRef<DelayInsertion> before);
std::vector<uint32_t> idleDelays(int idle);
bool isNop(Operation *op);

} // namespace mlir::atlas

#endif
