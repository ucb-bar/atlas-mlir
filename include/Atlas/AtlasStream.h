#ifndef ATLAS_STREAM_H
#define ATLAS_STREAM_H

#include "Atlas/AtlasTiming.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::atlas {

// A flat Atlas stream in basic blocks. A block ending in a
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
};

enum class AtlasStreamReadMode { Scheduling, Verification };
// Verification reads emitted delays and treats PC-dependent scalar values as
// unknown. It skips only the generated verifier hook to avoid recursive checks.
FailureOr<AtlasStream> readAtlasStream(
    ModuleOp module, AtlasStreamReadMode mode = AtlasStreamReadMode::Scheduling);
// Rejects illegal delay slots and DMA hazards that no delay can cover.
LogicalResult checkAtlasStream(
    const AtlasStream &stream, const timing::FootprintResolver &resolver = {});
// Checks actual issue spacing, reservations and drained CFG boundaries without
// using a scheduler dependency graph. Footprints are supplied; dependence,
// capacities, VPU overlap and reservation policy remain the named npu-model
// rtl-match model. This is not a complete provider-independent RTL verifier.
LogicalResult verifyAtlasTimedStream(
    const AtlasStream &stream, const timing::FootprintResolver &resolver);
LogicalResult verifyAtlasTiming(
    ModuleOp module, const timing::FootprintResolver &resolver);
// Validates explicit timing metadata. Legacy streams without a timing state
// keep their existing scope; requireTimed rejects explicitly untimed streams.
LogicalResult verifyAtlasTimingState(ModuleOp module, bool requireTimed = false);
void registerVerifyAtlasTimingPass();
// DMA_CONFIG updates one shared upper address word, irrespective of channel.
// Entry values are known only when all reachable incoming CFG edges agree.
std::vector<std::optional<uint32_t>>
atlasDMAUpperWordEntries(const AtlasStream &stream);
// Instruction-order ERF facts: e0 is writable and starts unknown too. SELI
// defines a raw code, SELD invalidates it, and reachable joins must agree.
// These facts alone do not establish completion of an asynchronous SELD.
std::vector<timing::RegValues>
atlasScaleRegisterEntries(const AtlasStream &stream);
void applyAtlasScaleRegister(const timing::Instr &in, timing::RegValues &regs);
// Rewrites the module as the ops in `order`, each after its insertion, and
// re-aims branches at the new first op of their target block. `order` keeps
// every block's ops at that block's positions. Rewriting alone leaves the
// stream untimed; timing passes explicitly request a checked timed artifact.
LogicalResult writeAtlasStream(ModuleOp module, const AtlasStream &stream,
                               llvm::ArrayRef<size_t> order,
                               llvm::ArrayRef<DelayInsertion> before,
                               bool timed = false);
std::vector<uint32_t> idleDelays(int idle);
bool isNop(Operation *op);

} // namespace mlir::atlas

#endif
