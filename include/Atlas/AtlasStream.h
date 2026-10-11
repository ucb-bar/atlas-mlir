#ifndef ATLAS_STREAM_H
#define ATLAS_STREAM_H

#include "Atlas/AtlasTimingProvider.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <deque>

namespace mlir::atlas {
struct AtlasVerificationContext;

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
// Verification reads emitted delays, treats PC-dependent scalars as unknown and
// only encodes; Scheduling first verifies the artifact. `words` receives the
// encoded stream.
FailureOr<AtlasStream> readAtlasStream(
    ModuleOp module, AtlasStreamReadMode mode = AtlasStreamReadMode::Scheduling,
    llvm::SmallVectorImpl<uint32_t> *words = nullptr);
// Number of readAtlasStream calls in this process, for decode-once tests.
unsigned atlasStreamDecodeCount();
// Rejects illegal delay slots and DMA hazards using complete supplied rules.
LogicalResult checkAtlasStream(
    const AtlasStream &stream, const timing::TimingProvider &provider);
// Checks actual issue spacing, reservations and drained CFG boundaries without
// consulting a scheduler graph or implicitly using any model timing policy.
LogicalResult verifyAtlasTimedStream(
    const AtlasStream &stream, const timing::TimingProvider &provider);
LogicalResult verifyAtlasTiming(const AtlasVerificationContext &ctx,
                                const timing::TimingProvider &provider);
LogicalResult verifyAtlasTiming(
    ModuleOp module, const timing::TimingProvider &provider);
// The provider a timing pass uses: `requested`, else the retained
// atlas.timing_provider, else the npu-model provider. A request that differs
// from the retained one, or an unregistered or incomplete provider, fails.
FailureOr<timing::TimingProvider>
selectAtlasTimingProvider(ModuleOp module, llvm::StringRef requested = {});
// Dispatches the explicitly retained complete provider. Legacy streams without
// timing metadata select the named model for compatibility.
LogicalResult verifyAtlasTiming(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasTiming(ModuleOp module);
// Validates explicit timing metadata. Legacy streams without a timing state
// keep their existing scope; requireTimed rejects explicitly untimed streams.
LogicalResult verifyAtlasTimingState(ModuleOp module, bool requireTimed = false);
void registerVerifyAtlasTimingPass();
// DMA_CONFIG updates one shared upper address word, irrespective of channel.
// Entry values are known only when all reachable incoming CFG edges agree.
std::vector<std::optional<uint32_t>>
atlasDMAUpperWordEntries(const AtlasStream &stream);
// ERF entry facts (e0 included, all unknown at entry): SELI defines a raw code,
// SELD invalidates it, and joins must agree. They do not prove SELD completion.
std::vector<timing::RegValues>
atlasScaleRegisterEntries(const AtlasStream &stream);
void applyAtlasScaleRegister(const timing::Instr &in, timing::RegValues &regs);
// Whether a block leaves the stream: no successor, a redirect to the stream
// end, or a branch whose fall-through is the stream end.
bool atlasBlockExits(const AtlasStream &stream, size_t block);

template <typename S> struct AtlasForwardEntries {
  std::vector<S> entries;
  std::vector<bool> reached;
};
// Forward worklist from block 0: `transfer` turns a copy of a block's entry
// state into its exit state (failure aborts), `join(into, incoming)` returns
// whether a reached entry changed, and `successors` may replace the edges.
template <typename S, typename Transfer, typename Join, typename Successors>
FailureOr<AtlasForwardEntries<S>>
atlasForwardEntries(const AtlasStream &stream, const S &init, Transfer transfer, Join join, Successors successors) {
  AtlasForwardEntries<S> result{std::vector<S>(stream.starts.size(), init), std::vector<bool>(stream.starts.size(), false)};
  if (stream.starts.empty())
    return result;
  result.reached[0] = true;
  std::deque<size_t> work = {0};
  while (!work.empty()) {
    size_t block = work.front();
    work.pop_front();
    S state = result.entries[block];
    if (failed(transfer(block, state)))
      return failure();
    for (size_t next : successors(block)) {
      if (!result.reached[next]) {
        result.reached[next] = true;
        result.entries[next] = state;
        work.push_back(next);
      } else if (join(result.entries[next], state)) {
        work.push_back(next);
      }
    }
  }
  return result;
}
template <typename S, typename Transfer, typename Join>
FailureOr<AtlasForwardEntries<S>>
atlasForwardEntries(const AtlasStream &stream, const S &init, Transfer transfer, Join join) {
  return atlasForwardEntries(stream, init, transfer, join, [&](size_t block) -> llvm::ArrayRef<size_t> { return stream.succs[block]; });
}
// Rewrites the module as the ops in `order`, each after its insertion, and
// re-aims branches at the new first op of their target block. `order` keeps
// every block's ops at that block's positions. `after`, empty or one entry per
// block, holds the delays that end a block falling through or ending the
// stream: they drain its work, belong to it, and a branch into the next block
// does not run them. A nonempty `timedBy` stamps the result timed by that
// atlas.timing_provider, else it is untimed. The module is then re-verified,
// and `stream` no longer describes it.
LogicalResult writeAtlasStream(ModuleOp module, const AtlasStream &stream,
                               llvm::ArrayRef<size_t> order,
                               llvm::ArrayRef<DelayInsertion> before,
                               llvm::ArrayRef<DelayInsertion> after = {},
                               llvm::StringRef timedBy = {});
std::vector<uint32_t> idleDelays(int idle);
bool isNop(Operation *op);

} // namespace mlir::atlas

#endif
