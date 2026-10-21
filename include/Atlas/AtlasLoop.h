#ifndef ATLAS_LOOP_H
#define ATLAS_LOOP_H

#include "Atlas/AtlasTiming.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <vector>

// Structured single-block loops in a flat Atlas machine stream. Discovery
// depends only on control flow and supported instruction effects, never on
// virtual-operation IDs or helper annotations. Loops contain no DMA lifecycle
// operations or interior redirects. Unknown addresses or exits yield unknown
// access facts, requiring DMA completion before entry.

namespace mlir::atlas {
class BranchOp;
struct AtlasStream;

// VMEM bytes [begin, end).
struct AtlasVmemRange {
  uint64_t begin = 0, end = 0;
  bool empty() const { return begin >= end; }
  bool overlaps(const AtlasVmemRange &other) const {
    return !empty() && !other.empty() && begin < other.end && other.begin < end;
  }
  // Grows to cover `other` too.
  void cover(const AtlasVmemRange &other);
};

// What a run of instructions reads and writes in VMEM, each as one range
// covering every byte it touches.
struct AtlasVmemAccess {
  AtlasVmemRange reads, writes;
};

// Where a loop sits in the flat stream.
struct AtlasLoop {
  size_t begin = 0;    // the first instruction, which the back-edge targets
  size_t branch = 0;   // the back-edge; it or its delay slot ends the loop
  bool delaySlot = true;

  size_t end() const { return branch + (delaySlot ? 2 : 1); }
  bool contains(size_t pc) const { return begin <= pc && pc < end(); }
};

// What decoding proves about a loop (AtlasStream::factsOf): the registers it
// is entered with, the exit of the block that falls into it, and what it
// touches over all its iterations, run from them. `access` is unknown when
// that run cannot decide an address or the loop's exit.
struct AtlasLoopFacts {
  timing::RegValues entry = timing::unknownRegs();
  std::optional<AtlasVmemAccess> access;
};

// The instruction-word distance of a branch or JAL: ScalarCore applies the
// byte offset shifted right by one to its word PC. Other operations, JALR
// included, have none.
std::optional<int64_t> atlasRedirectDistance(Operation *op);

// Discover disjoint single-block conditional loops with no other redirect into
// the body, whose delay slot, unless `delaySlots` are pending, is body work
// too. Indirect control flow disables discovery. Unsupported bodies remain
// ordinary conservative CFG boundaries.
std::vector<AtlasLoop> findAtlasLoops(llvm::ArrayRef<Operation *> ops,
                                      bool delaySlots);

// The loop that holds `pc`, else null.
const AtlasLoop *findAtlasLoopAt(llvm::ArrayRef<AtlasLoop> loops, size_t pc);

// Whether `branch` is taken when its operands hold `regs`; unknown when an
// operand is.
std::optional<bool> atlasBranchTaken(BranchOp branch, const timing::RegValues &regs);

enum class AtlasLoopExit { Exited, UnknownBranch, AnalysisLimit };
// Analysis budget, not a semantic trip-count restriction. Loops that exceed
// it retain an unknown footprint and cannot carry pending DMA.
constexpr unsigned kAtlasLoopIterationBound = 256;

// Runs `loop` over `ops`, one iteration after another: `step(pc)` executes an
// instruction and updates the registers `regs()` returns, from which the
// back-edge is decided as it issues. Fails only when a step fails.
FailureOr<AtlasLoopExit>
runAtlasLoop(llvm::ArrayRef<Operation *> ops, const AtlasLoop &loop,
             llvm::function_ref<const timing::RegValues &()> regs,
             llvm::function_ref<LogicalResult(size_t pc)> step);

// What `loop` touches in `stream` over all its iterations, run from `entry`;
// unknown when an address or the exit is.
std::optional<AtlasVmemAccess>
proveAtlasLoopAccess(const AtlasStream &stream, const AtlasLoop &loop,
                     const timing::RegValues &entry);

} // namespace mlir::atlas

#endif
