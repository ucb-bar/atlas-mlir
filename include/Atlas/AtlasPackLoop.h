#ifndef ATLAS_PACK_LOOP_H
#define ATLAS_PACK_LOOP_H

#include "Atlas/AtlasTiming.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <vector>

// PACK relayout loops in a flat Atlas stream.
//
// lowerPack (AtlasVirtualToMachine.cpp) repairs the selected PACK row layout
// with a bounded scalar copy loop: one block closed by its own back-edge, each
// instruction tagged atlas.virtual_cfg_helper = "pack" and with the PACK's
// atlas.virtual_cfg_source (timing passes tag their delays with the helper
// only). The loop launches and completes no DMA and has no other entry, so the
// explicit transfers in flight when it starts stay in flight through every
// iteration and when it exits. It is the only loop a DMA interval may hold
// (canOverlapAtlasGeneratedDMA); the DMA memory and timing checks compare the
// VMEM it touches with each pending transfer.

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
struct AtlasPackLoop {
  size_t begin = 0;    // the first instruction, which the back-edge targets
  size_t branch = 0;   // the back-edge; its delay slot ends the loop
  int32_t source = -1; // the PACK's atlas.virtual_cfg_source

  size_t end() const { return branch + 2; }
  bool contains(size_t pc) const { return begin <= pc && pc < end(); }
};

// What decoding proves about a loop (AtlasStream::factsOf): the registers it
// is entered with, the exit of the block that falls into it, and what it
// touches over all its iterations, run from them. `access` is unknown when
// that run cannot decide an address or the loop's exit.
struct AtlasPackLoopFacts {
  timing::RegValues entry = timing::unknownRegs();
  std::optional<AtlasVmemAccess> access;
};

// The instruction-word distance of a branch or JAL: ScalarCore applies the
// byte offset shifted right by one to its word PC. Other operations, JALR
// included, have none.
std::optional<int64_t> atlasRedirectDistance(Operation *op);

// Every PACK relayout loop in `ops` (program order, atlas.start excluded), in
// order. A loop's back-edge is a "pack" helper branch to an earlier
// instruction, with the exact NOP as its delay slot; everything from its
// target to the branch is that PACK's scalar loads, stores and ALU work, or a
// delay; and no other redirect lands in it. A stream with a JALR, whose target
// is unknown, has none.
std::vector<AtlasPackLoop> findAtlasPackLoops(llvm::ArrayRef<Operation *> ops);

// The loop that holds `pc`, else null.
const AtlasPackLoop *findAtlasPackLoopAt(llvm::ArrayRef<AtlasPackLoop> loops, size_t pc);

// Whether `branch` is taken when its operands hold `regs`; unknown when an
// operand is.
std::optional<bool> atlasBranchTaken(BranchOp branch, const timing::RegValues &regs);

enum class AtlasPackLoopExit { Exited, UnknownBranch, Unbounded };
// A relayout takes 32 iterations; a run gives up after this many.
constexpr unsigned kAtlasPackLoopIterationBound = 256;

// Runs `loop` over `ops`, one iteration after another: `step(pc)` executes an
// instruction and updates the registers `regs()` returns, from which the
// back-edge is decided as it issues. Fails only when a step fails.
FailureOr<AtlasPackLoopExit>
runAtlasPackLoop(llvm::ArrayRef<Operation *> ops, const AtlasPackLoop &loop,
                 llvm::function_ref<const timing::RegValues &()> regs,
                 llvm::function_ref<LogicalResult(size_t pc)> step);

// Contexts for exactly 32 decided iterations. Any unknown/unaligned/wrapping
// scalar address or unsupported memory operation leaves every context absent.
std::vector<timing::TimingContext> proveAtlasPackLoopTimingContexts(
    const AtlasStream &stream, const AtlasPackLoop &loop,
    const timing::RegValues &entry);

// What `loop` touches in `stream` over all its iterations, run from `entry`;
// unknown when an address or the exit is.
std::optional<AtlasVmemAccess> proveAtlasPackLoopAccess(const AtlasStream &stream,
                                                        const AtlasPackLoop &loop,
                                                        const timing::RegValues &entry);

} // namespace mlir::atlas

#endif
