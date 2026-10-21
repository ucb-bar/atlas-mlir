#include "Atlas/AtlasVerificationContext.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "llvm/ADT/DenseMap.h"
#include <set>

using namespace mlir;
using namespace mlir::atlas;

static LogicalResult verifyGeneratedStructure(ModuleOp module) {
  // One generated launch/wait pair. Intervals on different channels may overlap.
  struct AtlasDMAInterval {
    std::optional<int32_t> id;
    int64_t launchPC, waitPC;
    DMAOp launch;
  };
  llvm::SmallVector<AtlasDMAInterval> intervals;
  llvm::DenseMap<unsigned, AtlasDMAInterval> pending; // waitPC not yet known
  std::set<int32_t> launchIDs; // any tag; INT32_MAX is DenseMap's empty key
  llvm::SmallVector<Operation *> ops;
  for (Operation &op : module.getBody()->getOperations())
    if (!isa<StartOp>(op))
      ops.push_back(&op);
  // The only loops an interval may hold (canOverlapAtlasGeneratedDMA).
  bool delaySlots = atlasHasDelaySlots(module);
  std::vector<AtlasLoop> loops = findAtlasLoops(ops, delaySlots);
  int64_t pc = 0;
  for (Operation &op : module.getBody()->getOperations()) {
    std::optional<int32_t> id;
    if (op.hasAttr(kAtlasTagDMATransfer)) {
      id = contractTag(&op, kAtlasTagDMATransfer);
      if (!isa<DMAOp, DMAWaitOp>(op) || !id)
        return op.emitOpError(
            "atlas.virtual_dma_transfer requires a nonnegative i32 on DMA or DMA.WAIT");
    }
    if (isa<StartOp>(op)) {
      if (!pending.empty())
        return op.emitOpError("unexpected instruction while DMA is pending");
      continue;
    }
    int64_t currentPC = pc++;
    if (auto dma = dyn_cast<DMAOp>(op)) {
      if (pending.count(dma.getChannel()))
        return op.emitOpError("another DMA launch while its channel is pending");
      if (id && !launchIDs.insert(*id).second)
        return op.emitOpError("duplicate virtual DMA transfer launch ID");
      pending.try_emplace(dma.getChannel(), AtlasDMAInterval{id, currentPC, -1, dma});
    } else if (auto wait = dyn_cast<DMAWaitOp>(op)) {
      auto transfer = pending.find(wait.getChannel());
      if (transfer == pending.end())
        return op.emitOpError("DMA.WAIT has no pending DMA transfer");
      if (id != transfer->second.id)
        return op.emitOpError(
            "DMA.WAIT must match the pending DMA channel and transfer ID");
      transfer->second.waitPC = currentPC;
      intervals.push_back(transfer->second);
      pending.erase(transfer);
    } else if (!pending.empty() && !canOverlapAtlasGeneratedDMA(ops, loops, currentPC)) {
      return op.emitOpError("unexpected instruction while DMA is pending");
    }
    if (delaySlots && isa<BranchOp, JumpOp>(op) &&
        !isAtlasDelaySlotWork(op.getNextNode()))
      return op.emitOpError(
          "generated redirect requires a scalar ALU delay slot");
  }
  if (!pending.empty())
    return pending.begin()->second.launch.emitOpError("generated DMA has no matching DMA.WAIT");
  if (!intervals.empty()) {
    for (auto [currentPC, op] : llvm::enumerate(ops)) {
      if (auto jump = dyn_cast<JumpOp>(op); jump && jump.getKind() == "jalr")
        return op->emitOpError(
            "generated explicit DMA intervals forbid unproven JALR targets");
      std::optional<int64_t> distance = atlasRedirectDistance(op);
      // An analyzed loop's back-edge stays inside every interval that holds the
      // loop: intervals start and end at DMA instructions, which no loop holds.
      const AtlasLoop *loop = findAtlasLoopAt(loops, currentPC);
      if (!distance || (loop && loop->branch == currentPC))
        continue;
      int64_t target = int64_t(currentPC) + *distance;
      for (const AtlasDMAInterval &interval : intervals)
        if (target > interval.launchPC && target <= interval.waitPC)
          return op->emitOpError(
              "generated redirect target enters a pending DMA interval");
    }
  }
  return success();
}

FailureOr<AtlasVerificationContext>
mlir::atlas::buildAtlasVerificationContext(ModuleOp module, bool generated,
                                           bool requireStream) {
  AtlasVerificationContext ctx;
  ctx.module = module;
  ctx.generated = generated;
  for (Operation &op : module.getBody()->getOperations()) {
    if (isa<StartOp>(op))
      continue;
    ctx.pcOf[&op] = ctx.ops.size();
    ctx.ops.push_back(&op);
    ctx.hasDMA |= isa<DMAOp>(op);
    ctx.hasResourceTags |= op.hasAttr(kAtlasTagDMATransfer) || op.hasAttr(kAtlasTagMXUCommand) || op.hasAttr(kAtlasTagTileCommand);
  }
  ctx.hasCFGContract = module->hasAttr(kAtlasCFGContract);
  auto state = module->getAttrOfType<StringAttr>(kAtlasTimingState);
  ctx.timed = state && state.getValue() == "timed";
  // The structural stage precedes decoding, which rejects every JALR.
  if (generated && failed(verifyGeneratedStructure(module)))
    return failure();
  if (!requireStream && !ctx.needsDecodedStream())
    return ctx;
  auto stream = readAtlasStream(module, AtlasStreamReadMode::Verification, &ctx.words);
  if (failed(stream))
    return failure();
  ctx.stream = std::move(*stream);
  ctx.dmaUpperEntry = atlasDMAUpperWordEntries(*ctx.stream);
  ctx.scaleEntry = atlasScaleRegisterEntries(*ctx.stream);
  return ctx;
}
