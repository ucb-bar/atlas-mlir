#include "Atlas/AtlasVerificationContext.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::atlas;

static LogicalResult
verifyGeneratedStructure(ModuleOp module,
                         llvm::SmallVectorImpl<AtlasDMAInterval> &intervals) {
  llvm::DenseMap<unsigned, AtlasDMAInterval> pending; // waitPC not yet known
  llvm::DenseSet<int32_t> launchIDs;
  int64_t pc = 0;
  for (Operation &op : module.getBody()->getOperations()) {
    std::optional<int32_t> id;
    if (Attribute marker = op.getAttr(kAtlasTagDMATransfer)) {
      auto integer = dyn_cast<IntegerAttr>(marker);
      if (!isa<DMAOp, DMAWaitOp>(op) || !integer ||
          !integer.getType().isSignlessInteger(32) ||
          integer.getValue().isNegative())
        return op.emitOpError(
            "atlas.virtual_dma_transfer requires a nonnegative i32 on DMA or DMA.WAIT");
      id = static_cast<int32_t>(integer.getValue().getSExtValue());
    }
    if (isa<StartOp>(op)) {
      if (!pending.empty())
        return op.emitOpError("unexpected instruction while DMA is pending");
      continue;
    }
    int64_t currentPC = pc++;
    Operation *next = op.getNextNode();
    if (auto dma = dyn_cast<DMAOp>(op)) {
      if (pending.count(dma.getChannel()))
        return op.emitOpError("another DMA launch while its channel is pending");
      if (id && !launchIDs.insert(*id).second)
        return op.emitOpError("duplicate virtual DMA transfer launch ID");
      pending.try_emplace(dma.getChannel(), AtlasDMAInterval{dma.getChannel(), id, currentPC, -1, dma});
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
    } else if (!pending.empty() && !canOverlapAtlasGeneratedDMA(&op)) {
      return op.emitOpError("unexpected instruction while DMA is pending");
    }
    if (isa<BranchOp, JumpOp>(op)) {
      auto slot = next ? dyn_cast<ALUImmOp>(next) : ALUImmOp{};
      if (!slot || slot.getKind() != "addi" || slot.getDst() != 0 ||
          slot.getSrc() != 0 || slot.getImmediate() != 0)
        return op.emitOpError(
            "generated redirect requires a nonredirecting x0 NOP delay slot");
    }
  }
  if (!pending.empty())
    return pending.begin()->second.launch.emitOpError("generated DMA has no matching DMA.WAIT");
  if (!intervals.empty()) {
    pc = 0;
    for (Operation &op : module.getBody()->getOperations()) {
      if (isa<StartOp>(op))
        continue;
      int64_t currentPC = pc++;
      std::optional<int64_t> target;
      // Selected Atlas redirects encode twice the instruction-word distance.
      if (auto branch = dyn_cast<BranchOp>(op))
        target = currentPC +
                 branch.getOffsetBytesAttr().getValue().getSExtValue() / 2;
      else if (auto jump = dyn_cast<JumpOp>(op)) {
        if (jump.getKind() == "jalr")
          return op.emitOpError(
              "generated explicit DMA intervals forbid unproven JALR targets");
        target = currentPC + jump.getOffsetAttr().getValue().getSExtValue() / 2;
      }
      if (target)
        for (const AtlasDMAInterval &interval : intervals)
          if (*target > interval.launchPC && *target <= interval.waitPC)
            return op.emitOpError(
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
  if (generated && failed(verifyGeneratedStructure(module, ctx.dmaIntervals)))
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
