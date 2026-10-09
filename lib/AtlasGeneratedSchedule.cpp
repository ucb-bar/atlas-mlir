#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasDMAContractVerification.h"
#include "Atlas/AtlasDMAMemoryVerification.h"
#include "Atlas/AtlasMXUContractVerification.h"
#include "Atlas/AtlasTileContractVerification.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::atlas;

bool mlir::atlas::canOverlapAtlasGeneratedDMA(Operation *op) {
  if (isa<DelayOp, MXUPushOp, MXUMatmulOp, MXUPopOp, VPUUnaryOp,
          VPUBinaryOp, VLoadOp, VStoreOp>(op))
    return true;
  // ScalarCore captures DMA operands at launch; later scalar writes are safe.
  if (isa<ALURegOp, ALUImmOp, UpperOp>(op))
    return true;
  if (auto load = dyn_cast<ScalarLoadOp>(op))
    return load.getKind() == "seli";
  return false;
}

LogicalResult mlir::atlas::verifyAtlasGeneratedSchedule(ModuleOp module) {
  if (failed(verifyAtlasTimingState(module)))
    return failure();
  if (!module->hasAttr("atlas.generated_from_virtual"))
    return module.emitOpError("expected an Atlas virtual-to-machine artifact");
  struct PendingDMA {
    DMAOp launch;
    std::optional<int32_t> id;
    int64_t pc;
  };
  std::optional<PendingDMA> pending;
  llvm::DenseSet<int32_t> launchIDs;
  llvm::SmallVector<std::pair<int64_t, int64_t>> intervals;
  int64_t pc = 0;
  for (Operation &op : module.getBody()->getOperations()) {
    std::optional<int32_t> id;
    if (Attribute marker = op.getAttr("atlas.virtual_dma_transfer")) {
      auto integer = dyn_cast<IntegerAttr>(marker);
      if (!isa<DMAOp, DMAWaitOp>(op) || !integer ||
          !integer.getType().isSignlessInteger(32) ||
          integer.getValue().isNegative())
        return op.emitOpError(
            "atlas.virtual_dma_transfer requires a nonnegative i32 on DMA or DMA.WAIT");
      id = static_cast<int32_t>(integer.getValue().getSExtValue());
    }
    if (isa<StartOp>(op)) {
      if (pending)
        return op.emitOpError("unexpected instruction while DMA is pending");
      continue;
    }
    int64_t currentPC = pc++;
    Operation *next = op.getNextNode();
    if (auto dma = dyn_cast<DMAOp>(op)) {
      if (pending)
        return op.emitOpError("another DMA launch while DMA is pending");
      if (id) {
        if (!launchIDs.insert(*id).second)
          return op.emitOpError("duplicate virtual DMA transfer launch ID");
      } else if (!module->hasAttr("atlas.timing_state")) {
        auto wait = next ? dyn_cast<DMAWaitOp>(next) : DMAWaitOp{};
        if (!wait || wait.getChannel() != dma.getChannel())
          return op.emitOpError(
              "generated DMA requires immediate same-channel DMA.WAIT");
      }
      pending = PendingDMA{dma, id, currentPC};
    } else if (auto wait = dyn_cast<DMAWaitOp>(op)) {
      if (!pending)
        return op.emitOpError("DMA.WAIT has no pending DMA transfer");
      if (wait.getChannel() != pending->launch.getChannel() ||
          id != pending->id)
        return op.emitOpError(
            "DMA.WAIT must match the pending DMA channel and transfer ID");
      intervals.emplace_back(pending->pc, currentPC);
      pending.reset();
    } else if (pending) {
      if (!canOverlapAtlasGeneratedDMA(&op))
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
  if (pending)
    return pending->launch.emitOpError("generated DMA has no matching DMA.WAIT");
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
        for (auto [launchPC, waitPC] : intervals)
          if (*target > launchPC && *target <= waitPC)
            return op.emitOpError(
                "generated redirect target enters a pending DMA interval");
    }
  }
  if (failed(verifyAtlasGeneratedDMAMemory(module)))
    return failure();
  if (failed(verifyAtlasGeneratedDMAContract(module)))
    return failure();
  if (failed(verifyAtlasGeneratedMXUContract(module)))
    return failure();
  if (failed(verifyAtlasGeneratedTileContract(module)))
    return failure();
  auto state = module->getAttrOfType<StringAttr>("atlas.timing_state");
  if (state && state.getValue() == "timed")
    return verifyAtlasTiming(module);
  return success();
}

namespace {
struct VerifyAtlasGeneratedSchedulePass
    : PassWrapper<VerifyAtlasGeneratedSchedulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAtlasGeneratedSchedulePass)
  StringRef getArgument() const final {
    return "verify-atlas-generated-schedule";
  }
  StringRef getDescription() const final {
    return "Check generated resources, correspondence, branch slots and retained timing proofs";
  }
  void runOnOperation() override {
    if (failed(verifyAtlasGeneratedSchedule(getOperation())))
      signalPassFailure();
  }
};
} // namespace

void mlir::atlas::registerVerifyAtlasGeneratedSchedulePass() {
  PassRegistration<VerifyAtlasGeneratedSchedulePass>();
}
