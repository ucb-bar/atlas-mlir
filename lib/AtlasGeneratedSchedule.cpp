#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasDMAMemoryVerification.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::atlas;

LogicalResult mlir::atlas::verifyAtlasGeneratedSchedule(ModuleOp module) {
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
      } else {
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
      if (pending->id)
        intervals.emplace_back(pending->pc, currentPC);
      pending.reset();
    } else if (pending) {
      std::optional<unsigned> scalarDst;
      bool allowed = isa<DelayOp, MXUPushOp, MXUMatmulOp, MXUPopOp, VPUUnaryOp, VPUBinaryOp, VLoadOp, VStoreOp>(op);
      if (auto alu = dyn_cast<ALURegOp>(op))
        scalarDst = alu.getDst();
      else if (auto alu = dyn_cast<ALUImmOp>(op))
        scalarDst = alu.getDst();
      else if (auto upper = dyn_cast<UpperOp>(op))
        scalarDst = upper.getDst();
      else if (auto load = dyn_cast<ScalarLoadOp>(op))
        allowed = load.getKind() == "seli";
      if (scalarDst) {
        // ScalarCore forms every DMA operand at launch; DMA queues the command.
        // Reusing those scalar registers does not change the captured transfer.
        allowed = true;
      }
      if (!allowed)
        return op.emitOpError("unexpected instruction while DMA is pending");
    }
    if (isa<VLoadOp, VStoreOp, VPUUnaryOp, VPUBinaryOp, VPUPackOp,
            MXUPushOp, MXUMatmulOp, MXUPopOp>(op)) {
      auto wait = next ? dyn_cast<DelayOp>(next) : DelayOp{};
      if (!wait || wait.getCycles() < 256 ||
          !wait->getAttrOfType<StringAttr>("atlas.delay_reason"))
        return op.emitOpError(
            "generated async instruction requires an annotated DELAY >= 256");
    }
    if (auto load = dyn_cast<ScalarLoadOp>(op)) {
      auto wait = next ? dyn_cast<DelayOp>(next) : DelayOp{};
      if (load.getKind() == "lw" &&
          (!wait || wait.getCycles() < 8 ||
           wait->getAttrOfType<StringAttr>("atlas.delay_reason") !=
               StringAttr::get(module.getContext(), "scalar_load_completion")))
        return op.emitOpError(
            "generated scalar LW requires an annotated DELAY >= 8");
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
  return verifyAtlasGeneratedDMAMemory(module);
}

namespace {
struct VerifyAtlasGeneratedSchedulePass
    : PassWrapper<VerifyAtlasGeneratedSchedulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAtlasGeneratedSchedulePass)
  StringRef getArgument() const final {
    return "verify-atlas-generated-schedule";
  }
  StringRef getDescription() const final {
    return "Check generated DMA lifetimes, async waits, and selected-core branch slots";
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
