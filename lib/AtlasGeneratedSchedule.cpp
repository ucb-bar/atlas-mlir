#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::atlas;

LogicalResult mlir::atlas::verifyAtlasGeneratedSchedule(ModuleOp module) {
  if (!module->hasAttr("atlas.generated_from_virtual"))
    return module.emitOpError("expected an Atlas virtual-to-machine artifact");
  for (Operation &op : module.getBody()->getOperations()) {
    if (isa<StartOp>(op))
      continue;
    Operation *next = op.getNextNode();
    if (isa<VLoadOp, VStoreOp, VPUUnaryOp, VPUBinaryOp, VPUPackOp,
            MXUPushOp, MXUMatmulOp, MXUPopOp>(op)) {
      auto wait = next ? dyn_cast<DelayOp>(next) : DelayOp{};
      if (!wait || wait.getCycles() < 256 ||
          !wait->getAttrOfType<StringAttr>("atlas.delay_reason"))
        return op.emitOpError(
            "generated async instruction requires an annotated DELAY >= 256");
    }
    if (auto dma = dyn_cast<DMAOp>(op)) {
      auto wait = next ? dyn_cast<DMAWaitOp>(next) : DMAWaitOp{};
      if (!wait || wait.getChannel() != dma.getChannel())
        return op.emitOpError(
            "generated DMA requires immediate same-channel DMA.WAIT");
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
    return "Check generated serial waits and selected-core branch slots";
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
