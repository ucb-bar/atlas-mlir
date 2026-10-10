#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasBufferContractVerification.h"
#include "Atlas/AtlasDMAContractVerification.h"
#include "Atlas/AtlasDMAMemoryVerification.h"
#include "Atlas/AtlasMXUContractVerification.h"
#include "Atlas/AtlasTileContractVerification.h"
#include "Atlas/AtlasCFGContractVerification.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasSourceMemoryEffectContract.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasVerificationContext.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::atlas;

bool mlir::atlas::canOverlapAtlasGeneratedDMA(Operation *op) {
  // ScalarCore captures DMA operands at launch; later scalar writes are safe.
  if (isa<DelayOp, MXUPushOp, MXUMatmulOp, MXUPopOp, VPUUnaryOp, VPUBinaryOp,
          VLoadOp, VStoreOp, ALURegOp, ALUImmOp, UpperOp>(op))
    return true;
  if (auto load = dyn_cast<ScalarLoadOp>(op))
    return load.getKind() == "seli";
  return false;
}

LogicalResult mlir::atlas::verifyAtlasGeneratedSchedule(
    const AtlasVerificationContext &ctx) {
  assert(ctx.generated && "generated checks require the structural stage");
  ModuleOp module = ctx.module;
  if (failed(verifyAtlasTimingState(module)) || failed(requireAtlasGeneratedArtifact(module)) ||
      failed(verifyAtlasGeneratedDMAMemory(ctx)) || failed(verifyAtlasGeneratedDMAContract(ctx)) ||
      failed(verifyAtlasGeneratedMXUContract(ctx)) || failed(verifyAtlasGeneratedTileContract(ctx)) ||
      failed(verifyAtlasGeneratedCFGContract(ctx)) ||
      failed(verifyAtlasGeneratedSourceMemoryEffectContract(ctx)) ||
      failed(verifyAtlasGeneratedBufferContract(ctx)))
    return failure();
  return ctx.timed ? verifyAtlasTiming(ctx) : success();
}

LogicalResult mlir::atlas::verifyAtlasGeneratedSchedule(ModuleOp module) {
  auto ctx = buildAtlasVerificationContext(module, /*generated=*/true);
  return failed(ctx) ? failure() : verifyAtlasGeneratedSchedule(*ctx);
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
