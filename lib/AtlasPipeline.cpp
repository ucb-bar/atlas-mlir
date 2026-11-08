#include "Atlas/AtlasPipeline.h"
#include "Atlas/AtlasDMAWaitInsertion.h"
#include "Atlas/AtlasDelayInsertion.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasGuardedPack.h"
#include "Atlas/AtlasScheduling.h"
#include "Atlas/AtlasSpilling.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasStreamVerification.h"
#include "Atlas/AtlasVirtualBoundaryToDMA.h"
#include "Atlas/AtlasVirtualScheduling.h"
#include "Atlas/AtlasVirtualToMachine.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::atlas;

namespace {
// Fresh mandatory lowering clears any previous provider, and delay insertion
// produces a timed built-in stream. The machine verifier already includes all
// generated contracts and strict timing, plus LLVM target checks. Repeating
// those checks in two more passes adds no facts; retain their metadata update.
// This private pass is not used by standalone or custom-provider pipelines.
struct FinalizeAtlasMandatoryCompilationPass
    : PassWrapper<FinalizeAtlasMandatoryCompilationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeAtlasMandatoryCompilationPass)
  StringRef getArgument() const final { return "finalize-atlas-mandatory-compilation"; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(requireAtlasGeneratedArtifact(module)))
      return signalPassFailure();
    auto state = module->getAttrOfType<StringAttr>(kAtlasTimingState);
    auto provider = module->getAttrOfType<StringAttr>(kAtlasTimingProvider);
    if (!state || state.getValue() != "timed" || !provider ||
        provider.getValue() != timing::kNpuModelTimingProviderId) {
      module.emitError("mandatory finalization requires a timed built-in generated stream");
      return signalPassFailure();
    }
    SmallVector<uint32_t> words;
    if (failed(verifyAtlasArtifact(module, /*llvmBlock=*/true, words)))
      return signalPassFailure();
    recordAtlasVerifiedTiming(module, provider.getValue());
  }
};

// Live MXU handles become saved tiles before the strict virtual verifier,
// which bounds each unit's slots.
void buildVerifyStage(OpPassManager &pm, const AtlasCompileOptions &options) {
  pm.addPass(createSpillAtlasMXUStatePass());
  pm.addPass(createVerifyAtlasVirtualStreamPass());
  if (options.guardedPack)
    pm.addPass(createLowerAtlasGuardedPackPass());
}

// Boundary tiles become explicit DMA, which the scheduler can queue and
// overlap with other work.
void buildScheduleStage(OpPassManager &pm) {
  pm.addPass(createLowerAtlasVirtualBoundaryToDMAPass());
  pm.addPass(createScheduleAtlasVirtualPass());
  pm.addPass(createVerifyAtlasVirtualStreamPass());
}

// The schedule may exceed the registers; spilling recovers it.
void buildSpillStage(OpPassManager &pm, bool scheduled) {
  pm.addPass(createSpillAtlasVirtualPass());
  pm.addPass(createVerifyAtlasVirtualStreamPass());
  pm.addPass(createLowerAtlasPackRelayoutPass());
  // The first schedule costed each PACK as its scalar relayout loop; order
  // the MXU round trips that replaced them by their own cost.
  if (scheduled)
    pm.addPass(createScheduleAtlasVirtualPass());
  // Keep the allocated helper values visible in the virtual stage artifact,
  // and recover any register pressure introduced by the quantized BF16 tile.
  pm.addPass(createSpillAtlasVirtualPass());
  pm.addPass(createVerifyAtlasVirtualStreamPass());
}

void buildLowerStage(OpPassManager &pm, const AtlasCompileOptions &options) {
  pm.addPass(createLowerAtlasVirtualToMachinePass(options.allocationReport,
                                                options.programSizeLimit));
}

// Each transfer's wait goes before what needs it done; the generated-schedule
// check requires the waits.
void insertDMAWaits(OpPassManager &pm) {
  pm.addPass(createInsertAtlasDMAWaitsPass());
  pm.addPass(createVerifyAtlasMachineStreamPass());
  pm.addPass(createVerifyAtlasGeneratedSchedulePass());
}

void verifyTimedStream(OpPassManager &pm) {
  pm.addPass(createVerifyAtlasMachineStreamPass());
  pm.addPass(createVerifyAtlasTimingPass());
  pm.addPass(createVerifyAtlasGeneratedSchedulePass());
}

// The stream scheduler reorders each block and inserts the delays its order
// needs.
void buildTimeStage(OpPassManager &pm) {
  insertDMAWaits(pm);
  pm.addPass(createScheduleAtlasStreamPass());
  verifyTimedStream(pm);
}
} // namespace

void mlir::atlas::buildAtlasCompilePipeline(OpPassManager &pm,
                                            const AtlasCompileOptions &options) {
  buildVerifyStage(pm, options);
  buildScheduleStage(pm);
  buildSpillStage(pm, /*scheduled=*/true);
  buildLowerStage(pm, options);
  buildTimeStage(pm);
}

void mlir::atlas::buildAtlasCompileUnscheduledPipeline(
    OpPassManager &pm, const AtlasCompileOptions &options) {
  if (options.guardedPack) {
    pm.addPass(createSpillAtlasMXUStatePass());
    pm.addPass(createLowerAtlasGuardedPackPass());
  }
  // Spilling admits MXU pressure, so it runs before strict verification.
  buildSpillStage(pm, /*scheduled=*/false);
  buildLowerStage(pm, options);
  insertDMAWaits(pm);
  pm.addPass(createInsertAtlasDelaysPass());
  pm.addPass(std::make_unique<FinalizeAtlasMandatoryCompilationPass>());
}

void mlir::atlas::registerAtlasCompilePipelines() {
  PassPipelineRegistration<AtlasCompileOptions>(
      "atlas-compile",
      "Compile one virtual Atlas function to timed, verified machine IR",
      buildAtlasCompilePipeline);
  PassPipelineRegistration<AtlasCompileOptions>(
      "atlas-compile-unscheduled",
      "Compile one virtual Atlas function without boundary DMA or scheduling, "
      "inserting delays in source order",
      buildAtlasCompileUnscheduledPipeline);
  PassPipelineRegistration<AtlasCompileOptions>("atlas-compile-verify",
                             "atlas-compile stage 1: spill MXU state and verify",
                             buildVerifyStage);
  PassPipelineRegistration<>(
      "atlas-compile-schedule",
      "atlas-compile stage 2: expose boundary DMA and schedule",
      buildScheduleStage);
  PassPipelineRegistration<>("atlas-compile-spill",
                             "atlas-compile stage 3: spill register pressure",
                             [](OpPassManager &pm) { buildSpillStage(pm, /*scheduled=*/true); });
  PassPipelineRegistration<AtlasCompileOptions>(
      "atlas-compile-lower",
      "atlas-compile stage 4: allocate and lower to machine IR",
      buildLowerStage);
  PassPipelineRegistration<>(
      "atlas-compile-time",
      "atlas-compile stage 5: insert DMA waits, stream-schedule and verify timing",
      buildTimeStage);
}
