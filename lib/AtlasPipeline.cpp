#include "Atlas/AtlasPipeline.h"
#include "Atlas/AtlasDMAWaitInsertion.h"
#include "Atlas/AtlasDelayInsertion.h"
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
// Live MXU handles become saved tiles before the strict virtual verifier,
// which bounds each unit's slots.
void buildVerifyStage(OpPassManager &pm, const AtlasCompileOptions &options) {
  if (options.spillMXUState)
    pm.addPass(createSpillAtlasMXUStatePass());
  pm.addPass(createVerifyAtlasVirtualStreamPass());
  if (options.guardedPack)
    pm.addPass(createLowerAtlasGuardedPackPass());
}

// Boundary tiles become explicit DMA, which the scheduler can queue and
// overlap with other work.
void buildBoundaryStage(OpPassManager &pm, const AtlasCompileOptions &options) {
  if (options.boundary == AtlasCompileOptions::Boundary::All)
    pm.addPass(createLowerAtlasVirtualBoundaryToDMAPass());
}

void buildVirtualScheduleStage(OpPassManager &pm, const AtlasCompileOptions &options) {
  pm.addPass(createScheduleAtlasVirtualPass(options.provider, options.lookahead, options.randomSeed,
                                           options.orderFile, options.graphOutput));
  pm.addPass(createVerifyAtlasVirtualStreamPass());
}

void buildScheduleStage(OpPassManager &pm, const AtlasCompileOptions &options) {
  buildBoundaryStage(pm, options);
  buildVirtualScheduleStage(pm, options);
}

// Prepare once, so repeated candidate orders use the same graph node IDs.
void buildPrepareStage(OpPassManager &pm, const AtlasCompileOptions &options) {
  buildVerifyStage(pm, options);
  buildBoundaryStage(pm, options);
  pm.addPass(createVerifyAtlasVirtualStreamPass());
}

// The schedule may exceed the registers; spilling recovers it.
void buildSpillStage(OpPassManager &pm) {
  pm.addPass(createSpillAtlasVirtualPass());
  pm.addPass(createVerifyAtlasVirtualStreamPass());
}

void buildLowerStage(OpPassManager &pm, const AtlasCompileOptions &options) {
  pm.addPass(createLowerAtlasVirtualToMachinePass(options.allocationReport, options.provider, options.storeWindows,
                                                     options.mregRanks, options.packReloadAfterWeight));
}

// Each transfer's wait goes before what needs it done; the generated-schedule
// check requires the waits.
void insertDMAWaits(OpPassManager &pm, const AtlasCompileOptions &options) {
  pm.addPass(createInsertAtlasDMAWaitsPass(options.provider));
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
void buildTimeStage(OpPassManager &pm, const AtlasCompileOptions &options) {
  insertDMAWaits(pm, options);
  pm.addPass(createScheduleAtlasStreamPass(options.provider, options.waitPlacement));
  verifyTimedStream(pm);
}
} // namespace

void mlir::atlas::buildAtlasCompilePipeline(OpPassManager &pm,
                                            const AtlasCompileOptions &options) {
  buildVerifyStage(pm, options);
  buildScheduleStage(pm, options);
  if (options.spill)
    buildSpillStage(pm);
  buildLowerStage(pm, options);
  buildTimeStage(pm, options);
}

void mlir::atlas::buildAtlasCompileUnscheduledPipeline(
    OpPassManager &pm, const AtlasCompileOptions &options) {
  if (options.guardedPack) {
    pm.addPass(createSpillAtlasMXUStatePass());
    pm.addPass(createLowerAtlasGuardedPackPass());
  }
  // Spilling admits MXU pressure, so it runs before strict verification.
  buildSpillStage(pm);
  buildLowerStage(pm, options);
  insertDMAWaits(pm, options);
  pm.addPass(createInsertAtlasDelaysPass());
  verifyTimedStream(pm);
}

void mlir::atlas::registerAtlasCompilePipelines() {
  PassPipelineRegistration<AtlasCompileOptions>(
      "atlas-compile-prepare", "Prepare a shared virtual graph for candidate search", buildPrepareStage);
  PassPipelineRegistration<AtlasCompileOptions>(
      "atlas-compile-candidate", "Compile and verify one prepared virtual candidate",
      [](OpPassManager &pm, const AtlasCompileOptions &options) {
        buildVirtualScheduleStage(pm, options);
        if (options.spill)
          buildSpillStage(pm);
        buildLowerStage(pm, options);
        buildTimeStage(pm, options);
      });
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
  PassPipelineRegistration<AtlasCompileOptions>(
      "atlas-compile-schedule",
      "atlas-compile stage 2: expose boundary DMA and schedule",
      buildScheduleStage);
  PassPipelineRegistration<>("atlas-compile-spill",
                             "atlas-compile stage 3: spill register pressure",
                             buildSpillStage);
  PassPipelineRegistration<AtlasCompileOptions>(
      "atlas-compile-lower",
      "atlas-compile stage 4: allocate and lower to machine IR",
      buildLowerStage);
  PassPipelineRegistration<AtlasCompileOptions>(
      "atlas-compile-time",
      "atlas-compile stage 5: insert DMA waits, stream-schedule and verify timing",
      buildTimeStage);
}
