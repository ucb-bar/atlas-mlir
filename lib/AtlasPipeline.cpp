#include "Atlas/AtlasPipeline.h"
#include "Atlas/AtlasDMAWaitInsertion.h"
#include "Atlas/AtlasDelayInsertion.h"
#include "Atlas/AtlasGeneratedSchedule.h"
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
void buildVerifyStage(OpPassManager &pm) {
  pm.addPass(createSpillAtlasMXUStatePass());
  pm.addPass(createVerifyAtlasVirtualStreamPass());
}

// Boundary tiles become explicit DMA, which the scheduler can queue and
// overlap with other work.
void buildScheduleStage(OpPassManager &pm) {
  pm.addPass(createLowerAtlasVirtualBoundaryToDMAPass());
  pm.addPass(createScheduleAtlasVirtualPass());
  pm.addPass(createVerifyAtlasVirtualStreamPass());
}

// The schedule may exceed the registers; spilling recovers it.
void buildSpillStage(OpPassManager &pm) {
  pm.addPass(createSpillAtlasVirtualPass());
  pm.addPass(createVerifyAtlasVirtualStreamPass());
}

void buildLowerStage(OpPassManager &pm, const AtlasCompileOptions &options) {
  pm.addPass(createLowerAtlasVirtualToMachinePass(options.allocationReport));
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
  buildVerifyStage(pm);
  buildScheduleStage(pm);
  buildSpillStage(pm);
  buildLowerStage(pm, options);
  buildTimeStage(pm);
}

void mlir::atlas::buildAtlasCompileUnscheduledPipeline(
    OpPassManager &pm, const AtlasCompileOptions &options) {
  // Spilling admits MXU pressure, so it runs before strict verification.
  buildSpillStage(pm);
  buildLowerStage(pm, options);
  insertDMAWaits(pm);
  pm.addPass(createInsertAtlasDelaysPass());
  verifyTimedStream(pm);
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
  PassPipelineRegistration<>("atlas-compile-verify",
                             "atlas-compile stage 1: spill MXU state and verify",
                             buildVerifyStage);
  PassPipelineRegistration<>(
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
  PassPipelineRegistration<>(
      "atlas-compile-time",
      "atlas-compile stage 5: insert DMA waits, stream-schedule and verify timing",
      buildTimeStage);
}
