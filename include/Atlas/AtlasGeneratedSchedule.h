#ifndef ATLAS_GENERATED_SCHEDULE_H
#define ATLAS_GENERATED_SCHEDULE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::atlas {
struct AtlasVerificationContext;
// Instruction classes permitted between a generated DMA launch and its wait.
bool canOverlapAtlasGeneratedDMA(Operation *op);
// Entry point for generated artifacts (AtlasGeneratedArtifact.h): DMA
// lifecycle and redirect policy, then the memory, contract and timing checks.
// Implicit boundary and mailbox transfers carry no transfer ID; their wait is
// the next same-channel wait without an ID under the same interval policy.
// The context overload runs the checkers in order on one context built with
// `generated`; the ModuleOp overload builds that context.
LogicalResult verifyAtlasGeneratedSchedule(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasGeneratedSchedule(ModuleOp module);
void registerVerifyAtlasGeneratedSchedulePass();
}

#endif
