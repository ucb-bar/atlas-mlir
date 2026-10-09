#ifndef ATLAS_GENERATED_SCHEDULE_H
#define ATLAS_GENERATED_SCHEDULE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::atlas {
struct AtlasVerificationContext;
// Instruction classes permitted between a generated DMA launch and its wait.
bool canOverlapAtlasGeneratedDMA(Operation *op);
// Runs the generated checkers in order on one context built with `generated`.
LogicalResult verifyAtlasGeneratedSchedule(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasGeneratedSchedule(ModuleOp module);
void registerVerifyAtlasGeneratedSchedulePass();
}

#endif
