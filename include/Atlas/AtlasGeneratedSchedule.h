#ifndef ATLAS_GENERATED_SCHEDULE_H
#define ATLAS_GENERATED_SCHEDULE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::atlas {
struct AtlasVerificationContext;
// Instruction classes permitted between a generated DMA launch and its wait.
bool canOverlapAtlasGeneratedDMA(Operation *op);
// Generated-artifact entry: DMA lifecycle and redirect policy, then the memory,
// contract and timing checks, in order, on one `generated` context. Untagged
// boundary and mailbox transfers pair with the next same-channel untagged wait.
LogicalResult verifyAtlasGeneratedSchedule(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasGeneratedSchedule(ModuleOp module);
void registerVerifyAtlasGeneratedSchedulePass();
}

#endif
