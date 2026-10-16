#ifndef ATLAS_GENERATED_SCHEDULE_H
#define ATLAS_GENERATED_SCHEDULE_H

#include "Atlas/AtlasPackLoop.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::atlas {
struct AtlasVerificationContext;
// Whether the instruction at `pc` of `ops` (program order, atlas.start
// excluded) may issue between a generated DMA launch and its wait: work that
// moves no memory a transfer could own, or any instruction of one of `loops`,
// the stream's PACK relayout loops, whose VMEM the DMA memory check compares
// with each pending transfer.
bool canOverlapAtlasGeneratedDMA(llvm::ArrayRef<Operation *> ops,
                                 llvm::ArrayRef<AtlasPackLoop> loops, size_t pc);
// Generated-artifact entry: DMA lifecycle and redirect policy, then the memory,
// contract and timing checks, in order, on one `generated` context. Untagged
// boundary and mailbox transfers pair with the next same-channel untagged wait.
LogicalResult verifyAtlasGeneratedSchedule(const AtlasVerificationContext &ctx);
LogicalResult verifyAtlasGeneratedSchedule(ModuleOp module);
void registerVerifyAtlasGeneratedSchedulePass();
}

#endif
