#ifndef ATLAS_VIRTUAL_SCHEDULING_H
#define ATLAS_VIRTUAL_SCHEDULING_H

#include "mlir/Pass/Pass.h"

namespace mlir::atlas {

// Register the pre-allocation virtual scheduling pass with atlas-opt.
void registerScheduleAtlasVirtualPass();
std::unique_ptr<Pass> createScheduleAtlasVirtualPass(
    llvm::StringRef provider = "", unsigned lookahead = 0, int randomSeed = -1,
    llvm::StringRef orderFile = "", llvm::StringRef graphOutput = "",
    bool orderApplied = false, bool rememberOptions = false);

} // namespace mlir::atlas

#endif
