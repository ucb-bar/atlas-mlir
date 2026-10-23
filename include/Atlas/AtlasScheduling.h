#ifndef ATLAS_SCHEDULING_H
#define ATLAS_SCHEDULING_H

#include "mlir/Pass/Pass.h"

namespace mlir::atlas {

// Register the list scheduling pass with atlas-opt.
void registerScheduleAtlasStreamPass();
std::unique_ptr<Pass> createScheduleAtlasStreamPass(llvm::StringRef provider = "");

} // namespace mlir::atlas

#endif
