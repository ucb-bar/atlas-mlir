#ifndef ATLAS_VIRTUAL_SCHEDULING_H
#define ATLAS_VIRTUAL_SCHEDULING_H

#include "mlir/Pass/Pass.h"

namespace mlir::atlas {

// Register the pre-allocation virtual scheduling pass with atlas-opt.
void registerScheduleAtlasVirtualPass();
std::unique_ptr<Pass> createScheduleAtlasVirtualPass();

} // namespace mlir::atlas

#endif
