#ifndef ATLAS_VIRTUAL_SCHEDULING_H
#define ATLAS_VIRTUAL_SCHEDULING_H

namespace mlir::atlas {

// Register the pre-allocation virtual scheduling pass with atlas-opt.
void registerScheduleAtlasVirtualPass();

} // namespace mlir::atlas

#endif
