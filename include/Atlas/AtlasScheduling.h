#ifndef ATLAS_SCHEDULING_H
#define ATLAS_SCHEDULING_H

namespace mlir::atlas {

// Register the list scheduling pass with atlas-opt.
void registerScheduleAtlasStreamPass();

} // namespace mlir::atlas

#endif
