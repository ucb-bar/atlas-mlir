#ifndef ATLAS_DELAY_INSERTION_H
#define ATLAS_DELAY_INSERTION_H

#include "mlir/Pass/Pass.h"

namespace mlir::atlas {

// Register the delay insertion pass with atlas-opt.
void registerInsertAtlasDelaysPass();
std::unique_ptr<Pass> createInsertAtlasDelaysPass();

} // namespace mlir::atlas

#endif
