#ifndef ATLAS_DELAY_INSERTION_H
#define ATLAS_DELAY_INSERTION_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::atlas {

// Register the delay insertion pass with atlas-opt.
void registerInsertAtlasDelaysPass();
std::unique_ptr<Pass> createInsertAtlasDelaysPass(
    const std::string &provider = "");

} // namespace mlir::atlas

#endif
