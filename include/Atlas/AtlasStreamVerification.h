#ifndef ATLAS_STREAM_VERIFICATION_H
#define ATLAS_STREAM_VERIFICATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::atlas {

// Register the pre-lowering physical-stream check with atlas-opt.
void registerVerifyAtlasMachineStreamPass();
std::unique_ptr<Pass> createVerifyAtlasMachineStreamPass();

} // namespace mlir::atlas

#endif
