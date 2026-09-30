#ifndef ATLAS_STREAM_VERIFICATION_H
#define ATLAS_STREAM_VERIFICATION_H

namespace mlir::atlas {

// Register the pre-lowering physical-stream check with atlas-opt.
void registerVerifyAtlasMachineStreamPass();

} // namespace mlir::atlas

#endif
