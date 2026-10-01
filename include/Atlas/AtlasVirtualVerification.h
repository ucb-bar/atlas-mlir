#ifndef ATLAS_VIRTUAL_VERIFICATION_H
#define ATLAS_VIRTUAL_VERIFICATION_H

namespace mlir::atlas {

// Check a flat pre-allocation SSA candidate and its external state chain.
void registerVerifyAtlasVirtualStreamPass();

} // namespace mlir::atlas

#endif
