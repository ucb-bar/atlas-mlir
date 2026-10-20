#ifndef ATLAS_GUARDED_PACK_H
#define ATLAS_GUARDED_PACK_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::atlas {
// Opt-in finite-source PACK conversion before virtual scheduling/allocation.
std::unique_ptr<Pass> createLowerAtlasGuardedPackPass();
void registerLowerAtlasGuardedPackPass();
} // namespace mlir::atlas

#endif
