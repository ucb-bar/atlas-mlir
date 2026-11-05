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
// Repair PACK's physical layout through an available SSA-owned accumulator.
std::unique_ptr<Pass> createLowerAtlasPackRelayoutPass();
void registerLowerAtlasPackRelayoutPass();
} // namespace mlir::atlas

#endif
