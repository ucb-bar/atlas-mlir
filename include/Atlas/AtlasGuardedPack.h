#ifndef ATLAS_GUARDED_PACK_H
#define ATLAS_GUARDED_PACK_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::atlas {
// Marks shared clamp splats that allocation may copy per use.
inline constexpr char kAtlasGuardedPackBoundAttr[] = "atlas.guarded_pack_bound";

// Opt-in finite-source PACK conversion before virtual scheduling/allocation.
std::unique_ptr<Pass> createLowerAtlasGuardedPackPass();
void registerLowerAtlasGuardedPackPass();
} // namespace mlir::atlas

#endif
