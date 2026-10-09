#ifndef ATLAS_MXU_OWNERSHIP_H
#define ATLAS_MXU_OWNERSHIP_H

#include <array>
#include <cstdint>
#include <optional>

namespace mlir::atlas {

// Engaged when a transition finds a slot owner other than the one its
// invariant requires; holds that owner, which may be kFree or kUnknown.
using MXUOwnershipViolation = std::optional<int32_t>;

// Logical ownership of each MXU's two weight and two accumulator slots
// (WeightBuffers.scala and AccumulationBuffers.scala; geometry, not a physical
// release rule). A weight owns its slot until its last use, an accumulator
// version chain stays in one slot from seed or reset through continuation
// until readout, no push may overwrite a live slot, and every accumulator
// must be read out before an exit. Drivers supply their own owner ids, use
// facts and diagnostics. Transitions check their invariant and then apply the
// state change unconditionally, so a driver propagating state along every
// path can ignore the result.
struct MXUOwnership {
  static constexpr unsigned kUnits = 2, kSlots = 2;
  static constexpr int32_t kFree = -1;
  // A path join may store this; it satisfies no expectation.
  static constexpr int32_t kUnknown = -2;
  using Owners = std::array<std::array<int32_t, kSlots>, kUnits>;

  Owners weights = {{{kFree, kFree}, {kFree, kFree}}};
  Owners accumulators = {{{kFree, kFree}, {kFree, kFree}}};

  // A push without uses writes its slot and leaves it free.
  MXUOwnershipViolation pushWeight(unsigned unit, unsigned slot, int32_t weight, bool hasUses);
  MXUOwnershipViolation useWeight(unsigned unit, unsigned slot, int32_t weight, bool lastUse);
  // Seeds and resets start a version chain in a free slot.
  MXUOwnershipViolation startAccumulator(unsigned unit, unsigned slot, int32_t acc);
  MXUOwnershipViolation continueAccumulator(unsigned unit, unsigned slot, int32_t previous, int32_t next);
  MXUOwnershipViolation readoutAccumulator(unsigned unit, unsigned slot, int32_t acc);
  // Lowering pushes, resets and reads out a legacy matmul's fixed slots within
  // one operation: both must be free, and nothing stays owned afterwards.
  MXUOwnershipViolation legacyMatmul(unsigned unit, unsigned weightSlot, unsigned accSlot) const;
  MXUOwnershipViolation blockExit() const;
};

} // namespace mlir::atlas

#endif
