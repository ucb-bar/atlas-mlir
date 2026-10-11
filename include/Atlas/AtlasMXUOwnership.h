#ifndef ATLAS_MXU_OWNERSHIP_H
#define ATLAS_MXU_OWNERSHIP_H

#include <array>
#include <cstdint>
#include <optional>

namespace mlir::atlas {

// Engaged when a transition finds a slot owner other than the one its
// invariant requires; holds that owner, which may be kFree or kUnknown.
using MXUOwnershipViolation = std::optional<int32_t>;

// Logical owners of Weight/AccumulationBuffers.scala slots, not a release rule:
// a weight until last use, an accumulator version chain until readout before
// exit; no push over live slots. Transitions apply despite reported violations.
struct MXUOwnership {
  static constexpr unsigned kUnits = 2, kSlots = 2;
  static constexpr int32_t kFree = -1;
  // A path join may store this; it satisfies no expectation.
  static constexpr int32_t kUnknown = -2;
  using Owners = std::array<std::array<int32_t, kSlots>, kUnits>;

  Owners weights = {{{kFree, kFree}, {kFree, kFree}}};
  Owners accumulators = {{{kFree, kFree}, {kFree, kFree}}};

  // A push without uses writes its slot and leaves it free.
  MXUOwnershipViolation pushWeight(unsigned unit, unsigned slot, int32_t weight, bool hasUses) {
    return replace(weights[unit][slot], kFree, hasUses ? weight : kFree);
  }
  MXUOwnershipViolation useWeight(unsigned unit, unsigned slot, int32_t weight, bool lastUse) {
    int32_t &owner = weights[unit][slot];
    return replace(owner, weight, lastUse ? kFree : owner);
  }
  // Seeds and resets start a version chain in a free slot.
  MXUOwnershipViolation startAccumulator(unsigned unit, unsigned slot, int32_t acc) {
    return replace(accumulators[unit][slot], kFree, acc);
  }
  MXUOwnershipViolation continueAccumulator(unsigned unit, unsigned slot, int32_t previous, int32_t next) {
    return replace(accumulators[unit][slot], previous, next);
  }
  MXUOwnershipViolation readoutAccumulator(unsigned unit, unsigned slot, int32_t acc) {
    return replace(accumulators[unit][slot], acc, kFree);
  }
  // Lowering pushes, resets and reads out a legacy matmul's fixed slots within
  // one operation: both must be free, and nothing stays owned afterwards.
  MXUOwnershipViolation legacyMatmul(unsigned unit, unsigned weightSlot, unsigned accSlot) const {
    if (MXUOwnershipViolation violation = expect(weights[unit][weightSlot], kFree))
      return violation;
    return expect(accumulators[unit][accSlot], kFree);
  }
  MXUOwnershipViolation blockExit() const {
    for (const auto &unit : accumulators)
      for (int32_t owner : unit)
        if (owner != kFree)
          return owner;
    return std::nullopt;
  }

private:
  static MXUOwnershipViolation expect(int32_t owner, int32_t expected) {
    if (owner == expected)
      return std::nullopt;
    return owner;
  }
  // Checks the owner a transition requires, then installs the next owner.
  static MXUOwnershipViolation replace(int32_t &owner, int32_t expected, int32_t next) {
    MXUOwnershipViolation violation = expect(owner, expected);
    owner = next;
    return violation;
  }
};

} // namespace mlir::atlas

#endif
