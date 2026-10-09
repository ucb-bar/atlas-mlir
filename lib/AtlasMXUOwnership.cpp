#include "Atlas/AtlasMXUOwnership.h"

using namespace mlir::atlas;

namespace {
MXUOwnershipViolation expect(int32_t owner, int32_t expected) {
  if (owner == expected)
    return std::nullopt;
  return owner;
}

// Checks the owner a transition requires, then installs the next owner.
MXUOwnershipViolation replace(int32_t &owner, int32_t expected, int32_t next) {
  MXUOwnershipViolation violation = expect(owner, expected);
  owner = next;
  return violation;
}
} // namespace

MXUOwnershipViolation MXUOwnership::pushWeight(unsigned unit, unsigned slot, int32_t weight, bool hasUses) {
  return replace(weights[unit][slot], kFree, hasUses ? weight : kFree);
}

MXUOwnershipViolation MXUOwnership::useWeight(unsigned unit, unsigned slot, int32_t weight, bool lastUse) {
  int32_t &owner = weights[unit][slot];
  return replace(owner, weight, lastUse ? kFree : owner);
}

MXUOwnershipViolation MXUOwnership::startAccumulator(unsigned unit, unsigned slot, int32_t acc) {
  return replace(accumulators[unit][slot], kFree, acc);
}

MXUOwnershipViolation MXUOwnership::continueAccumulator(unsigned unit, unsigned slot, int32_t previous, int32_t next) {
  return replace(accumulators[unit][slot], previous, next);
}

MXUOwnershipViolation MXUOwnership::readoutAccumulator(unsigned unit, unsigned slot, int32_t acc) {
  return replace(accumulators[unit][slot], acc, kFree);
}

MXUOwnershipViolation MXUOwnership::legacyMatmul(unsigned unit, unsigned weightSlot, unsigned accSlot) const {
  if (MXUOwnershipViolation violation = expect(weights[unit][weightSlot], kFree))
    return violation;
  return expect(accumulators[unit][accSlot], kFree);
}

MXUOwnershipViolation MXUOwnership::blockExit() const {
  for (const auto &unit : accumulators)
    for (int32_t owner : unit)
      if (owner != kFree)
        return owner;
  return std::nullopt;
}
