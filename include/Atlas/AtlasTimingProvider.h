#ifndef ATLAS_TIMING_PROVIDER_H
#define ATLAS_TIMING_PROVIDER_H

#include "Atlas/AtlasTiming.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::atlas {
struct AtlasStream;

namespace timing {

struct DMACompletionConflict {
  bool conflict = false;
  EdgeKind kind = EdgeKind::Order;
};

// Estimated queue end and matching wait issue; completion requires a DMA wait.
struct DMAProposal {
  int queueEnd = 0, waitIssue = 0;
};

// A fresh instance belongs to one emitted basic block. The implementation owns
// capacities, port sharing, engine overlap and reservations across a DMA wait.
class TimingReservations {
public:
  virtual ~TimingReservations() = default;
  virtual TimingRuleResult<std::string>
  conflict(const Instr &in, const Footprint &footprint, int cycle) const = 0;
  virtual std::string reserve(const Instr &in, const Footprint &footprint,
                              int cycle) = 0;
  virtual std::string onWait(const Instr &in, int cycle) = 0;
};

std::unique_ptr<TimingReservations> createTimingReservations(
    std::shared_ptr<const TimingTables> tables = {});

// Frontend issue spacing and the number of delay slots.
struct FrontendSpacing {
  int ordinaryGap = 1, delaySlots = 1;
  int successorGap() const { return (delaySlots + 1) * ordinaryGap; }
};

struct MachineOpTiming {
  int latency, interval;
};

// Complete inputs to final timing verification; scope validation runs first.
// CFG and host-release obligations stay in the common verifier. A provider id
// names policy, not RTL qualification.
struct TimingProvider {
  std::string id;
  // Format a model refusal when a consumer emits its diagnostic.
  std::string diagnostic(llvm::StringRef error) const;
  unsigned packRelayoutRows = 0; // Explicit compiler code-size policy override.
  FrontendSpacing frontendSpacing;
  std::shared_ptr<const TimingTables> model;
  std::map<std::string, MachineOpTiming> machineTiming;
  std::function<std::string(const AtlasStream &)> validateScope;
  FootprintResolver footprint;
  std::function<TimingRuleResult<Dependence>(
      const Instr &, const Footprint &, const Instr &, const Footprint &)>
      dependence;
  std::function<TimingRuleResult<DMACompletionConflict>(
      const Footprint &, const Footprint &)> dmaConflict;
  std::function<TimingRuleResult<int>(const Instr &)> issueGap;
  std::function<TimingRuleResult<std::unique_ptr<TimingReservations>>()>
      createReservations;
  // Proposal only: completion is still established by matching DMA waits.
  // Optional for providers that never estimate DMA; no consumer borrows another provider's policy.
  std::function<TimingRuleResult<int>(int64_t bytes, bool store)> dmaEstimate;
  // Shared by critical-height ranking and physical wait placement. No fallback
  // to another provider's offsets when a transfer has a positive estimate.
  std::function<TimingRuleResult<DMAProposal>(
      int latency, int issue, std::optional<int> previousEnd)> dmaProposal;
  bool fillPendingDelaySlots = false;
  // Optional fixed proposal cost; zero derives the scalar address/load/store sequence.
  int scalarSlotProposalCycles = 0;
  // DMA wait insertion completes launch n-8 before launch n reuses its command slot.
  bool waitReusedDMASlots = false;
};

std::string validateTimingProvider(const TimingProvider &provider);

// Latency is the final event or release age plus one. The issue interval uses
// reservation queries for two representative operands. Actual instructions
// still require their own dependence and reservation checks.
TimingRuleResult<MachineOpTiming> machineTimingOf(
    const TimingProvider &provider, const std::string &name);
std::string fillMachineTiming(TimingProvider &provider);

constexpr llvm::StringLiteral kBuiltinTimingProviderId = "builtin";
// Virtual costs and physical filling must use the same provider capability.
inline bool fillsDelaySlots(const TimingProvider &provider) {
  return provider.fillPendingDelaySlots;
}
// Built-in rules ported from npu_model's rtl-match mode.
TimingProvider builtinTimingProvider();
// Module-scoped factory; its returned ID must match the registered ID.
using TimingProviderFactory =
    std::function<TimingRuleResult<TimingProvider>(mlir::ModuleOp module)>;
// Adds `id` to the process-wide registry, which is seeded with the built-in
// provider. Returns an error for an empty, duplicate or null registration.
std::string registerAtlasTimingProvider(const std::string &id,
                                        TimingProviderFactory factory);
// Only complete registered policies are returned; unknown and footprint-only
// CIRCT evidence ids fail rather than borrow model rules.
TimingRuleResult<TimingProvider> lookupTimingProvider(const std::string &id,
                                                      mlir::ModuleOp module = {});

} // namespace timing
} // namespace mlir::atlas

#endif
