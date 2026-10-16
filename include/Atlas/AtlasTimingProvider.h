#ifndef ATLAS_TIMING_PROVIDER_H
#define ATLAS_TIMING_PROVIDER_H

#include "Atlas/AtlasTiming.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <utility>

namespace mlir::atlas {
struct AtlasStream;

namespace timing {

// Missing coverage is an error, distinct from a covered rule that reports no
// dependence or no resource conflict. No generic consumer fills missing rules.
template <typename T> struct TimingRuleResult {
  T value{};
  std::string error;
};

struct DMACompletionConflict {
  bool conflict = false;
  EdgeKind kind = EdgeKind::Order;
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
  virtual TimingRuleResult<std::string> conflictWithContext(
      const Instr &in, const Footprint &f, int cycle, const TimingContext &) const {
    return conflict(in, f, cycle);
  }
  virtual std::string reserveWithContext(const Instr &in, const Footprint &f,
                                         int cycle, const TimingContext &) {
    return reserve(in, f, cycle);
  }
};

// Complete inputs to final timing verification; scope validation runs first.
// CFG and host-release obligations stay in the common verifier. A provider id
// names policy, not RTL qualification.
struct TimingProvider {
  std::string id;
  std::function<std::string(const AtlasStream &)> validateScope;
  FootprintResolver footprint;
  std::function<TimingRuleResult<Dependence>(
      const Instr &, const Footprint &, const Instr &, const Footprint &)>
      dependence;
  std::function<TimingRuleResult<Dependence>(
      const Instr &, const Footprint &, const TimingContext &,
      const Instr &, const Footprint &, const TimingContext &)>
      contextualDependence;
  TimingRuleResult<Dependence> dependenceFor(
      const Instr &a, const Footprint &fa, const TimingContext &ca,
      const Instr &b, const Footprint &fb, const TimingContext &cb) const {
    return contextualDependence ? contextualDependence(a, fa, ca, b, fb, cb)
                                : dependence(a, fa, b, fb);
  }
  std::function<TimingRuleResult<DMACompletionConflict>(
      const Footprint &, const Footprint &)> dmaConflict;
  std::function<TimingRuleResult<int>(const Instr &)> issueGap;
  std::function<TimingRuleResult<std::unique_ptr<TimingReservations>>()>
      createReservations;
  // Optional legal virtual-cost operands; absent preserves legacy probes.
  std::function<TimingRuleResult<std::pair<Instr, Instr>>(const OpInfo &)>
      costProbes;
};

std::string validateTimingProvider(const TimingProvider &provider);
// Provider graph; absent incoming DMA facts retain broad post-wait guards.
TimingRuleResult<DepGraph> buildTimingGraph(
    const std::vector<Instr> &instrs, const RegValues &entry,
    const TimingProvider &provider, const IncomingDma *incomingDma = nullptr,
    const std::vector<TimingContext> *contexts = nullptr);

struct MachineOpTiming {
  int latency, interval;
};
// Cost probes retain representative operands and unknown bases; they are not verified instructions.
FailureOr<MachineOpTiming> machineTimingOf(const TimingProvider &provider,
                                        const std::string &name, Operation *at);

constexpr llvm::StringLiteral kBuiltinTimingProviderId = "builtin";
constexpr llvm::StringLiteral kNpuModelTimingProviderId = "npu-model-rtl-match-v1"; // Legacy ID.
llvm::StringRef canonicalTimingProviderId(llvm::StringRef id);
// Built-in rules ported from npu_model's rtl-match mode.
TimingProvider builtinTimingProvider();
// Legacy C++ entry point.
TimingProvider npuModelTimingProvider();
// Module-scoped factory; its returned ID must match the canonical registered ID.
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
