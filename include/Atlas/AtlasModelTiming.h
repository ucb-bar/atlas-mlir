#ifndef ATLAS_MODEL_TIMING_H
#define ATLAS_MODEL_TIMING_H

#include "Atlas/AtlasTimingProvider.h"
#include "Atlas/AtlasModelReplay.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <map>
#include <optional>
#include <set>

namespace mlir::atlas {

constexpr llvm::StringLiteral kAtlasModelEvidence = "atlas.model_evidence";

namespace timing {
constexpr llvm::StringLiteral kModelTimingProviderId = "model";
class ModelOverlapFacts;

// Shared recovery line for model derivation/evidence refusals; nesting is idempotent.
std::string modelTimingDiagnostic(llvm::StringRef message);

struct ModelTimingGroup {
  int firstAge = 0, lastAge = 0, count = 0, step = 1;
};

struct ModelReservationRelease {
  int firstAge = 0, releaseAge = 0;
  std::vector<int> banks;
};

struct ModelTimingBlock {
  std::string engine;
  std::map<std::string, ModelTimingGroup> events;
  int firstFreeAge = 0, readLatency = 1;
  std::map<std::string, ModelReservationRelease> releases;
};

// Reviewed cold isolated transfers, with pure read/store history and no VMEM grant denial.
// These fields are proposal inputs and make no completion or overlap claim.
struct ModelDMATiming {
  int beatBytes, maxBytes, maxInFlight, queueDepth, issueInterval;
  int loadStartup, storeStartup;
  double readLatency, writeLatency, readSpacing, writeSpacing, epsilon, resetReady;
  TimingRuleResult<int> estimate(int64_t bytes, bool store) const;
};

struct ModelFrontendAccess {
  std::string operand;
  bool write, scale;
};

struct ModelFrontendTiming {
  int ordinaryGap, delayBase, delayMask, readAge, writeAge, controlReadAge;
  int delaySlots;
  std::map<std::string, std::vector<ModelFrontendAccess>> instructions;
};

// Engine issue predicates are relational; isolated lifetimes remain envelopes.
struct ModelEngineAdmission {
  int vpuCapacity = 0;
  int scalarLoadGap = 0;
  std::map<std::string, int> vpuFinalWrite;
  std::set<std::string> vpuDouble;
  std::vector<std::set<std::string>> vpuLogicGroups;
};

class ModelTimingFacts {
public:
  Footprint resolve(const Instr &in, const RegValues &regs) const;
  const std::string &factsSha256() const { return sha256; }
  const ProposalGeometry &proposalGeometry() const { return geometry; }
  int dmaCapacity() const { return dmaChannels; }
  const ModelReplayProgram *replayProgram() const { return replay ? &*replay : nullptr; }
  TimingRuleResult<std::pair<Instr, Instr>> costProbes(const OpInfo &op) const;
  FrontendSpacing frontendSpacing() const {
    return frontend ? FrontendSpacing{frontend->ordinaryGap, frontend->delaySlots} : FrontendSpacing{};
  }
  const std::map<std::string, std::string> &modelProvenance() const { return provenance; }
  bool dmaWait() const { return dma; }
  const std::string &overlapRulesSha256() const { return overlapRules; }
  bool allowsUnboundTestOverlap() const { return testOverlap; }
  const ModelEngineAdmission *engineAdmission() const {
    return admission ? &*admission : nullptr;
  }
  TimingRuleResult<int> estimateDMACycles(int64_t bytes, bool store) const;
  TimingRuleResult<int> issueGap(const Instr &in) const;
  const ModelTimingBlock *block(llvm::StringRef name) const {
    auto found = blocks.find(name.str());
    return found == blocks.end() ? nullptr : &found->second;
  }

private:
  std::map<std::string, ModelTimingBlock> blocks;
  std::map<std::string, std::vector<ModelTimingGroup>> weightUses;
  std::map<std::string, std::string> provenance;
  std::string sha256;
  bool dma = false;
  bool testOverlap = false;
  std::string overlapRules;
  std::optional<ModelDMATiming> dmaTiming;
  std::optional<ModelFrontendTiming> frontend;
  ProposalGeometry geometry;
  int dmaChannels = kAtlasDMAChannels;
  std::optional<ModelReplayProgram> replay;
  std::map<std::string, std::pair<Instr, Instr>> probes;
  std::optional<ModelEngineAdmission> admission;
  static llvm::Expected<ModelTimingFacts> parseImpl(llvm::StringRef, llvm::StringRef, bool, bool);
  friend llvm::Expected<ModelTimingFacts> parseModelTimingFacts(llvm::StringRef, llvm::StringRef, bool);
  friend llvm::Expected<ModelTimingFacts> parseModelTimingFactsForTesting(llvm::StringRef, llvm::StringRef, bool);
};

bool hashSyntax(llvm::StringRef value);
std::string modelSha256(llvm::StringRef bytes);
std::string canonicalJson(const llvm::json::Value &value);
llvm::Expected<ModelTimingFacts> parseModelTimingFacts(llvm::StringRef bytes, llvm::StringRef expectedSha256,
                                                       bool dmaWait = false);
// Explicit unit-test boundary for raw observations and synthetic timing operands.
// Production loaders and command-line selectors always use parseModelTimingFacts.
llvm::Expected<ModelTimingFacts> parseModelTimingFactsForTesting(llvm::StringRef bytes, llvm::StringRef expectedSha256,
                                                                 bool dmaWait = false);
llvm::Expected<ModelTimingFacts> loadModelTimingFacts(llvm::StringRef path, llvm::StringRef expectedSha256,
                                                      bool dmaWait = false);
TimingProvider modelTimingProvider(std::shared_ptr<const ModelTimingFacts> facts,
                                  std::shared_ptr<const ModelOverlapFacts> overlap = {}, bool chains = false);
TimingRuleResult<TimingProvider> modelTimingProviderFactory(ModuleOp module);
} // namespace timing

void registerSelectAtlasModelEvidencePass();
} // namespace mlir::atlas

#endif
