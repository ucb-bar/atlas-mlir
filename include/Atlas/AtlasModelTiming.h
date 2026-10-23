#ifndef ATLAS_MODEL_TIMING_H
#define ATLAS_MODEL_TIMING_H

#include "Atlas/AtlasTimingProvider.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <map>
#include <optional>

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

class ModelTimingFacts {
public:
  Footprint resolve(const Instr &in, const RegValues &regs) const;
  const std::string &factsSha256() const { return sha256; }
  const std::map<std::string, std::string> &modelProvenance() const { return provenance; }
  bool dmaWait() const { return dma; }
  TimingRuleResult<int> estimateDMACycles(int64_t bytes, bool store) const;
  const ModelTimingBlock *block(llvm::StringRef name) const {
    auto found = blocks.find(name.str());
    return found == blocks.end() ? nullptr : &found->second;
  }

private:
  std::map<std::string, ModelTimingBlock> blocks;
  std::map<std::string, std::string> provenance;
  std::string sha256;
  bool dma = false;
  std::optional<ModelDMATiming> dmaTiming;
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
