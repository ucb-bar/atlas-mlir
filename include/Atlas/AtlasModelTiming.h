#ifndef ATLAS_MODEL_TIMING_H
#define ATLAS_MODEL_TIMING_H

#include "Atlas/AtlasTimingProvider.h"
#include "llvm/Support/Error.h"
#include <map>

namespace mlir::atlas {

constexpr llvm::StringLiteral kAtlasModelEvidence = "atlas.model_evidence";

namespace timing {
constexpr llvm::StringLiteral kNpuModelObservedTimingProviderId = "npu-model";

struct ModelTimingGroup {
  int firstAge = 0, lastAge = 0, count = 0, step = 1;
};

struct ModelTimingBlock {
  std::string engine;
  std::map<std::string, ModelTimingGroup> events;
  int firstFreeAge = 0, nextIssueAge = 0, readLatency = 1;
};

class ModelTimingFacts {
public:
  Footprint resolve(const Instr &in, const RegValues &regs) const;
  const std::string &factsSha256() const { return sha256; }
  const std::map<std::string, std::string> &modelProvenance() const { return provenance; }
  bool dmaWait() const { return dma; }

private:
  std::map<std::string, ModelTimingBlock> blocks;
  std::map<std::string, std::string> provenance;
  std::string sha256;
  bool dma = false;
  friend llvm::Expected<ModelTimingFacts> parseModelTimingFacts(llvm::StringRef, llvm::StringRef, bool);
};

llvm::Expected<ModelTimingFacts> parseModelTimingFacts(llvm::StringRef bytes, llvm::StringRef expectedSha256,
                                                       bool dmaWait = false);
llvm::Expected<ModelTimingFacts> loadModelTimingFacts(llvm::StringRef path, llvm::StringRef expectedSha256,
                                                      bool dmaWait = false);
TimingProvider modelTimingProvider(std::shared_ptr<const ModelTimingFacts> facts);
TimingRuleResult<TimingProvider> modelTimingProviderFactory(ModuleOp module);
} // namespace timing

void registerSelectAtlasModelEvidencePass();
} // namespace mlir::atlas

#endif
