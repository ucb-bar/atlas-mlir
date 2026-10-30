#ifndef ATLAS_MODEL_TABLES_H
#define ATLAS_MODEL_TABLES_H

#include "Atlas/AtlasTimingProvider.h"
#include "llvm/Support/Error.h"

namespace mlir::atlas {
constexpr llvm::StringLiteral kAtlasModelEvidence = "atlas.model_evidence";
namespace timing {
constexpr llvm::StringLiteral kModelTimingProviderId = "model";

struct ModelDMATiming {
  int beatBytes = 0, maxBytes = 0, loadStartup = 0, storeStartup = 0;
  int perBeatBound = 0, retireToWait = 0;
  // Compiler policy: serialize DMA estimates by one cycle between commands.
  int serializationGap = 1;
  TimingRuleResult<int> estimate(int64_t bytes, bool store) const;
  TimingRuleResult<DMAProposal> propose(
      int latency, int issue, std::optional<int> previousEnd) const;
};

struct ModelTimingTables {
  std::shared_ptr<const TimingTables> timing;
  ModelDMATiming dma;
  FrontendSpacing frontend;
  int instructionMemoryWords = 0;
  std::vector<std::string> frontendInstructions;
};

std::string modelTimingDiagnostic(llvm::StringRef message);
std::string modelSha256(llvm::StringRef bytes);
llvm::Expected<ModelTimingTables> parseModelTimingTables(llvm::StringRef bytes);
llvm::Expected<ModelTimingTables> loadModelTimingTables(
    llvm::StringRef path, llvm::StringRef expectedHash);
TimingProvider modelTimingProvider(std::shared_ptr<const ModelTimingTables> facts);
TimingRuleResult<TimingProvider> modelTimingProviderFactory(ModuleOp module);
} // namespace timing
void registerSelectAtlasModelEvidencePass();
} // namespace mlir::atlas
#endif
