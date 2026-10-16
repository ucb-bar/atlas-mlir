#ifndef ATLAS_MODEL_PAIRS_H
#define ATLAS_MODEL_PAIRS_H

#include "Atlas/AtlasModelTiming.h"

namespace mlir::atlas::timing {

struct ModelOverlapCommand {
  std::string block;
  Engine engine;
  std::vector<int> registers;
  std::optional<int> accumulator, weightSlot;
  int issue = 0, firstFreeAge = 0;
};

class ModelOverlapFacts {
public:
  bool admits(const std::vector<ModelOverlapCommand> &commands, bool chains) const;
  const std::map<std::string, std::string> &metadata() const { return provenance; }

private:
  using Relation = std::map<std::string, std::string>;
  struct Pair {
    std::string first, second;
    Relation relation;
    int minimum = 0, high = 0;
  };
  struct Chain {
    std::vector<std::string> operations;
    std::vector<int> gaps;
    std::map<std::pair<int, int>, Relation> relations;
    int inFlight = 0;
  };
  std::vector<Pair> pairs;
  std::vector<Chain> sequences;
  std::map<std::string, std::string> provenance;
  friend llvm::Expected<ModelOverlapFacts> parseModelOverlapFacts(llvm::StringRef, llvm::StringRef,
                                                                  const ModelTimingFacts &);
};

llvm::Expected<ModelOverlapFacts> parseModelOverlapFacts(llvm::StringRef bytes, llvm::StringRef expectedSha256,
                                                         const ModelTimingFacts &operations);
llvm::Expected<ModelOverlapFacts> loadModelOverlapFacts(llvm::StringRef path, llvm::StringRef expectedSha256,
                                                        const ModelTimingFacts &operations);
} // namespace mlir::atlas::timing

#endif
