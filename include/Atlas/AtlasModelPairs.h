#ifndef ATLAS_MODEL_PAIRS_H
#define ATLAS_MODEL_PAIRS_H

#include "Atlas/AtlasModelTiming.h"
#include <set>

namespace mlir::atlas::timing {

struct ModelOverlapCommand {
  std::string block;
  Engine engine;
  std::vector<int> registers;
  std::optional<int> accumulator, weightSlot;
  int issue = 0, firstFreeAge = 0;
  std::optional<Instr> scalarInstruction;
  TimingContext context;
};

class ModelOverlapFacts {
public:
  bool admits(const std::vector<ModelOverlapCommand> &commands, bool chains,
              llvm::StringRef origin = "dependence") const;
  bool rejectsPortHistory(const std::vector<ModelOverlapCommand> &commands) const;
  const std::map<std::string, std::string> &metadata() const { return provenance; }

private:
  using Relation = std::map<std::string, std::string>;
  struct Pair {
    std::string first, second;
    Relation relation;
    int minimum = 0, high = 0;
    std::string source = "source-derived";
  };
  std::vector<Pair> pairs;
  std::map<std::pair<std::string, std::string>, std::vector<size_t>> pairIndices;
  std::map<std::string, std::set<Relation>> compositionClasses;
  std::set<std::string> composedOperations;
  bool structuralCapacity = false;
  int sequencerCycles = 0, mxu0Compute = 0, mxu1Compute = 0, vpuSlots = 0;
  std::map<std::string, std::string> provenance;
  mutable std::set<std::string> loggedQueries;
  static llvm::Expected<ModelOverlapFacts> parseDerivedFacts(const llvm::json::Object &, llvm::StringRef,
                                                            const ModelTimingFacts &);
  friend llvm::Expected<ModelOverlapFacts> parseModelOverlapFacts(llvm::StringRef, llvm::StringRef,
                                                                  const ModelTimingFacts &);
};

llvm::Expected<ModelOverlapFacts> parseModelOverlapFacts(llvm::StringRef bytes, llvm::StringRef expectedSha256,
                                                         const ModelTimingFacts &operations);
llvm::Expected<ModelOverlapFacts> loadModelOverlapFacts(llvm::StringRef path, llvm::StringRef expectedSha256,
                                                        const ModelTimingFacts &operations);
} // namespace mlir::atlas::timing

#endif
