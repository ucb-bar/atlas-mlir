#ifndef ATLAS_RTL_EVIDENCE_H
#define ATLAS_RTL_EVIDENCE_H

#include "Atlas/AtlasTiming.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace mlir::atlas::timing {

// One event group of a merlin.op_timing.v1 block; ages count from issue.
struct OpTimingGroup {
  int firstAge = 0, lastAge = 0, count = 0;
  std::optional<int> step;
};

struct OpTimingBlock {
  bool resolved = false; // events is an object, not null
  std::string engine;
  std::map<std::string, OpTimingGroup> events;
  std::optional<int> firstFreeAge, nextIssueAge, readLatency;
};

// A merlin.op_pairs.v1 record admits stableGap <= gap <= lastGap when operands satisfy every relation entry.
struct OpPairFact {
  std::map<std::string, std::string> relation;
  int stableGap = 0, lastGap = 0;
};

// Assertion-checked pair/chain results pinned to their source facts; capacity
// bounds live commands through last_event_age or until next_issue_age.
struct RTLPairFacts {
  std::string sha256;
  std::map<std::pair<std::string, std::string>, std::vector<OpPairFact>> pairs;
  std::map<std::string, int> capacity;
  std::map<std::string, std::string> window;
};

// SHA-pinned RTL ages fill compiler footprints; unresolved or structurally mismatched blocks are rejected.
class RTLEvidence {
public:
  // ScalarCore fetches with a 15-bit word index.
  static constexpr unsigned maximumProgramWords() { return 32768; }
  const std::string &factsSha256() const { return sha256; }
  const std::string &hardwareIRSha256() const { return hardwareIR; }
  // dma=wait: capture operands at launch; synchronous config; completion at matching wait without a latency bound.
  bool dmaWait() const { return dma; }

  // VLOAD and VSTORE hold their own path; the VPU and XLU exclude each other; an MXU command holds its MXU through its last event.
  Footprint resolve(const Instr &in, const RegValues &regs) const;
  llvm::json::Object applicability() const;

  // Pair facts must match this hardware IR and timing SHA-256 and name only resolved blocks.
  llvm::Error selectPairFacts(llvm::StringRef path, llvm::StringRef expectedSha256);
  const RTLPairFacts *pairFacts() const { return pairs.get(); }
  // The facts block of a VPU mnemonic, or null.
  static const char *vpuBlockName(const std::string &mnemonic);
  // The facts block of an MXU command (mxu<unit>.<operation>), or empty.
  static std::string mxuBlockName(const OpInfo &op);
  const OpTimingBlock *block(const std::string &name) const;

private:
  std::map<std::string, OpTimingBlock> blocks;
  std::string sha256, hardwareIR;
  bool dma = false;
  std::shared_ptr<const RTLPairFacts> pairs;
  friend llvm::Expected<RTLEvidence> loadRTLTimingFacts(llvm::StringRef, llvm::StringRef, bool);
};

llvm::Expected<RTLEvidence> loadRTLTimingFacts(llvm::StringRef path, llvm::StringRef expectedSha256, bool dmaWait);

} // namespace mlir::atlas::timing
#endif
