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

// One merlin.op_pairs.v1 record (tools/rtl_extract/pairfacts.py): the second
// operation may issue `gap` cycles after the first for stableGap <= gap <=
// lastGap when the two commands' operands satisfy every `relation` entry.
struct OpPairFact {
  std::map<std::string, std::string> relation;
  int stableGap = 0, lastGap = 0;
};

// Assertion-checked pair and chain results selected with the facts they were
// derived from; `capacity` is the most operations of an engine observed live
// at once, each live from issue until its next_issue_age.
struct RTLPairFacts {
  std::string sha256;
  std::map<std::pair<std::string, std::string>, std::vector<OpPairFact>> pairs;
  std::map<std::string, int> capacity;
};

// RTL-computed operation timing (tools/rtl_extract), pinned by the facts
// file's SHA-256. The compiler supplies footprint structure; the facts supply
// every age, count, step, hold and release. An operation without a resolved,
// structurally matching block is rejected.
class RTLEvidence {
public:
  // ScalarCore fetches with a 15-bit word index.
  static constexpr unsigned maximumProgramWords() { return 32768; }
  const std::string &factsSha256() const { return sha256; }
  const std::string &hardwareIRSha256() const { return hardwareIR; }
  // dma=wait: operands captured at launch, synchronous config, queued
  // transfers completed only at their matching wait, no completion latency.
  bool dmaWait() const { return dma; }

  // VLOAD and VSTORE hold their own path; the VPU and XLU exclude each other;
  // an MXU command holds every engine through its last event.
  Footprint resolve(const Instr &in, const RegValues &regs) const;
  llvm::json::Object applicability() const;

  // Loads pair facts pinned by SHA-256; they must describe this hardware IR
  // and derive from these facts, and name only resolved blocks.
  llvm::Error selectPairFacts(llvm::StringRef path, llvm::StringRef expectedSha256);
  const RTLPairFacts *pairFacts() const { return pairs.get(); }
  // The facts block of a VPU mnemonic, or null.
  static const char *vpuBlockName(const std::string &mnemonic);
  const OpTimingBlock *block(const std::string &name) const;

private:
  std::map<std::string, OpTimingBlock> blocks;
  std::string sha256, hardwareIR;
  bool dma = false;
  std::shared_ptr<const RTLPairFacts> pairs;
  friend llvm::Expected<RTLEvidence> loadRTLTimingFacts(llvm::StringRef,
                                                        llvm::StringRef, bool);
};

llvm::Expected<RTLEvidence> loadRTLTimingFacts(llvm::StringRef path,
                                               llvm::StringRef expectedSha256,
                                               bool dmaWait);

} // namespace mlir::atlas::timing
#endif
