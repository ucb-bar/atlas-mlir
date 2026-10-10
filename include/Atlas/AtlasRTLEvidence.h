#ifndef ATLAS_RTL_EVIDENCE_H
#define ATLAS_RTL_EVIDENCE_H

#include "Atlas/AtlasTiming.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <map>
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
  std::map<std::string, OpTimingGroup> events;
  std::optional<int> firstFreeAge, nextIssueAge, readLatency;
};

// RTL-computed operation timing (tools/rtl_extract), pinned by the facts
// file's SHA-256. The compiler supplies footprint structure; the facts supply
// every age, count, step, hold and release. An operation without a resolved,
// structurally matching block is rejected.
class RTLEvidence {
public:
  static constexpr const char *resolverID() { return "atlas.op_timing.serialized.v1"; }
  // ScalarCore fetches with a 15-bit word index.
  static constexpr unsigned maximumProgramWords() { return 32768; }
  const std::string &factsSha256() const { return sha256; }
  const std::string &hardwareIRSha256() const { return hardwareIR; }
  // dma=wait: operands captured at launch, synchronous config, one pending
  // transfer, VMEM exclusive until its wait, no completion latency.
  bool dmaWait() const { return dma; }

  // Every engine operation also holds both VLS paths, serializing VLS, XLU
  // and VPU work.
  Footprint resolve(const Instr &in, const RegValues &regs) const;
  TargetTiming targetTiming() const;
  llvm::json::Object applicability() const;

private:
  std::map<std::string, OpTimingBlock> blocks;
  std::string sha256, hardwareIR;
  bool dma = false;
  friend llvm::Expected<RTLEvidence> loadRTLTimingFacts(llvm::StringRef,
                                                        llvm::StringRef, bool);
};

llvm::Expected<RTLEvidence> loadRTLTimingFacts(llvm::StringRef path,
                                               llvm::StringRef expectedSha256,
                                               bool dmaWait);

} // namespace mlir::atlas::timing
#endif
