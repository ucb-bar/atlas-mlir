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

// One event group of a merlin.op_timing.v1 block; ages count cycles after issue.
struct OpTimingGroup {
  int firstAge = 0, lastAge = 0, count = 0;
  std::optional<int> step;
};

struct OpTimingBlock {
  bool resolved = false; // events were computed (not null)
  std::map<std::string, OpTimingGroup> events;
  std::optional<int> firstFreeAge, nextIssueAge, readLatency;
};

// Operation timing computed from the RTL (tools/rtl_extract, schema
// merlin.op_timing.v1), selected by the SHA-256 of the facts file. Footprint
// structure (resources, registers, pairs) comes from the compiler; ages,
// counts, steps, holds and releases come from the mapped facts block. Any
// mnemonic without a resolved, structurally agreeing block is rejected.
class RTLEvidence {
public:
  static constexpr const char *resolverID() { return "atlas.op_timing.serialized.v1"; }
  static constexpr int resolverVersion() { return 1; }
  // Reviewed ScalarCore fetch uses a 15-bit word index into InstrMem.
  static constexpr unsigned maximumProgramWords() { return 32768; }
  const char *qualificationStatus() const { return "conditional"; }
  const std::string &factsSha256() const { return sha256; }
  const std::string &hardwareIRSha256() const { return hardwareIR; }
  // dma=wait: launch-time operand capture, synchronous config, one pending
  // transfer, VMEM exclusive until the matching wait, no completion latency.
  bool dmaWait() const { return dma; }

  // Every selected engine operation reserves BOTH VLS paths, deliberately
  // serializing vector memory, XLU and VPU work.
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
