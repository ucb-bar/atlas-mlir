#ifndef ATLAS_VIRTUAL_SCHEDULE_TRACE_H
#define ATLAS_VIRTUAL_SCHEDULE_TRACE_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace mlir::atlas {

// The modeled run of one block in one order. Live-value counts are indexed
// as RegisterKind: BF16, FP8, scalar.
struct ScheduleTimeline {
  // One virtual operation: its first issue and the end of its own work.
  struct Span {
    unsigned node;
    int start;
    int end;
  };
  // One machine instruction it lowers to.
  struct Instruction {
    unsigned node;
    std::string mnemonic;
    std::string engine;
    int issue;
    int end;
  };

  std::vector<unsigned> order;
  std::vector<Span> spans;
  std::vector<Instruction> instructions;
  std::vector<std::array<int, 3>> live;
  std::array<int, 3> peak{};
  int cycles = 0;
};

// What the scheduler decided for one block. Nodes are the block's schedulable
// operations in source order.
struct BlockScheduleRecord {
  unsigned index = 0;
  std::vector<std::string> labels;
  std::vector<std::vector<unsigned>> predecessors;
  std::array<int, 3> caps{};
  // "source", "latency", "pressure", or "random".
  std::string strategy = "source";
  std::optional<std::string> keptReason;
  // Absent when the function cannot be timed.
  std::optional<ScheduleTimeline> source, scheduled;
};

struct FunctionScheduleRecord {
  std::string name;
  // Set when the whole function kept its source order unexamined.
  std::optional<std::string> keptReason;
  std::vector<BlockScheduleRecord> blocks;
};

// A short name for a virtual operation: its mnemonic, the attribute that
// tells it apart, and its source line.
std::string describeVirtualOperation(Operation *op);

// --schedule-atlas-virtual's trace-file format.
void writeScheduleTrace(llvm::raw_ostream &os,
                        llvm::ArrayRef<FunctionScheduleRecord> functions);

} // namespace mlir::atlas

#endif
