#include "Atlas/AtlasTarget.h"
#include "Atlas/AtlasPackLoop.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

bool mlir::atlas::isAtlasPackRelayoutLoop(ArrayRef<Operation *> ops,
                                         const AtlasLoop &loop) {
  auto source = contractTag(ops[loop.branch], kAtlasTagCFGSource);
  if (!source)
    return false;
  for (size_t pc = loop.begin; pc < loop.end(); ++pc) {
    Operation *op = ops[pc];
    if (isa<DelayOp>(op) || (pc > loop.branch && isNop(op)))
      continue;
    auto helper = op->getAttrOfType<StringAttr>(kAtlasTagCFGHelper);
    if (!helper || helper.getValue() != "pack" ||
        contractTag(op, kAtlasTagCFGSource) != source)
      return false;
  }
  return true;
}

namespace mlir::atlas::timing {
struct AtlasPackLoopTimingProofBuilder {
  static TimingContext make(const Instr &in, std::shared_ptr<const uint8_t> region,
                            const std::array<uint32_t, 32> &addresses) {
    return TimingContext(new InstructionTimingContext(in, std::move(region), addresses));
  }
};
} // namespace mlir::atlas::timing

std::pair<TimingContext, TimingContext>
mlir::atlas::atlasPackRelayoutCopyContexts(const Instr &first, const Instr &second) {
  std::array<uint32_t, 32> low, high;
  for (uint32_t row = 0; row < 32; ++row) {
    low[row] = 16 * row;
    high[row] = 512 + 16 * row;
  }
  auto region = std::make_shared<const uint8_t>(0);
  return {AtlasPackLoopTimingProofBuilder::make(first, region, low),
          AtlasPackLoopTimingProofBuilder::make(second, region, high)};
}

std::vector<TimingContext> mlir::atlas::proveAtlasPackLoopTimingContexts(
    const AtlasStream &stream, const AtlasLoop &loop, const RegValues &entry) {
  std::vector<TimingContext> result(loop.end() - loop.begin);
  if (!isAtlasPackRelayoutLoop(stream.ops, loop))
    return result;
  unsigned loads = 0, stores = 0;
  for (size_t pc = loop.begin; pc < loop.end(); ++pc) {
    loads += stream.instrs[pc].op->name == "lw";
    stores += stream.instrs[pc].op->name == "sw";
  }
  // Grouped rows retain the same bounded tile copy. Count actual decoded
  // words rather than trusting a lowering attribute to select its trip count.
  unsigned rowsPerIteration = loads / kPackRelayoutRowWords;
  bool completeGroupedBody = loads && loads == stores &&
      loads % kPackRelayoutRowWords == 0 && rowsPerIteration <= kAtlasMregRows &&
      kAtlasMregRows % rowsPerIteration == 0;
  std::vector<std::vector<uint32_t>> addresses(result.size());
  RegValues regs = entry;
  unsigned iterations = 0;
  bool known = true;
  auto step = [&](size_t pc) {
    if (pc == loop.begin) ++iterations;
    const Instr &in = stream.instrs[pc];
    if (in.op->opClass == OpClass::ScalarLoad || in.op->opClass == OpClass::ScalarStore) {
      if ((in.op->name != "lw" && in.op->name != "sw") || in.rs1 < 0 || in.rs1 >= 32 || !regs[in.rs1]) {
        known = false;
      } else {
        // Retain low bits before alignment, and reject address truncation/wrap.
        int64_t raw = int64_t(*regs[in.rs1]) + signExtend(in.imm, 12);
        uint32_t effective = uint32_t(raw) & 0x1fffff;
        if (raw < 0 || uint64_t(raw) != effective || effective % 4 ||
            uint64_t(effective) + 4 > uint64_t(kAtlasVmemBytes))
          known = false;
        addresses[pc - loop.begin].push_back(effective);
      }
    }
    applyScalar(in, regs);
    return success();
  };
  auto exit = runAtlasLoop(stream.ops, loop, [&]() -> const RegValues & { return regs; }, step);
  // Preserve the existing exact 32-case proof (including isolated scalar-pair
  // witnesses). A shorter replay additionally proves a complete grouped copy.
  if (failed(exit) || *exit != AtlasLoopExit::Exited || !known ||
      (iterations != kAtlasMregRows &&
       (!completeGroupedBody || iterations != kAtlasMregRows / rowsPerIteration)))
    return result;
  auto region = std::make_shared<const uint8_t>(0);
  for (size_t i = 0; i < addresses.size(); ++i) {
    if (addresses[i].size() != iterations) continue;
    std::array<uint32_t, 32> cases;
    // Every paired access repeats the same decided iteration ordinal. The
    // fixed-width context thus adds no address cases to the finite replay.
    for (size_t ordinal = 0; ordinal < cases.size(); ++ordinal)
      cases[ordinal] = addresses[i][ordinal % iterations];
    result[i] = AtlasPackLoopTimingProofBuilder::make(stream.instrs[loop.begin + i], region, cases);
  }
  return result;
}
