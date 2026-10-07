#include "Atlas/AtlasDelayInsertion.h"
#include "Atlas/AtlasStream.h"
#include "mlir/Pass/Pass.h"
#include <numeric>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {

LogicalResult timeBlock(const AtlasStream &s, size_t block,
                        std::vector<DelayInsertion> &before) {
  ArrayRef<Operation *> ops = s.ops;
  ArrayRef<Instr> instrs = s.instrs;
  size_t begin = s.starts[block];
  size_t end = s.blockEnd(block);
  InOrderIssue issue(s.entry[block], WaitRelease::Unknown);

  auto name = [&](size_t i) { return ops[i]->getName().getStringRef().str(); };
  auto earliest = [&](const Instr &in, const Footprint &f, int cycle,
                      std::string &reason) {
    std::optional<InOrderIssue::Binding> binding;
    cycle = issue.earliest(in, f, cycle, &binding);
    if (binding)
      reason = binding->dependence.reason + " after " + name(binding->id);
    return cycle;
  };
  auto place = [&](size_t i, const Footprint &f, int cycle,
                   const std::string &reason) {
    if (cycle > issue.nextFree())
      before[i] = {idleDelays(cycle - issue.nextFree()), false, reason};
    issue.issue(instrs[i], f, cycle, i);
  };
  auto search = [&](size_t i, int &cycle, std::string &reason,
                    function_ref<std::string(int)> fits) -> LogicalResult {
    std::optional<int> found = firstFit(cycle, fits, reason);
    if (!found)
      return ops[i]->emitOpError("found no free issue cycle: ") << reason;
    cycle = *found;
    return success();
  };

  for (size_t i = begin; i < end; ++i) {
    const Instr &in = instrs[i];
    Footprint f = footprintOf(in, issue.regs());
    std::string reason;
    int cycle = earliest(in, f, issue.nextFree(), reason);

    if (in.op->opClass == OpClass::Halt) {
      // A halt neither drains in-flight work nor waits for a delay, so its
      // stall ends on a NOP, reusing one that is already there.
      if (auto last = issue.lastToFinish(cycle)) {
        cycle = last->second;
        reason = "halt waits for " + name(last->first) + " to finish";
      }
      int idle = cycle - issue.nextFree();
      if (idle > 0) {
        size_t prev = i - 1;
        bool reuse =
            i > begin && isNop(ops[prev]) && before[prev].delays.empty();
        if (reuse)
          before[prev] = {idleDelays(idle), false, reason};
        else
          before[i] = {idleDelays(idle - 1), true, reason};
      }
      issue.issue(in, f, cycle, i);
      continue;
    }

    if (isControlFlow(*in.op)) {
      // The slot issues next; the successors start drained two cycles later.
      size_t slotIndex = i + 1;
      const Instr &slot = instrs[slotIndex];
      RegValues after = issue.regs();
      applyScalar(in, after);
      Footprint sf = footprintOf(slot, after);
      if (issue.drained() - 2 > cycle) {
        cycle = issue.drained() - 2;
        reason = "this block finishes before the branch's successors start";
      }
      auto fits = [&](int c) -> std::string {
        std::string why = issue.table().conflict(in, f, c);
        if (!why.empty())
          return why;
        std::string slotReason;
        int slotCycle = earliest(slot, sf, c + 1, slotReason);
        Dependence d = dependence(in, f, slot, sf);
        if (c + d.distance > slotCycle) {
          slotCycle = c + d.distance;
          slotReason = d.reason + " after " + name(i);
        }
        if (slotCycle > c + 1)
          return "delay slot: " + slotReason;
        ReservationTable withBranch = issue.table();
        withBranch.reserve(in, f, c);
        why = withBranch.conflict(slot, sf, c + 1);
        return why.empty() ? why : "delay slot: " + why;
      };
      if (failed(search(i, cycle, reason, fits)))
        return failure();
      place(i, f, cycle, reason);
      place(slotIndex, sf, cycle + 1, "");
      i = slotIndex;
      continue;
    }

    if (failed(search(i, cycle, reason, [&](int c) {
          return issue.table().conflict(in, f, c);
        })))
      return failure();
    place(i, f, cycle, reason);
  }

  if (s.fallsThrough(block) && issue.drained() > issue.nextFree())
    before[end] = {idleDelays(issue.drained() - issue.nextFree()), false,
                   "this block finishes before the next one starts"};
  return success();
}

LogicalResult insertDelays(ModuleOp module) {
  FailureOr<AtlasStream> stream = readAtlasStream(module);
  if (failed(stream) || failed(checkAtlasStream(*stream)))
    return failure();
  std::vector<DelayInsertion> before(stream->ops.size());
  for (size_t b = 0; b < stream->starts.size(); ++b)
    if (failed(timeBlock(*stream, b, before)))
      return failure();
  std::vector<size_t> order(stream->ops.size());
  std::iota(order.begin(), order.end(), 0);
  return writeAtlasStream(module, *stream, order, before);
}

struct InsertAtlasDelaysPass
    : PassWrapper<InsertAtlasDelaysPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertAtlasDelaysPass)

  StringRef getArgument() const final { return "insert-atlas-delays"; }
  StringRef getDescription() const final {
    return "Insert the minimum in-order delays of the npu_model rtl-match "
           "timing model, ported from atlas-compiler-experiments, into a "
           "stream without atlas.delay";
  }

  void runOnOperation() override {
    if (failed(insertDelays(getOperation())))
      signalPassFailure();
  }
};

} // namespace

void mlir::atlas::registerInsertAtlasDelaysPass() {
  PassRegistration<InsertAtlasDelaysPass>();
}
