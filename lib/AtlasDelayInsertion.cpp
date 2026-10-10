#include "Atlas/AtlasDelayInsertion.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasRTLSelection.h"
#include "mlir/Pass/Pass.h"
#include <numeric>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {

struct Issued {
  size_t index;
  Footprint f;
  int cycle;
};

constexpr int kMaxSearch = 100000;
constexpr const char *kHaltGuard = "a halt does not wait for a delay";
constexpr const char *kRedirectGuard = "a redirect does not wait for a delay";

LogicalResult timeBlock(const AtlasStream &s, size_t block,
                        std::vector<DelayInsertion> &before,
                        const TargetTiming &target) {
  ArrayRef<Operation *> ops = s.ops;
  ArrayRef<Instr> instrs = s.instrs;
  size_t begin = s.starts[block];
  size_t end = s.blockEnd(block);
  RegValues regs = s.entry[block];
  ReservationTable table(target);
  std::vector<Issued> issued;
  int nextFree = 0;

  auto name = [&](size_t i) { return ops[i]->getName().getStringRef().str(); };
  auto earliest = [&](const Instr &in, const Footprint &f, int cycle,
                      std::string &reason) {
    for (const Issued &x : issued) {
      Dependence d = dependence(instrs[x.index], x.f, in, f, target);
      if (d.distance > 0 && x.cycle + d.distance > cycle) {
        cycle = x.cycle + d.distance;
        reason = d.reason + " after " + name(x.index);
      }
    }
    return cycle;
  };
  auto drained = [&] {
    int cycle = 0;
    for (const Issued &x : issued)
      cycle = std::max(cycle, x.cycle + x.f.doneAge + 1);
    return cycle;
  };
  auto place = [&](size_t i, const Footprint &f, int cycle,
                   const std::string &reason) {
    if (cycle > nextFree)
      before[i] = {idleDelays(cycle - nextFree), false, reason};
    const Instr &in = instrs[i];
    table.reserve(in, f, cycle);
    if (in.op->opClass == OpClass::DmaWait)
      table.extendForWait(cycle);
    issued.push_back({i, f, cycle});
    applyScalar(in, regs);
    nextFree = cycle + naturalGap(in);
  };
  auto search = [&](size_t i, int &cycle, std::string &reason,
                    function_ref<std::string(int)> fits) -> LogicalResult {
    for (int start = cycle;; ++cycle) {
      std::string why = fits(cycle);
      if (why.empty())
        return success();
      reason = why;
      if (cycle - start > kMaxSearch)
        return ops[i]->emitOpError("found no free issue cycle: ") << why;
    }
  };

  for (size_t i = begin; i < end; ++i) {
    const Instr &in = instrs[i];
    Footprint f = target.resolve(in, regs);
    std::string reason;
    int cycle = earliest(in, f, nextFree, reason);

    if (in.op->opClass == OpClass::Halt) {
      // A halt neither drains in-flight work nor waits for a delay, so its
      // stall ends on a NOP, reusing one that is already there.
      for (const Issued &x : issued)
        if (x.cycle + x.f.doneAge + bool(target) > cycle) {
          cycle = x.cycle + x.f.doneAge + bool(target);
          reason = "halt waits for " + name(x.index) + " to finish";
        }
      int idle = cycle - nextFree;
      if (idle > 0) {
        size_t prev = i - 1;
        bool reuse =
            i > begin && isNop(ops[prev]) && before[prev].delays.empty();
        if (reuse)
          before[prev] = {idleDelays(idle), false, reason};
        else
          before[i] = {idleDelays(idle - 1), true, reason};
        nextFree = cycle;
      }
      place(i, f, cycle, reason);
      continue;
    }

    if (isControlFlow(*in.op)) {
      // The slot issues next; the successors start drained two cycles later.
      size_t slotIndex = i + 1;
      const Instr &slot = instrs[slotIndex];
      RegValues after = regs;
      applyScalar(in, after);
      Footprint sf = target.resolve(slot, after);
      if (drained() - 2 > cycle) {
        cycle = drained() - 2;
        reason = "this block finishes before the branch's successors start";
      }
      auto fits = [&](int c) -> std::string {
        std::string why = table.conflict(in, f, c);
        if (!why.empty())
          return why;
        std::string slotReason;
        int slotCycle = earliest(slot, sf, c + 1, slotReason);
        Dependence d = dependence(in, f, slot, sf, target);
        if (c + d.distance > slotCycle) {
          slotCycle = c + d.distance;
          slotReason = d.reason + " after " + name(i);
        }
        if (slotCycle > c + 1)
          return "delay slot: " + slotReason;
        ReservationTable withBranch = table;
        withBranch.reserve(in, f, c);
        why = withBranch.conflict(slot, sf, c + 1);
        return why.empty() ? why : "delay slot: " + why;
      };
      if (failed(search(i, cycle, reason, fits)))
        return failure();
      const int idle = cycle - nextFree;
      place(i, f, cycle, reason);
      // Under selected timing a redirect does not wait for a DELAY either.
      if (target && idle > 0)
        before[i] = {idleDelays(idle - 1), true, reason, kRedirectGuard};
      place(slotIndex, sf, cycle + 1, "");
      i = slotIndex;
      continue;
    }

    if (failed(search(i, cycle, reason,
                      [&](int c) { return table.conflict(in, f, c); })))
      return failure();
    place(i, f, cycle, reason);
  }

  if (s.fallsThrough(block) && drained() > nextFree) {
    const int idle = drained() - nextFree;
    const char *why = "this block finishes before the next one starts";
    const OpInfo &next = *instrs[end].op;
    if (target && (next.opClass == OpClass::Halt || isControlFlow(next)))
      before[end] = {idleDelays(idle - 1), true, why,
                     next.opClass == OpClass::Halt ? kHaltGuard : kRedirectGuard};
    else
      before[end] = {idleDelays(idle), false, why};
  }
  return success();
}

LogicalResult insertDelays(ModuleOp module) {
  TargetTiming target;
  FailureOr<AtlasStream> stream = readTimedStream(module, target);
  if (failed(stream))
    return failure();
  std::vector<DelayInsertion> before(stream->ops.size());
  for (size_t b = 0; b < stream->starts.size(); ++b)
    if (failed(timeBlock(*stream, b, before, target)))
      return failure();
  std::vector<size_t> order(stream->ops.size());
  std::iota(order.begin(), order.end(), 0);
  if (failed(writeAtlasStream(module, *stream, order, before)))
    return failure();
  return target ? checkSelectedRTLProgramSize(module) : success();
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
