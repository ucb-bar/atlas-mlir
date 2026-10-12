#include "Atlas/AtlasDelayInsertion.h"
#include "Atlas/AtlasStream.h"
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

// In-order placement of one block from `entry`. Each op issues no earlier
// than its `floor` cycle, when given, and the block drains no earlier than
// `drainFloor`; `cycles` and `drainedAt` receive the placement. `tail` is the
// idle time that drains a block without a branch or halt. A nonempty floor
// marks one of several entry contexts, which diagnostics name.
LogicalResult timeBlock(const AtlasStream &s, size_t block,
                        const TimingProvider &p, const RegValues &entry,
                        ArrayRef<int> floor, int drainFloor,
                        std::vector<DelayInsertion> &before,
                        DelayInsertion &after, std::vector<int> &cycles,
                        int &drainedAt, int &tail) {
  ArrayRef<Operation *> ops = s.ops;
  ArrayRef<Instr> instrs = s.instrs;
  size_t begin = s.starts[block];
  size_t end = s.blockEnd(block);
  RegValues regs = entry;
  cycles.assign(end - begin, 0);
  tail = 0;
  // The first uncovered provider rule; checked after each placement.
  std::string fault;
  auto note = [&](const std::string &why) {
    if (!why.empty() && fault.empty())
      fault = why;
  };
  auto rule = [&](auto result) {
    note(result.error);
    return std::move(result.value);
  };
  auto reserve = [&](TimingReservations &table, const Instr &in,
                     const Footprint &f, int cycle) {
    std::string why = table.reserve(in, f, cycle);
    if (why.empty() && in.op->opClass == OpClass::DmaWait)
      why = table.onWait(in, cycle);
    note(why);
  };
  std::vector<Issued> issued;
  // Rebuilds the provider's reservation state for the placements so far.
  auto replay = [&]() -> std::unique_ptr<TimingReservations> {
    auto table = rule(p.createReservations());
    if (!table)
      note("timing provider returned no reservation state");
    else
      for (const Issued &x : issued)
        reserve(*table, instrs[x.index], x.f, x.cycle);
    return table;
  };
  std::unique_ptr<TimingReservations> table = replay();
  if (!fault.empty())
    return ops[begin]->emitOpError(fault);
  int nextFree = 0;

  auto name = [&](size_t i) { return ops[i]->getName().getStringRef().str(); };
  auto earliest = [&](const Instr &in, const Footprint &f, int cycle,
                      std::string &reason) {
    for (const Issued &x : issued) {
      Dependence d = rule(p.dependence(instrs[x.index], x.f, in, f));
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
    reserve(*table, in, f, cycle);
    issued.push_back({i, f, cycle});
    cycles[i - begin] = cycle;
    applyScalar(in, regs);
    int gap = rule(p.issueGap(in));
    if (gap <= 0)
      note("timing provider issue gap exceeds positive cycle domain");
    nextFree = cycle + gap;
  };
  auto search = [&](size_t i, int &cycle, std::string &reason,
                    function_ref<std::string(int)> fits) -> LogicalResult {
    for (int start = cycle;; ++cycle) {
      std::string why = fits(cycle);
      if (!fault.empty())
        return ops[i]->emitOpError(fault);
      if (why.empty())
        return success();
      reason = why;
      if (cycle - start > kMaxSearch)
        return ops[i]->emitOpError("found no free issue cycle: ") << why;
    }
  };

  for (size_t i = begin; i < end; ++i) {
    const Instr &in = instrs[i];
    Footprint f = p.footprint(in, regs);
    if (!f.error.empty())
      return emitAtlasFootprintError(s, ops[i], f.error, floor.empty() ? nullptr : &entry);
    std::string reason;
    int cycle = earliest(in, f, nextFree, reason);
    if (!floor.empty() && floor[i - begin] > cycle) {
      cycle = floor[i - begin];
      reason = "another entry context of this block issues it later";
    }

    if (in.op->opClass == OpClass::Halt) {
      // A halt neither drains in-flight work nor waits for a delay, so its
      // stall ends on a NOP, reusing one that is already there.
      for (const Issued &x : issued)
        if (x.cycle + x.f.doneAge + 1 > cycle) {
          cycle = x.cycle + x.f.doneAge + 1;
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
      if (!fault.empty())
        return ops[i]->emitOpError(fault);
      continue;
    }

    if (isControlFlow(*in.op)) {
      // The slot issues next; the successors start drained two cycles later.
      size_t slotIndex = i + 1;
      const Instr &slot = instrs[slotIndex];
      RegValues after = regs;
      applyScalar(in, after);
      Footprint sf = p.footprint(slot, after);
      if (!sf.error.empty())
        return emitAtlasFootprintError(s, ops[slotIndex], sf.error, floor.empty() ? nullptr : &entry);
      if (drained() - 2 > cycle) {
        cycle = drained() - 2;
        reason = "this block finishes before the branch's successors start";
      }
      auto fits = [&](int c) -> std::string {
        std::string why = rule(table->conflict(in, f, c));
        if (!why.empty())
          return why;
        std::string slotReason;
        int slotCycle = earliest(slot, sf, c + 1, slotReason);
        Dependence d = rule(p.dependence(in, f, slot, sf));
        if (c + d.distance > slotCycle) {
          slotCycle = c + d.distance;
          slotReason = d.reason + " after " + name(i);
        }
        if (slotCycle > c + 1)
          return "delay slot: " + slotReason;
        auto withBranch = replay();
        if (!withBranch)
          return why;
        note(withBranch->reserve(in, f, c));
        why = rule(withBranch->conflict(slot, sf, c + 1));
        return why.empty() ? why : "delay slot: " + why;
      };
      if (failed(search(i, cycle, reason, fits)))
        return failure();
      int idle = cycle - nextFree;
      place(i, f, cycle, reason);
      if (!p.redirectWaitsForDelay && idle > 0)
        before[i] = {idleDelays(idle - 1), true, reason, kRedirectGuard};
      place(slotIndex, sf, cycle + 1, "");
      if (!fault.empty())
        return ops[i]->emitOpError(fault);
      i = slotIndex;
      continue;
    }

    if (failed(search(i, cycle, reason,
                      [&](int c) { return rule(table->conflict(in, f, c)); })))
      return failure();
    place(i, f, cycle, reason);
    if (!fault.empty())
      return ops[i]->emitOpError(fault);
  }

  // A block that neither branches nor halts drains before the next block, or
  // the stream's end, as its own trailing delays; a next block starting with
  // an instruction that issues during a DELAY gets a NOP guard.
  drainedAt = drained();
  int drainCycle = std::max(drainedAt, drainFloor);
  if (!s.endsInBranch(block) && !s.endsInHalt(block) && drainCycle > nextFree) {
    int idle = tail = drainCycle - nextFree;
    after = {idleDelays(idle), false,
             s.fallsThrough(block)
                 ? "this block finishes before the next one starts"
                 : "work finishes before the stream ends"};
    if (s.fallsThrough(block)) {
      const OpInfo &next = *instrs[end].op;
      bool halt = next.opClass == OpClass::Halt;
      if (halt || (isControlFlow(next) && !p.redirectWaitsForDelay))
        after = {idleDelays(idle - 1), true, after.reason, halt ? kHaltGuard : kRedirectGuard};
    }
  }
  return success();
}

constexpr int kMaxContextRounds = 64;

} // namespace

LogicalResult mlir::atlas::atlasInOrderBlockDelays(
    const AtlasStream &s, size_t block, const TimingProvider &p,
    std::vector<DelayInsertion> &before, DelayInsertion &after, int &tail) {
  std::vector<AtlasContextGroup> groups = atlasBlockContexts(s, block, p);
  std::vector<int> cycles;
  int drained = 0;
  if (groups.size() == 1)
    return timeBlock(s, block, p, groups.front().entries.front(), {}, 0, before, after, cycles, drained, tail);
  // Raise each op's issue cycle to the latest any context needs until one
  // round leaves every context's placement unchanged, so the delays hold
  // under all of them.
  size_t begin = s.starts[block], end = s.blockEnd(block);
  std::vector<int> floor(end - begin, 0);
  int drainFloor = 0;
  for (int round = 0; round < kMaxContextRounds; ++round) {
    bool stable = true;
    for (const AtlasContextGroup &group : groups) {
      for (size_t i = begin; i < end; ++i)
        before[i] = {};
      after = {};
      if (failed(timeBlock(s, block, p, group.entries.front(), floor, drainFloor, before, after, cycles, drained, tail)))
        return failure();
      for (size_t k = 0; k < floor.size(); ++k)
        if (cycles[k] > floor[k]) {
          floor[k] = cycles[k];
          stable = false;
        }
      if (drained > drainFloor) {
        drainFloor = drained;
        stable = false;
      }
    }
    if (stable)
      return success();
  }
  return s.ops[begin]->emitOpError("delays do not converge over the block's ")
         << groups.size() << " entry contexts";
}

namespace {

LogicalResult insertDelays(ModuleOp module, StringRef requested) {
  FailureOr<TimingProvider> provider = selectAtlasTimingProvider(module, requested);
  if (failed(provider))
    return failure();
  FailureOr<AtlasStream> stream = readAtlasStream(module);
  if (failed(stream) || failed(checkAtlasStream(*stream, *provider)))
    return failure();
  std::vector<DelayInsertion> before(stream->ops.size());
  std::vector<DelayInsertion> after(stream->starts.size());
  int tail;
  for (size_t b = 0; b < stream->starts.size(); ++b)
    if (failed(atlasInOrderBlockDelays(*stream, b, *provider, before, after[b], tail)))
      return failure();
  std::vector<size_t> order(stream->ops.size());
  std::iota(order.begin(), order.end(), 0);
  return writeAtlasStream(module, *stream, order, before, after, provider->id);
}

struct InsertAtlasDelaysPass
    : PassWrapper<InsertAtlasDelaysPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertAtlasDelaysPass)
  InsertAtlasDelaysPass() = default;
  InsertAtlasDelaysPass(const InsertAtlasDelaysPass &other) : PassWrapper(other) {}
  Option<std::string> provider{*this, "provider",
                               llvm::cl::desc("Registered timing provider id (default: the retained atlas.timing_provider, else the RTL provider over selected evidence, else npu-model-rtl-match-v1)"),
                               llvm::cl::init("")};

  StringRef getArgument() const final { return "insert-atlas-delays"; }
  StringRef getDescription() const final {
    return "Insert the minimum in-order delays of the selected timing "
           "provider (default: the npu_model rtl-match model, ported from "
           "atlas-compiler-experiments) into a stream without atlas.delay";
  }

  void runOnOperation() override {
    if (failed(insertDelays(getOperation(), provider.getValue())))
      signalPassFailure();
  }
};

} // namespace

void mlir::atlas::registerInsertAtlasDelaysPass() {
  PassRegistration<InsertAtlasDelaysPass>();
}
