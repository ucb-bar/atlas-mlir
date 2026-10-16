#include "Atlas/AtlasScheduling.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasStream.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <climits>
#include <numeric>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {

constexpr int kMaxIdle = 100000;

// scheduleBlock from atlas-compiler-experiments src/passes/schedule.cpp.
// Fixed-latency engines drain between blocks; DMA uses waits.
LogicalResult scheduleBlock(const AtlasStream &s, size_t block, bool generated, const TimingProvider &p,
                            std::vector<size_t> &order, std::vector<DelayInsertion> &before, int &tailIdle) {
  size_t begin = s.starts[block];
  size_t end = s.blockEnd(block);
  bool branch = s.endsInBranch(block);
  bool halt = s.endsInHalt(block);
  std::vector<Instr> nodes(s.instrs.begin() + begin, s.instrs.begin() + end);
  int n = static_cast<int>(nodes.size());
  int nb = n - (branch ? 2 : halt ? 1 : 0);
  int term = branch || halt ? nb : -1;
  int slot = branch ? nb + 1 : -1;
  auto op = [&](int i) { return s.ops[begin + i]; };
  auto name = [&](int i) { return op(i)->getName().getStringRef().str(); };

  // Keep the first provider error, including speculative checks.
  std::string fault;
  auto note = [&](const std::string &why) {
    if (!why.empty() && fault.empty()) fault = why;
  };
  auto rule = [&](auto result) {
    note(result.error);
    return std::move(result.value);
  };
  auto add = [&](int cycle, int distance, const char *message) {
    int64_t result = int64_t(cycle) + distance;
    if (result < 0 || result > INT_MAX) {
      note(message);
      return cycle;
    }
    return static_cast<int>(result);
  };

  auto built = buildTimingGraph(nodes, s.entry[block], p);
  if (!built.error.empty()) return op(0)->emitOpError(built.error);
  DepGraph g = std::move(built.value);
  if (generated) {
    // Preserve the generated DMA interval policy while allowing its permitted
    // compute to overlap. Transfers on several channels may be pending at
    // once, so a launch, or an instruction that cannot overlap a transfer,
    // stays after every wait before it, as in the source. Commands outside an
    // interval cannot move into it.
    std::vector<int> waits, outside;
    auto edge = [&](int from, int to) {
      int index = static_cast<int>(g.edges.size());
      g.edges.push_back({from, to, 1, EdgeKind::Order,
                         "generated DMA lifecycle"});
      g.out[from].push_back(index);
      g.in[to].push_back(index);
    };
    for (int i = 0; i < n; ++i) {
      if (nodes[i].op->opClass == OpClass::DmaWait) {
        waits.push_back(i);
      } else if (nodes[i].op->opClass == OpClass::DmaLoad ||
                 nodes[i].op->opClass == OpClass::DmaStore) {
        for (int wait : waits)
          edge(wait, i);
        for (int previous : outside)
          edge(previous, i);
        outside.clear();
      } else if (!canOverlapAtlasGeneratedDMA(s.ops, s.packLoops, begin + i)) {
        for (int wait : waits)
          edge(wait, i);
        outside.push_back(i);
      }
    }
  }
  auto reservations = [&]() {
    auto table = rule(p.createReservations());
    if (!table) note("timing provider returned no reservation state");
    return table;
  };
  std::unique_ptr<TimingReservations> alone = reservations();
  if (!fault.empty()) return op(0)->emitOpError(fault);
  std::vector<int> gaps(n);
  for (int i = 0; i < n; i++) {
    std::string conflict = rule(alone->conflict(nodes[i], g.footprints[i], 0));
    gaps[i] = rule(p.issueGap(nodes[i]));
    if (gaps[i] <= 0) note("timing provider issue gap exceeds positive cycle domain");
    if (!fault.empty()) return op(i)->emitOpError(fault);
    if (!conflict.empty()) return op(i)->emitOpError("can never issue: ") << conflict;
  }
  // Branch, delay slot and successor issue one cycle apart.
  if (slot >= 0 && (gaps[term] != 1 || gaps[slot] != 1))
    return op(gaps[term] != 1 ? term : slot)->emitOpError(
        "timing provider issue gap does not fit the single-cycle delay slot");
  if (slot >= 0)
    for (int e : g.out[term])
      if (g.edges[e].to == slot && g.edges[e].distance > gaps[term])
        return op(slot)->emitOpError("delay-slot dependence exceeds its fixed issue spacing: ") << g.edges[e].reason;

  // Widen critical heights to avoid overflow.
  std::vector<int64_t> height(n);
  for (int i = n - 1; i >= 0; --i) {
    height[i] = int64_t(g.footprints[i].doneAge) + 1;
    for (int e : g.out[i]) {
      const Edge &ed = g.edges[e];
      int64_t distance = ed.distance;
      if (nodes[ed.to].op->opClass == OpClass::DmaWait && nodes[ed.to].op->channel == nodes[i].op->channel)
        distance = std::max(distance, int64_t(g.footprints[i].dmaCycles) + 2);
      height[i] = std::max(height[i], distance + height[ed.to]);
    }
  }
  std::vector<int> earliest(n, 0), waitingPreds(n, 0), issue(n, -1);
  std::vector<std::string> why(n), reason(n);
  for (const Edge &e : g.edges)
    waitingPreds[e.to]++;

  // npu_model's DMA timing only decides when to issue a dma.wait; after the
  // wait the schedule assumes nothing.
  int channelRelease[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int dmaQueueEnd = 0;
  auto release = [&](int i) { return channelRelease[nodes[i].op->channel]; };
  auto isWait = [&](int i) {
    return nodes[i].op->opClass == OpClass::DmaWait;
  };

  std::unique_ptr<TimingReservations> table = reservations();
  if (!fault.empty()) return op(0)->emitOpError(fault);
  auto checkCycle = [&](int i, int c) {
    add(add(c, g.footprints[i].doneAge, "timing provider completion age exceeds cycle domain"),
        1, "timing provider completion age exceeds cycle domain");
    add(c, gaps[i], "timing provider issue gap exceeds positive cycle domain");
  };
  auto busy = [&](int i, int c) {
    checkCycle(i, c);
    if (!fault.empty()) return std::string();
    return rule(table->conflict(nodes[i], g.footprints[i], c));
  };
  auto reserve = [&](TimingReservations &state, int i, int c) {
    checkCycle(i, c);
    if (!fault.empty()) return;
    std::string error = state.reserve(nodes[i], g.footprints[i], c);
    if (error.empty() && isWait(i)) error = state.onWait(nodes[i], c);
    note(error);
  };
  std::vector<int> placedOrder;
  int cycle = 0, placed = 0, nextFree = 0, lastPlaced = 0;
  while (placed < nb) {
    // Take the ready instruction on the longest path that fits this cycle.
    int best = -1, bestWait = -1;
    bool otherWork = false;
    for (int i = 0; i < nb; i++) {
      if (issue[i] >= 0 || waitingPreds[i] > 0)
        continue;
      if (!isWait(i))
        otherWork = true;
      if (earliest[i] > cycle)
        continue;
      if (isWait(i)) {
        if (!busy(i, cycle).empty()) continue;
        if (bestWait < 0 || release(i) < release(bestWait))
          bestWait = i;
        continue;
      }
      if (best >= 0 && height[i] <= height[best])
        continue;
      if (!busy(i, cycle).empty()) continue;
      best = i;
    }
    // Prefer independent work until DMA is expected to finish, then critical
    // waits, or a wait that unlocks a DMA launch overlapping ALU work.
    bool unlocksDma = false;
    if (bestWait >= 0 && release(bestWait) == 0 && best >= 0 &&
        nodes[best].op->opClass == OpClass::Alu)
      for (int e : g.out[bestWait]) {
        int t = g.edges[e].to;
        if (t < nb && waitingPreds[t] == 1 && int64_t(earliest[t]) <= int64_t(cycle) + gaps[bestWait] &&
            nodes[t].op->engine == Engine::Dma && !isWait(t))
          unlocksDma = true;
      }
    bool readyCriticalWait =
        bestWait >= 0 && best >= 0 && height[bestWait] > height[best] &&
        ((release(bestWait) > 0 && cycle >= release(bestWait)) || unlocksDma);
    if ((best < 0 || readyCriticalWait) && bestWait >= 0 &&
        (!otherWork || cycle >= release(bestWait))) {
      // Idle cycles before a wait overlap the transfer and those after it do
      // not, so issue the wait just before its most critical user can go.
      int firstUse = INT_MAX, critical = -1;
      for (int e : g.out[bestWait]) {
        int t = g.edges[e].to;
        if (t >= nb || waitingPreds[t] != 1)
          continue;
        int c = std::max(add(cycle, gaps[bestWait], "timing provider issue gap exceeds positive cycle domain"),
                         earliest[t]);
        int start = c;
        while (fault.empty() && !busy(t, c).empty()) {
          c = add(c, 1, "scheduler search exceeds cycle domain");
          if (c - start > kMaxIdle) note("scheduler found no free issue cycle for a DMA wait user");
        }
        if (critical < 0 || height[t] > height[critical] ||
            (height[t] == height[critical] && c < firstUse)) {
          critical = t;
          firstUse = c;
        }
      }
      if (firstUse == INT_MAX || firstUse - gaps[bestWait] <= cycle)
        best = bestWait;
    }
    if (!fault.empty()) return op(0)->emitOpError(fault);
    if (best < 0) {
      cycle = add(cycle, 1, "scheduler search exceeds cycle domain");
      if (!fault.empty()) return op(0)->emitOpError(fault);
      if (cycle - lastPlaced > kMaxIdle)
        return op(0)->emitOpError("scheduler made no progress");
      continue;
    }

    if (cycle > nextFree) {
      std::string held = busy(best, cycle - 1);
      if (earliest[best] == cycle)
        reason[best] = why[best];
      else if (!held.empty()) reason[best] = held;
      else
        reason[best] = isWait(best) ? "dma.wait issues just before its user"
                                    : "list scheduler priority";
    }
    issue[best] = cycle;
    reserve(*table, best, cycle);
    placedOrder.push_back(best);
    if (g.footprints[best].dmaCycles > 0) {
      // Transfers run one at a time, in issue order.
      int latency = g.footprints[best].dmaCycles;
      dmaQueueEnd = std::max(add(cycle, latency - 1, "DMA queue estimate exceeds cycle domain"),
                             add(dmaQueueEnd, latency, "DMA queue estimate exceeds cycle domain"));
      channelRelease[nodes[best].op->channel] = add(dmaQueueEnd, 2, "DMA queue estimate exceeds cycle domain");
    }
    for (int e : g.out[best]) {
      const Edge &ed = g.edges[e];
      int ready = add(cycle, ed.distance, "timing provider issue distance exceeds cycle domain");
      if (ready > earliest[ed.to]) {
        earliest[ed.to] = ready;
        why[ed.to] = ed.reason + " after " + name(best);
      }
      waitingPreds[ed.to]--;
    }
    nextFree = add(cycle, gaps[best], "timing provider issue gap exceeds positive cycle domain");
    if (!fault.empty()) return op(best)->emitOpError(fault);
    lastPlaced = cycle;
    cycle = nextFree;
    placed++;
  }

  int drain = 0;
  for (int i = 0; i < nb; i++)
    drain = std::max(drain, add(add(issue[i], g.footprints[i].doneAge,
                                   "timing provider completion age exceeds cycle domain"),
                               1, "timing provider completion age exceeds cycle domain"));
  if (!fault.empty()) return op(0)->emitOpError(fault);

  int termCycle = nextFree;
  std::string termReason;
  if (term >= 0) {
    auto raise = [&](int c, const std::string &why) {
      if (c > termCycle) {
        termCycle = c;
        termReason = why;
      }
    };
    raise(earliest[term], why[term]);
    // With a delay slot, the next block starts two cycles after the branch.
    raise(halt ? drain : drain - 2,
          halt ? "halt waits for this block to finish"
               : "this block finishes before the branch's successors start");
    if (slot >= 0)
      raise(earliest[slot] - 1, "delay slot: " + why[slot]);
    for (;;) {
      std::string held = busy(term, termCycle);
      if (held.empty() && slot >= 0 && fault.empty()) {
        int slotCycle = add(termCycle, gaps[term], "timing provider issue gap exceeds positive cycle domain");
        // Replay placements to check the slot with the branch reserved.
        auto withBranch = reservations();
        if (withBranch && fault.empty()) {
          for (int i : placedOrder)
            reserve(*withBranch, i, issue[i]);
          checkCycle(slot, slotCycle);
          if (fault.empty()) reserve(*withBranch, term, termCycle);
          if (fault.empty()) held = rule(withBranch->conflict(nodes[slot], g.footprints[slot], slotCycle));
          if (!held.empty()) held = "delay slot: " + held;
        }
      }
      if (!fault.empty()) return op(term)->emitOpError(fault);
      if (held.empty()) break;
      termCycle = add(termCycle, 1, "scheduler search exceeds cycle domain");
      if (!fault.empty()) return op(term)->emitOpError(fault);
      termReason = held;
      if (termCycle - nextFree > kMaxIdle) return op(term)->emitOpError("found no free issue cycle: ") << held;
    }
    reserve(*table, term, termCycle);
    int termEnd = add(termCycle, gaps[term], "timing provider issue gap exceeds positive cycle domain");
    if (slot >= 0) {
      reserve(*table, slot, termEnd);
      add(termEnd, gaps[slot], "timing provider issue gap exceeds positive cycle domain");
    }
    if (!fault.empty()) return op(term)->emitOpError(fault);
  }

  std::vector<int> body(nb);
  std::iota(body.begin(), body.end(), 0);
  std::stable_sort(body.begin(), body.end(),
                   [&](int a, int b) { return issue[a] < issue[b]; });
  int free = 0;
  for (int i : body) {
    if (issue[i] > free)
      before[begin + i] = {idleDelays(issue[i] - free), false, reason[i]};
    order.push_back(begin + i);
    free = add(issue[i], gaps[i], "timing provider issue gap exceeds positive cycle domain");
  }
  if (term >= 0) {
    int idle = termCycle - free;
    size_t last = nb > 0 ? order.back() : 0;
    if (idle > 0 && halt && nb > 0 && isNop(s.ops[last]) &&
        before[last].delays.empty())
      before[last] = {idleDelays(idle), false, termReason};
    else if (idle > 0 && halt)
      before[begin + term] = {idleDelays(idle - 1), true, termReason};
    else if (idle > 0)
      before[begin + term] = {idleDelays(idle), false, termReason};
    order.push_back(begin + term);
    if (slot >= 0)
      order.push_back(begin + slot);
  } else {
    // No branch or halt: the block drains before the next one, or the
    // stream's end.
    tailIdle = std::max(0, drain - free);
  }
  return success();
}

LogicalResult scheduleStream(ModuleOp module, bool insertDelays, StringRef requested) {
  FailureOr<TimingProvider> provider = selectAtlasTimingProvider(module, requested);
  if (failed(provider))
    return failure();
  FailureOr<AtlasStream> stream = readAtlasStream(module);
  if (failed(stream) || failed(checkAtlasStream(*stream, *provider)))
    return failure();
  auto kind = classifyAtlasGeneratedArtifact(module);
  if (failed(kind))
    return failure();
  std::vector<size_t> order;
  std::vector<DelayInsertion> before(stream->ops.size());
  std::vector<int> tailIdle(stream->starts.size(), 0);
  for (size_t b = 0; b < stream->starts.size(); ++b)
    if (failed(scheduleBlock(*stream, b, *kind == AtlasArtifactKind::Generated, *provider, order, before, tailIdle[b])))
      return failure();
  // A block without a branch or halt drains as its own trailing delays, so a
  // branch into the next block does not run them.
  std::vector<DelayInsertion> after(stream->starts.size());
  for (size_t b = 0; b < stream->starts.size(); ++b)
    if (tailIdle[b] > 0)
      after[b] = {idleDelays(tailIdle[b]), false,
                  stream->fallsThrough(b)
                      ? "this block finishes before the next one starts"
                      : "work finishes before the stream ends"};
  if (!insertDelays) {
    before.assign(stream->ops.size(), DelayInsertion{});
    after.clear();
  }
  if (failed(writeAtlasStream(module, *stream, order, before, after,
                              insertDelays ? StringRef(provider->id) : StringRef())))
    return failure();
  if (insertDelays && provider->id != kNpuModelTimingProviderId)
    return verifyAtlasTiming(module, *provider);
  return success();
}

struct ScheduleAtlasStreamPass
    : PassWrapper<ScheduleAtlasStreamPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleAtlasStreamPass)
  ScheduleAtlasStreamPass() = default;
  ScheduleAtlasStreamPass(const ScheduleAtlasStreamPass &other) : PassWrapper(other) {}
  Option<bool> insertDelays{*this, "insert-delays",
                           llvm::cl::desc("Insert timing delays after reordering"),
                           llvm::cl::init(true)};
  Option<std::string> provider{*this, "provider",
                               llvm::cl::desc("Registered timing provider id (default: the retained atlas.timing_provider, else npu-model-rtl-match-v1)"),
                               llvm::cl::init("")};

  StringRef getArgument() const final { return "schedule-atlas-stream"; }
  StringRef getDescription() const final {
    return "Reorder each basic block with the list scheduler ported from "
           "atlas-compiler-experiments and insert the delays it needs";
  }

  void runOnOperation() override {
    if (failed(scheduleStream(getOperation(), insertDelays, provider.getValue())))
      signalPassFailure();
  }
};

} // namespace

void mlir::atlas::registerScheduleAtlasStreamPass() {
  PassRegistration<ScheduleAtlasStreamPass>();
}
