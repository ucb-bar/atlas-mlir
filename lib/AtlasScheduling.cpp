#include "Atlas/AtlasScheduling.h"
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
LogicalResult scheduleBlock(const AtlasStream &s, size_t block,
                            uint32_t dmaRegs, bool generated,
                            std::vector<size_t> &order,
                            std::vector<DelayInsertion> &before,
                            int &tailIdle) {
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

  DepGraph g = buildGraph(nodes, s.entry[block], dmaRegs);
  if (generated) {
    // Preserve the generated DMA interval policy while allowing its permitted
    // compute to overlap. Commands outside an interval cannot move into it.
    int wait = -1;
    std::vector<int> outside;
    auto edge = [&](int from, int to) {
      int index = static_cast<int>(g.edges.size());
      g.edges.push_back({from, to, 1, EdgeKind::Order,
                         "generated DMA lifecycle"});
      g.out[from].push_back(index);
      g.in[to].push_back(index);
    };
    for (int i = 0; i < n; ++i) {
      if (nodes[i].op->opClass == OpClass::DmaWait) {
        wait = i;
      } else if (nodes[i].op->opClass == OpClass::DmaLoad ||
                 nodes[i].op->opClass == OpClass::DmaStore) {
        if (wait >= 0)
          edge(wait, i);
        for (int previous : outside)
          edge(previous, i);
        outside.clear();
      } else if (!canOverlapAtlasGeneratedDMA(op(i))) {
        if (wait >= 0)
          edge(wait, i);
        outside.push_back(i);
      }
    }
  }
  for (int i = 0; i < n; i++) {
    std::string alone =
        ReservationTable().conflict(nodes[i], g.footprints[i], 0);
    if (!alone.empty())
      return op(i)->emitOpError("can never issue: ") << alone;
  }

  std::vector<int> height = criticalHeights(g);
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

  ReservationTable table;
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
        if (bestWait < 0 || release(i) < release(bestWait))
          bestWait = i;
        continue;
      }
      if (best >= 0 && height[i] <= height[best])
        continue;
      if (!table.conflict(nodes[i], g.footprints[i], cycle).empty())
        continue;
      best = i;
    }
    // Prefer independent work until DMA is expected to finish, then critical
    // waits, or a wait that unlocks a DMA launch overlapping ALU work.
    bool unlocksDma = false;
    if (bestWait >= 0 && release(bestWait) == 0 && best >= 0 &&
        nodes[best].op->opClass == OpClass::Alu)
      for (int e : g.out[bestWait]) {
        int t = g.edges[e].to;
        if (t < nb && waitingPreds[t] == 1 && earliest[t] <= cycle + 1 &&
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
        int c = std::max(cycle + 1, earliest[t]);
        while (!table.conflict(nodes[t], g.footprints[t], c).empty())
          c++;
        if (critical < 0 || height[t] > height[critical] ||
            (height[t] == height[critical] && c < firstUse)) {
          critical = t;
          firstUse = c;
        }
      }
      if (firstUse == INT_MAX || firstUse - 1 <= cycle)
        best = bestWait;
    }
    if (best < 0) {
      cycle++;
      if (cycle - lastPlaced > kMaxIdle)
        return op(0)->emitOpError("scheduler made no progress");
      continue;
    }

    if (cycle > nextFree) {
      std::string busy =
          table.conflict(nodes[best], g.footprints[best], cycle - 1);
      if (earliest[best] == cycle)
        reason[best] = why[best];
      else if (!busy.empty())
        reason[best] = busy;
      else
        reason[best] = isWait(best) ? "dma.wait issues just before its user"
                                    : "list scheduler priority";
    }
    issue[best] = cycle;
    table.reserve(nodes[best], g.footprints[best], cycle);
    if (isWait(best))
      table.extendForWait(cycle);
    if (g.footprints[best].dmaCycles > 0) {
      // Transfers run one at a time, in issue order.
      int latency = g.footprints[best].dmaCycles;
      dmaQueueEnd = std::max(cycle + latency - 1, dmaQueueEnd + latency);
      channelRelease[nodes[best].op->channel] = dmaQueueEnd + 2;
    }
    for (int e : g.out[best]) {
      const Edge &ed = g.edges[e];
      if (cycle + ed.distance > earliest[ed.to]) {
        earliest[ed.to] = cycle + ed.distance;
        why[ed.to] = ed.reason + " after " + name(best);
      }
      waitingPreds[ed.to]--;
    }
    nextFree = cycle + naturalGap(nodes[best]);
    lastPlaced = cycle;
    cycle = nextFree;
    placed++;
  }

  int drain = 0;
  for (int i = 0; i < nb; i++)
    drain = std::max(drain, issue[i] + g.footprints[i].doneAge + 1);

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
      std::string busy =
          table.conflict(nodes[term], g.footprints[term], termCycle);
      if (busy.empty() && slot >= 0) {
        busy = table.conflict(nodes[slot], g.footprints[slot], termCycle + 1);
        if (!busy.empty())
          busy = "delay slot: " + busy;
      }
      if (busy.empty())
        break;
      termCycle++;
      termReason = busy;
    }
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
    free = issue[i] + naturalGap(nodes[i]);
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
  } else if (s.fallsThrough(block)) {
    tailIdle = std::max(0, drain - free);
  }
  return success();
}

LogicalResult scheduleStream(ModuleOp module, bool insertDelays) {
  FailureOr<AtlasStream> stream = readAtlasStream(module);
  if (failed(stream) || failed(checkAtlasStream(*stream, npuModelTimingProvider())))
    return failure();
  uint32_t dmaRegs = dmaOperandRegisters(stream->instrs);
  std::vector<size_t> order;
  std::vector<DelayInsertion> before(stream->ops.size());
  std::vector<int> tailIdle(stream->starts.size(), 0);
  for (size_t b = 0; b < stream->starts.size(); ++b)
    if (failed(scheduleBlock(*stream, b, dmaRegs,
                             module->hasAttr("atlas.generated_from_virtual"),
                             order, before, tailIdle[b])))
      return failure();
  // A block that falls through finishes before the next block's first op.
  for (size_t b = 0; b + 1 < stream->starts.size(); ++b) {
    if (tailIdle[b] == 0)
      continue;
    DelayInsertion &next = before[order[stream->starts[b + 1]]];
    std::vector<uint32_t> delays = idleDelays(tailIdle[b]);
    delays.insert(delays.end(), next.delays.begin(), next.delays.end());
    next.delays = delays;
    std::string reason = "this block finishes before the next one starts";
    next.reason = next.reason.empty() ? reason : reason + "; " + next.reason;
  }
  if (!insertDelays)
    before.assign(stream->ops.size(), DelayInsertion{});
  return writeAtlasStream(module, *stream, order, before, insertDelays);
}

struct ScheduleAtlasStreamPass
    : PassWrapper<ScheduleAtlasStreamPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleAtlasStreamPass)
  ScheduleAtlasStreamPass() = default;
  ScheduleAtlasStreamPass(const ScheduleAtlasStreamPass &other) : PassWrapper(other) {}
  Option<bool> insertDelays{*this, "insert-delays",
                           llvm::cl::desc("Insert timing delays after reordering"),
                           llvm::cl::init(true)};

  StringRef getArgument() const final { return "schedule-atlas-stream"; }
  StringRef getDescription() const final {
    return "Reorder each basic block with the list scheduler ported from "
           "atlas-compiler-experiments and insert the delays it needs";
  }

  void runOnOperation() override {
    if (failed(scheduleStream(getOperation(), insertDelays)))
      signalPassFailure();
  }
};

} // namespace

void mlir::atlas::registerScheduleAtlasStreamPass() {
  PassRegistration<ScheduleAtlasStreamPass>();
}
