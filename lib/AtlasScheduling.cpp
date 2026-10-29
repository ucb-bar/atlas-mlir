#include "Atlas/AtlasScheduling.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasModelTables.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasVerificationContext.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <climits>
#include <numeric>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {

constexpr int kMaxIdle = 100000;

void constrainSourceWaits(DepGraph &g) {
  auto edge = [&](int from, int to) {
    int index = static_cast<int>(g.edges.size());
    g.edges.push_back({from, to, 1, EdgeKind::Order, "source DMA wait placement"});
    g.out[from].push_back(index);
    g.in[to].push_back(index);
  };
  int n = static_cast<int>(g.nodes.size());
  for (int wait = 0; wait < n; ++wait) {
    if (g.nodes[wait].op->opClass == OpClass::DmaWait) {
      for (int i = 0; i < wait; ++i)
        edge(i, wait);
      for (int i = wait + 1; i < n; ++i)
        edge(wait, i);
    }
  }
}

// scheduleBlock from atlas-compiler-experiments src/passes/schedule.cpp.
// Fixed-latency engines drain between blocks; DMA uses waits.
LogicalResult scheduleBlock(const AtlasStream &s, size_t block, bool generated,
                            bool sourceWaitPlacement,
                            const TimingProvider &p, bool fillSlots, std::vector<size_t> &order,
                            std::vector<DelayInsertion> &before, int &tailIdle) {
  size_t begin = s.starts[block];
  size_t end = s.blockEnd(block);
  bool branch = s.endsInBranch(block);
  bool halt = s.endsInHalt(block);
  // A pending delay slot is timed as a NOP after the block's ops, and when
  // filled may take one of them once they are scheduled.
  bool pendingSlot = branch && !s.delaySlots;
  std::vector<Instr> nodes(s.instrs.begin() + begin, s.instrs.begin() + end);
  if (pendingSlot)
    nodes.push_back(atlasNopInstr());
  if (nodes.size() > INT_MAX)
    return s.ops[begin]->emitOpError("block has too many instructions");
  int n = static_cast<int>(nodes.size());
  int nb = n - (branch ? 2 : halt ? 1 : 0);
  int term = branch || halt ? nb : -1;
  int slot = branch ? nb + 1 : -1;
  auto op = [&](int i) { return s.ops[std::min(begin + i, end - 1)]; };
  auto name = [&](int i) { return op(i)->getName().getStringRef().str(); };

  const TimingTables *tables = p.model.get();
  DepGraph g = buildGraph(nodes, s.entry[block], tables);
  if (sourceWaitPlacement)
    constrainSourceWaits(g);
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
    for (int i = 0; i < static_cast<int>(end - begin); ++i) {
      if (nodes[i].op->opClass == OpClass::DmaWait) {
        waits.push_back(i);
      } else if (nodes[i].op->opClass == OpClass::DmaLoad ||
                 nodes[i].op->opClass == OpClass::DmaStore) {
        for (int wait : waits)
          edge(wait, i);
        for (int previous : outside)
          edge(previous, i);
        outside.clear();
      } else if (!canOverlapAtlasGeneratedDMA(s.ops, s.loops, begin + i)) {
        for (int wait : waits)
          edge(wait, i);
        outside.push_back(i);
      }
    }
  }
  ReservationTable alone(p.model);
  if (p.frontendSpacing.ordinaryGap <= 0 || p.frontendSpacing.delaySlots < 0 ||
      (int64_t(p.frontendSpacing.delaySlots) + 1) *
              p.frontendSpacing.ordinaryGap > INT_MAX)
    return op(0)->emitOpError("timing provider frontend spacing exceeds positive cycle domain");
  int maxSpan = p.frontendSpacing.successorGap();
  std::vector<int> gaps(n);
  for (int i = 0; i < n; i++) {
    std::string conflict = alone.conflict(nodes[i], g.footprints[i], 0);
    gaps[i] = tables ? naturalGap(nodes[i], *tables) : naturalGap(nodes[i]);
    if (!conflict.empty())
      return op(i)->emitOpError("can never issue: ") << conflict;
    maxSpan = std::max({maxSpan, gaps[i], g.footprints[i].doneAge + 1});
  }
  for (const Edge &edge : g.edges)
    maxSpan = std::max(maxSpan, edge.distance);
  // Each placement can advance by a dependence, an issue gap, or the idle
  // limit. Leave room for speculative users, the terminator, and its slot.
  if ((int64_t(n) + 3) * (int64_t(maxSpan) + kMaxIdle + 1) > INT_MAX)
    return op(0)->emitOpError("block schedule exceeds cycle domain");
  // Branch, delay slot and successor issue one cycle apart.
  if (slot >= 0 && (gaps[term] != p.frontendSpacing.ordinaryGap || gaps[slot] != p.frontendSpacing.ordinaryGap))
    return op(gaps[term] != p.frontendSpacing.ordinaryGap ? term : slot)->emitOpError(
        "timing provider issue gap does not fit the single-cycle delay slot");
  if (slot >= 0)
    for (int e : g.out[term])
      if (g.edges[e].to == slot && g.edges[e].distance > gaps[term])
        return op(slot)->emitOpError("delay-slot dependence exceeds its fixed issue spacing: ") << g.edges[e].reason;

  auto heights = criticalHeights(g, p);
  if (!heights.error.empty())
    return op(0)->emitOpError(p.diagnostic(heights.error));
  std::vector<int64_t> height = std::move(heights.value);
  std::vector<int> earliest(n, 0), waitingPreds(n, 0), issue(n, -1);
  std::vector<std::string> why(n), reason(n);
  for (const Edge &e : g.edges)
    waitingPreds[e.to]++;

  // npu_model's DMA timing only decides when to issue a dma.wait; after the
  // wait the schedule assumes nothing.
  int channelRelease[kAtlasDMAChannels] = {};
  std::optional<int> dmaQueueEnd;
  auto release = [&](int i) { return channelRelease[nodes[i].op->channel]; };
  auto isWait = [&](int i) {
    return nodes[i].op->opClass == OpClass::DmaWait;
  };

  ReservationTable table(p.model);
  auto completion = [&](int i, int c) { return c + g.footprints[i].doneAge + 1; };
  auto nextIssue = [&](int i, int c) { return c + gaps[i]; };
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
        if (!table.conflict(nodes[i], g.footprints[i], cycle).empty())
          continue;
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
        if (t < nb && waitingPreds[t] == 1 &&
            earliest[t] <= cycle + gaps[bestWait] &&
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
        int c = std::max(nextIssue(bestWait, cycle), earliest[t]);
        int start = c;
        while (!table.conflict(nodes[t], g.footprints[t], c).empty()) {
          c++;
          if (c - start > kMaxIdle)
            return op(t)->emitOpError("scheduler found no free issue cycle for a DMA wait user");
        }
        if (critical < 0 || height[t] > height[critical] ||
            (height[t] == height[critical] && c < firstUse)) {
          critical = t;
          firstUse = c;
        }
      }
      // Without a non-terminator successor for which this wait is the last
      // missing predecessor, defer it while graph-ready non-wait work remains,
      // even if that work must wait for an engine or register.
      if ((firstUse == INT_MAX && !otherWork) ||
          firstUse - gaps[bestWait] <= cycle)
        best = bestWait;
    }
    if (best < 0) {
      cycle++;
      if (cycle - lastPlaced > kMaxIdle)
        return op(0)->emitOpError("scheduler made no progress");
      continue;
    }

    if (cycle > nextFree) {
      std::string held = table.conflict(nodes[best], g.footprints[best], cycle - 1);
      if (earliest[best] == cycle)
        reason[best] = why[best];
      else if (!held.empty()) reason[best] = held;
      else
        reason[best] = isWait(best) ? "dma.wait issues just before its user"
                                    : "list scheduler priority";
    }
    issue[best] = cycle;
    table.reserve(nodes[best], g.footprints[best], cycle);
    if (isWait(best))
      table.extendForWait(cycle);
    if (g.footprints[best].dmaCycles > 0) {
      if (!p.dmaProposal)
        return op(best)->emitOpError("timing provider lacks DMA proposal policy");
      auto proposal = p.dmaProposal(g.footprints[best].dmaCycles, cycle, dmaQueueEnd);
      if (!proposal.error.empty())
        return op(best)->emitOpError(p.diagnostic(proposal.error));
      dmaQueueEnd = proposal.value.queueEnd;
      channelRelease[nodes[best].op->channel] = proposal.value.waitIssue;
    }
    for (int e : g.out[best]) {
      const Edge &ed = g.edges[e];
      int ready = cycle + ed.distance;
      if (ready > earliest[ed.to]) {
        earliest[ed.to] = ready;
        why[ed.to] = ed.reason + " after " + name(best);
      }
      waitingPreds[ed.to]--;
    }
    nextFree = nextIssue(best, cycle);
    lastPlaced = cycle;
    cycle = nextFree;
    placed++;
  }

  std::vector<int> body(nb);
  std::iota(body.begin(), body.end(), 0);
  std::stable_sort(body.begin(), body.end(),
                   [&](int a, int b) { return issue[a] < issue[b]; });
  if (pendingSlot && fillSlots) {
    // The slot takes the last-issued scalar ALU work that nothing after it
    // depends on: it still runs after the rest of the block, on both paths.
    auto filler = llvm::find_if(llvm::reverse(body), [&](int i) {
      return isAtlasDelaySlotWork(op(i)) && g.out[i].empty() &&
             g.footprints[i].doneAge == 0 && gaps[i] == p.frontendSpacing.ordinaryGap;
    });
    if (filler != body.rend()) {
      slot = *filler;
      body.erase(std::next(filler).base());
      // Keep the chosen order, but reclaim the filler's old issue cycle.
      table = ReservationTable(p.model);
      nextFree = 0;
      earliest.assign(n, 0);
      why.assign(n, "");
      for (int i : body) {
        cycle = std::max(nextFree, earliest[i]);
        reason[i] = cycle > nextFree ? why[i] : "";
        for (;;) {
          std::string busy = table.conflict(nodes[i], g.footprints[i], cycle);
          if (busy.empty())
            break;
          cycle++;
          reason[i] = busy;
          if (cycle - nextFree > kMaxIdle)
            return op(i)->emitOpError("scheduler made no progress");
        }
        issue[i] = cycle;
        table.reserve(nodes[i], g.footprints[i], cycle);
        if (isWait(i))
          table.extendForWait(cycle);
        for (int e : g.out[i]) {
          const Edge &ed = g.edges[e];
          int ready = cycle + ed.distance;
          if (ready > earliest[ed.to]) {
            earliest[ed.to] = ready;
            why[ed.to] = ed.reason + " after " + name(i);
          }
        }
        nextFree = nextIssue(i, cycle);
      }
    }
  }

  // Inclusive issue-edge-zero last events are visible to Core S1 only
  // on the next edge (S1 precedes this edge's EXUs, including for ECALL).
  int drain = 0;
  for (int i : body)
    drain = std::max(drain, completion(i, issue[i]));

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
    raise(halt ? drain : drain - p.frontendSpacing.successorGap(),
          halt ? "halt waits for this block to finish"
               : "this block finishes before the branch's successors start");
    if (slot >= 0)
      raise(earliest[slot] - p.frontendSpacing.ordinaryGap, "delay slot: " + why[slot]);
    for (;;) {
      std::string held = table.conflict(nodes[term], g.footprints[term], termCycle);
      if (held.empty() && slot >= 0) {
        int slotCycle = nextIssue(term, termCycle);
        // Replay placements to check the slot with the branch reserved.
        ReservationTable withBranch(p.model);
        for (int i : body) {
          withBranch.reserve(nodes[i], g.footprints[i], issue[i]);
          if (isWait(i))
            withBranch.extendForWait(issue[i]);
        }
        withBranch.reserve(nodes[term], g.footprints[term], termCycle);
        held = withBranch.conflict(nodes[slot], g.footprints[slot], slotCycle);
        if (!held.empty())
          held = "delay slot: " + held;
      }
      if (held.empty())
        break;
      termCycle++;
      termReason = held;
      if (termCycle - nextFree > kMaxIdle)
        return op(term)->emitOpError("found no free issue cycle: ") << held;
    }
    table.reserve(nodes[term], g.footprints[term], termCycle);
    int termEnd = nextIssue(term, termCycle);
    if (slot >= 0) {
      table.reserve(nodes[slot], g.footprints[slot], termEnd);
    }
  }

  int free = 0;
  for (int i : body) {
    if (issue[i] > free)
      before[begin + i] = {idleDelays(issue[i] - free), false, reason[i]};
    order.push_back(begin + i);
    free = nextIssue(i, issue[i]);
  }
  if (term >= 0) {
    int idle = termCycle - free;
    size_t last = !body.empty() ? order.back() : 0;
    if (idle > 0 && halt && !body.empty() && isNop(s.ops[last]) &&
        before[last].delays.empty())
      before[last] = {idleDelays(idle), false, termReason};
    else if (idle > 0 && halt)
      before[begin + term] = {idleDelays(idle - 1), true, termReason};
    else if (idle > 0)
      before[begin + term] = {idleDelays(idle), false, termReason};
    order.push_back(begin + term);
    // writeAtlasStream fills a pending slot that took no op with a NOP.
    if (slot >= 0 && begin + slot < end)
      order.push_back(begin + slot);
  } else {
    // No branch or halt: the block drains before the next one, or the
    // stream's end.
    tailIdle = std::max(0, drain - free);
  }
  return success();
}

LogicalResult scheduleStream(ModuleOp module, bool insertDelays, StringRef requested, StringRef waitPlacement) {
  FailureOr<TimingProvider> provider = selectAtlasTimingProvider(module, requested);
  if (failed(provider))
    return failure();
  if (waitPlacement != "default" && waitPlacement != "source")
    return module.emitError("wait-placement must be default or source");
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
    if (failed(scheduleBlock(*stream, b, *kind == AtlasArtifactKind::Generated,
                             waitPlacement == "source",
                             *provider, insertDelays && fillsDelaySlots(*provider),
                             order, before, tailIdle[b])))
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
  if (insertDelays && provider->id != kBuiltinTimingProviderId)
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
                               llvm::cl::desc("Timing provider (default: atlas.timing_provider, else builtin)"),
                               llvm::cl::init("")};
  Option<std::string> waitPlacement{
      *this, "wait-placement",
      llvm::cl::desc("DMA wait placement: default or source"),
      llvm::cl::init("default")};

  StringRef getArgument() const final { return "schedule-atlas-stream"; }
  StringRef getDescription() const final {
    return "Reorder each basic block with the list scheduler ported from "
           "atlas-compiler-experiments and insert the delays it needs";
  }

  void runOnOperation() override {
    if (failed(scheduleStream(getOperation(), insertDelays, provider.getValue(), waitPlacement.getValue())))
      signalPassFailure();
  }
};

} // namespace

void mlir::atlas::registerScheduleAtlasStreamPass() {
  PassRegistration<ScheduleAtlasStreamPass>();
}

std::unique_ptr<Pass> mlir::atlas::createScheduleAtlasStreamPass(StringRef provider, StringRef waitPlacement) {
  auto pass = std::make_unique<ScheduleAtlasStreamPass>();
  pass->provider = provider.str();
  pass->waitPlacement = waitPlacement.str();
  return pass;
}
