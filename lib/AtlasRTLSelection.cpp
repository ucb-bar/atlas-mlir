#include "Atlas/AtlasRTLSelection.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasStream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <map>
#include <optional>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {

llvm::Error selectionError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

// Bounds explicit DELAY streams before reservation entries are allocated.
constexpr int kMaximumIssueCycle = 1000000;
constexpr int kMaximumPermittedSearch = 4096;

// The DMA engine queues launches into eight command slots indexed by launch
// count; the RTL protects neither slot reuse nor channel reuse.
constexpr size_t kDMASlots = 8;

bool isLaunch(const Instr &in) {
  return in.op->opClass == OpClass::DmaLoad || in.op->opClass == OpClass::DmaStore;
}

int holdEnd(const Footprint &f, Unit unit) {
  int end = -1;
  for (const Hold &h : f.holds)
    if (h.unit == unit) end = std::max(end, h.to);
  return end;
}

bool isVls(const Footprint &f) {
  return std::any_of(f.holds.begin(), f.holds.end(), [](const Hold &h) {
    return h.unit == Unit::VloadPath || h.unit == Unit::VstorePath;
  });
}

// A VLS tile may not touch a pending transfer's VMEM window at all; the queued
// transfer reads or writes it whenever it reaches the head of the queue.
bool vlsOverlapsWindow(const Footprint &dma, const Footprint &f) {
  for (const Access &window : dma.accesses) {
    if (window.res != Res::Vmem || !window.atCompletion) continue;
    for (const Access &tile : f.accesses)
      if (tile.res == Res::Vmem && tile.first < window.first + window.count && window.first < tile.first + tile.count)
        return true;
  }
  return false;
}

bool isEngineHold(const Hold &h) { return h.unit == Unit::Vpu || h.unit == Unit::Xlu; }

Footprint holdsWhere(const Footprint &f, bool engine) {
  Footprint out = engine ? Footprint{} : f;
  out.holds.clear();
  for (const Hold &h : f.holds)
    if (isEngineHold(h) == engine) out.holds.push_back(h);
  return out;
}

// MREG registers a command names, reads then writes in operand order.
std::vector<int> mregOperands(const Footprint &f) {
  std::vector<int> regs = f.mregReads;
  regs.insert(regs.end(), f.mregWrites.begin(), f.mregWrites.end());
  return regs;
}

// Pair facts name equal registers as i=j, equal banks as i~j, or disjoint.
std::string mregOverlap(const std::vector<int> &first, const std::vector<int> &second) {
  std::string out;
  for (size_t i = 0; i < first.size(); ++i)
    for (size_t j = 0; j < second.size(); ++j)
      if ((first[i] & 31) == (second[j] & 31))
        out += (out.empty() ? "" : ",") + std::to_string(i) + (first[i] == second[j] ? "=" : "~") + std::to_string(j);
  return out.empty() ? "disjoint" : out;
}

Footprint withoutHold(const Footprint &f, Unit unit) {
  Footprint out = f;
  out.holds.erase(std::remove_if(out.holds.begin(), out.holds.end(), [&](const Hold &h) { return h.unit == unit; }),
                  out.holds.end());
  return out;
}

// Pair facts admit overlap only at covered gaps, matching operands and observed
// live capacity. VPU windows end at next_issue_age; MXU windows at the busy end.
class PairSlots {
public:
  explicit PairSlots(const RTLEvidence &evidence) : evidence(evidence), facts(*evidence.pairFacts()) {}

  struct Command {
    const OpInfo *op = nullptr;
    std::string block, engine, holder;
    std::vector<int> regs;
    std::optional<int> acc, slot;
    int issue = 0, window = 0;
  };

  std::optional<Command> command(const Instr &in, const Footprint &f, int cycle) const {
    if (!in.op) return std::nullopt;
    if (in.op->engine == Engine::Vpu) {
      const char *name = RTLEvidence::vpuBlockName(in.op->name);
      const OpTimingBlock *b = name ? evidence.block(name) : nullptr;
      if (!b || !b->nextIssueAge) return std::nullopt;
      return Command{in.op, name, b->engine, "VPU slot", mregOperands(f), std::nullopt, std::nullopt, cycle,
                     *b->nextIssueAge};
    }
    std::string name = RTLEvidence::mxuBlockName(*in.op);
    const OpTimingBlock *b = name.empty() ? nullptr : evidence.block(name);
    auto window = facts.window.find(b ? b->engine : "");
    int busy = holdEnd(f, Unit::MxuCompute);
    if (!b || window == facts.window.end() || window->second != "last_event_age" || busy < 0)
      return std::nullopt;
    Command command{in.op, name, b->engine, "MXU in-flight matmuls " + std::to_string(in.op->mxu),
                    mregOperands(f), std::nullopt, std::nullopt, cycle, busy + 1};
    switch (in.op->opClass) {
    case OpClass::WeightPush: command.slot = in.rd; break;
    case OpClass::AccPushFp8:
    case OpClass::AccPushBf16: command.acc = in.rd; break;
    case OpClass::PopFp8:
    case OpClass::PopBf16: command.acc = in.rs2; break;
    default: command.acc = in.rd; command.slot = in.rs2; break; // matmul
    }
    return command;
  }

  std::string conflict(const Command &next) const {
    std::vector<const Command *> overlapping;
    for (const Command &live : commands) {
      if (live.engine != next.engine) continue;
      const Command &first = live.issue <= next.issue ? live : next;
      const Command &second = live.issue <= next.issue ? next : live;
      int gap = second.issue - first.issue;
      if (gap >= first.window) continue;
      overlapping.push_back(&live);
      if (!admits(first, second, gap))
        return live.holder + " held by " + live.op->name + " issued at cycle " + std::to_string(live.issue) +
               " until cycle " + std::to_string(live.issue + live.window) + "; no pair fact admits " +
               second.op->name + " " + std::to_string(gap) + " cycles after " + first.op->name +
               " with these operands";
    }
    auto limit = facts.capacity.find(next.engine);
    int capacity = limit == facts.capacity.end() ? 1 : limit->second;
    for (int start = next.issue; start < next.issue + next.window; ++start) {
      int live = 1;
      for (const Command *other : overlapping)
        live += other->issue <= start && start < other->issue + other->window;
      if (live > capacity)
        return "all " + std::to_string(capacity) + " " + (next.op->engine == Engine::Vpu ? "VPU" : next.engine) +
               " slots busy at cycle " + std::to_string(start) + " (live capacity from the pair facts)";
    }
    return {};
  }

  void reserve(Command command) { commands.push_back(std::move(command)); }

private:
  // A pair fact covers gaps from its stable gap through the end of its sweep,
  // which must reach the end of the first command's live window.
  bool admits(const Command &first, const Command &second, int gap) const {
    auto found = facts.pairs.find({first.block, second.block});
    if (found == facts.pairs.end() || gap < 1) return false;
    for (const OpPairFact &fact : found->second)
      if (fact.stableGap <= gap && fact.lastGap + 1 >= first.window && satisfied(fact, first, second))
        return true;
    return false;
  }
  // Require known relations for MREG banks and every shared accumulator or slot.
  static bool satisfied(const OpPairFact &fact, const Command &first, const Command &second) {
    bool banks = false, acc = false, slot = false;
    auto same = [](const std::optional<int> &a, const std::optional<int> &b, const std::string &value) {
      return a && b && (value == "same" || value == "other") && ((*a == *b) == (value == "same"));
    };
    for (const auto &[name, value] : fact.relation) {
      if (name == "mreg_banks") {
        if (value != mregOverlap(first.regs, second.regs)) return false;
        banks = true;
      } else if (name == "accumulator" && same(first.acc, second.acc, value)) {
        acc = true;
      } else if (name == "weight_slot" && same(first.slot, second.slot, value)) {
        slot = true;
      } else {
        return false;
      }
    }
    return banks && (acc || !(first.acc && second.acc)) && (slot || !(first.slot && second.slot));
  }

  const RTLEvidence &evidence;
  const RTLPairFacts &facts;
  std::vector<Command> commands;
};

class RTLReservations final : public TimingReservations {
public:
  explicit RTLReservations(std::shared_ptr<const RTLEvidence> evidence) : evidence(std::move(evidence)) {
    if (this->evidence && this->evidence->pairFacts()) slots.emplace(*this->evidence);
  }
  TimingRuleResult<std::string>
  conflict(const Instr &in, const Footprint &f, int cycle) const override {
    if (cycle > kMaximumIssueCycle)
      return {{}, "exceeds the bounded RTL timing cycle limit"};
    std::string busy = busyAt(in, f, cycle);
    if (busy.empty())
      return {};
    int permitted = cycle + 1;
    while (permitted - cycle < kMaximumPermittedSearch && !busyAt(in, f, permitted).empty())
      ++permitted;
    return {busy + "; issue cycle " + std::to_string(cycle) +
                ", first permitted cycle " + std::to_string(permitted),
            {}};
  }
  std::string reserve(const Instr &in, const Footprint &f, int cycle) override {
    if (auto command = slots ? slots->command(in, f, cycle) : std::nullopt) {
      if (in.op->engine == Engine::Vpu) {
        table.reserve(in, holdsWhere(f, false), cycle);
        engines.reserve(in, holdsWhere(f, true), cycle);
      } else {
        table.reserve(in, withoutHold(f, Unit::MxuCompute), cycle);
      }
      slots->reserve(std::move(*command));
      return {};
    }
    table.reserve(in, f, cycle);
    return {};
  }
  std::string onWait(const Instr &, int cycle) override {
    table.extendForWait(cycle);
    engines.extendForWait(cycle);
    return {};
  }

private:
  // Without pair facts every unit, the VPU and each MXU included, is exclusive
  // in `table`. Slot commands keep no own-engine hold there, and a later
  // stall only widens the gaps the facts admit.
  std::string busyAt(const Instr &in, const Footprint &f, int cycle) const {
    std::string busy = table.conflict(in, f, cycle);
    if (!busy.empty() || !slots)
      return busy;
    if (auto command = slots->command(in, f, cycle))
      return slots->conflict(*command);
    return engines.conflict(in, holdsWhere(f, true), cycle);
  }

  std::shared_ptr<const RTLEvidence> evidence;
  ReservationTable table{ReservationPolicy::Exclusive};
  ReservationTable engines{ReservationPolicy::Exclusive};
  std::optional<PairSlots> slots;
};

// Every reachable block starts idle and drains before its successor or ECALL.
// DMA launches need distinct channels and free slots until their matching waits.
std::string validateScope(const AtlasStream &s) {
  if (s.ops.size() > RTLEvidence::maximumProgramWords())
    return "program exceeds the 32768-word instruction memory";
  size_t blocks = s.starts.size();
  if (blocks == 0)
    return "selected RTL timing requires a terminal instruction";
  std::vector<bool> reached(blocks, false);
  std::vector<size_t> work = {0};
  reached[0] = true;
  while (!work.empty()) {
    size_t b = work.back();
    work.pop_back();
    for (size_t next : s.succs[b])
      if (!reached[next]) {
        reached[next] = true;
        work.push_back(next);
      }
  }
  auto at = [&](size_t i, const std::string &why) {
    return "word " + std::to_string(i) + " (" + s.ops[i]->getName().getStringRef().str() + ") " + why;
  };
  for (size_t b = 0; b < blocks; ++b) {
    size_t begin = s.starts[b], end = s.blockEnd(b);
    if (!reached[b])
      return at(begin, b > 0 && s.endsInHalt(b - 1) && !s.endsInBranch(b - 1)
                           ? "follows the terminal instruction"
                           : "is unreachable; selected RTL timing verifies only reachable blocks");
    if (s.endsInBranch(b)) {
      if (!s.targetOf.lookup(s.ops[end - 2]))
        return at(end - 2, "targets the stream end; selected RTL timing requires every path to reach ECALL");
      if (s.instrs[end - 1].op->opClass != OpClass::Alu)
        return at(end - 1, "is a delay-slot instruction; selected RTL timing admits only ADDI or LUI in a delay slot");
    } else if (!s.endsInHalt(b) && end == s.ops.size()) {
      return at(end - 1, "ends the stream without ECALL; selected RTL timing requires a terminal instruction");
    }
    struct Launch { size_t index; int channel; bool waited; };
    std::vector<Launch> launches;
    std::array<int, 8> pendingOn;
    pendingOn.fill(-1);
    for (size_t i = begin; i < end; ++i) {
      const Instr &in = s.instrs[i];
      int channel = in.op->channel;
      if (isLaunch(in)) {
        if (pendingOn[channel] >= 0)
          return at(i, "launches on channel " + std::to_string(channel) + " while its transfer at word " +
                           std::to_string(launches[pendingOn[channel]].index) +
                           " has not been waited; the RTL does not protect channel reuse, so add atlas.dma_wait first");
        if (launches.size() >= kDMASlots && !launches[launches.size() - kDMASlots].waited)
          return at(i, "is launch " + std::to_string(launches.size()) + " of its block while launch " +
                           std::to_string(launches.size() - kDMASlots) + " (word " +
                           std::to_string(launches[launches.size() - kDMASlots].index) +
                           ") has not been waited; the DMA engine has eight command slots");
        pendingOn[channel] = static_cast<int>(launches.size());
        launches.push_back({i, channel, false});
      } else if (in.op->opClass == OpClass::DmaWait) {
        if (pendingOn[channel] < 0)
          return at(i, "has no pending transfer to wait for on channel " + std::to_string(channel));
        launches[pendingOn[channel]].waited = true;
        pendingOn[channel] = -1;
      }
    }
    for (const Launch &launch : launches)
      if (!launch.waited)
        return at(launch.index, "launches a DMA transfer that is still pending when its block ends; selected RTL timing requires atlas.dma_wait on channel " +
                                    std::to_string(launch.channel) + " within the block");
  }
  return {};
}

} // namespace

static bool sameAccess(const Access &a, const Access &b) {
  return std::tie(a.res, a.write, a.first, a.count, a.age, a.step, a.anywhere, a.atCompletion) ==
         std::tie(b.res, b.write, b.first, b.count, b.age, b.step, b.anywhere, b.atCompletion);
}

static bool sameHold(const Hold &a, const Hold &b) {
  return std::tie(a.unit, a.index, a.from, a.to, a.alt) == std::tie(b.unit, b.index, b.from, b.to, b.alt);
}

static bool sameFootprint(const Footprint &a, const Footprint &b) {
  return std::equal(a.accesses.begin(), a.accesses.end(), b.accesses.begin(), b.accesses.end(), sameAccess) &&
         std::equal(a.holds.begin(), a.holds.end(), b.holds.begin(), b.holds.end(), sameHold) &&
         std::tie(a.mregReads, a.mregWrites, a.readRelease, a.writeRelease, a.writeDuringRead, a.vpuLive, a.doneAge, a.dmaCycles, a.error) ==
             std::tie(b.mregReads, b.mregWrites, b.readRelease, b.writeRelease, b.writeDuringRead, b.vpuLive, b.doneAge, b.dmaCycles, b.error);
}

std::vector<AtlasContextGroup> mlir::atlas::atlasBlockContexts(
    const AtlasStream &s, size_t block, const TimingProvider &provider) {
  if (!provider.concreteEntryContexts || s.contexts.empty() || s.contexts[block].empty())
    return {{{s.entry[block]}}};
  // Footprints are pure in the instruction and its register reads, so contexts
  // with equal footprint sequences time identically.
  std::vector<AtlasContextGroup> groups;
  std::vector<std::vector<Footprint>> resolved;
  for (const RegValues &entry : s.contexts[block]) {
    std::vector<Footprint> footprints;
    RegValues regs = entry;
    for (size_t i = s.starts[block]; i < s.blockEnd(block); ++i) {
      footprints.push_back(provider.footprint(s.instrs[i], regs));
      applyScalar(s.instrs[i], regs);
    }
    auto same = [&](const std::vector<Footprint> &other) {
      return std::equal(footprints.begin(), footprints.end(), other.begin(), other.end(), sameFootprint);
    };
    auto it = llvm::find_if(resolved, same);
    if (it == resolved.end()) {
      resolved.push_back(std::move(footprints));
      groups.push_back({{entry}});
    } else {
      groups[it - resolved.begin()].entries.push_back(entry);
    }
  }
  return groups;
}

InFlightDiagnostic mlir::atlas::emitAtlasFootprintError(const AtlasStream &s, Operation *op, StringRef error,
                                                        const RegValues *context) {
  InFlightDiagnostic diagnostic = op->emitOpError(error);
  if (!s.contextLimit.empty())
    diagnostic.attachNote() << s.contextLimit;
  if (context) {
    std::string known;
    for (int r = 1; r < 32; ++r)
      if ((*context)[r])
        known += " x" + std::to_string(r) + "=0x" + llvm::utohexstr(*(*context)[r]);
    diagnostic.attachNote() << "in the block entry context with" << (known.empty() ? " no known registers" : known);
  }
  return diagnostic;
}

TimingRuleResult<DepGraph> mlir::atlas::timing::buildTimingGraph(
    const std::vector<Instr> &instrs, const RegValues &entry,
    const TimingProvider &p) {
  TimingRuleResult<DepGraph> result;
  DepGraph &g = result.value;
  g.nodes = instrs;
  int n = static_cast<int>(instrs.size());
  RegValues regs = entry;
  for (const Instr &in : instrs) {
    Footprint f = p.footprint(in, regs);
    if (!f.error.empty())
      return {{}, f.error};
    g.footprints.push_back(std::move(f));
    applyScalar(in, regs);
  }
  // Adds an edge, or raises the distance of an existing edge between the nodes.
  std::map<std::pair<int, int>, int> index;
  auto add = [&](int from, int to, int distance, EdgeKind kind, const std::string &reason) {
    auto it = index.find({from, to});
    if (it == index.end()) {
      index[{from, to}] = static_cast<int>(g.edges.size());
      g.edges.push_back({from, to, distance, kind, reason});
    } else if (distance > g.edges[it->second].distance) {
      g.edges[it->second] = {from, to, distance, kind, reason};
    }
  };
  for (int b = 0; b < n; b++)
    for (int a = 0; a < b; a++) {
      auto d = p.dependence(g.nodes[a], g.footprints[a], g.nodes[b], g.footprints[b]);
      if (!d.error.empty())
        return {{}, d.error};
      if (d.value.distance > 0)
        add(a, b, d.value.distance, d.value.kind, d.value.reason);
    }
  // A DMA's completion is known only once its dma.wait issues, so later
  // conflicting accesses wait for that dma.wait.
  for (int d = 0; d < n; d++) {
    if (!isLaunch(g.nodes[d]))
      continue;
    const OpInfo &op = *g.nodes[d].op;
    int wait = -1;
    for (int k = d + 1; k < n && wait < 0; k++)
      if (g.nodes[k].op->opClass == OpClass::DmaWait && g.nodes[k].op->channel == op.channel)
        wait = k;
    for (int k = d + 1; k < n; k++) {
      if (k == wait)
        continue;
      auto conflict = p.dmaConflict(g.footprints[d], g.footprints[k]);
      if (!conflict.error.empty())
        return {{}, conflict.error};
      if (!conflict.value.conflict)
        continue;
      if (wait >= 0 && wait < k)
        add(wait, k, 1, conflict.value.kind,
            std::string(edgeKindName(conflict.value.kind)) + " with " + op.name + " (done once the wait issues)");
      else
        add(d, k, 1, EdgeKind::Order, op.name + " may still be in flight (no dma.wait in between)");
    }
  }
  // A wait for a transfer from an earlier block guards conflicting accesses
  // and channel reuse; the transfer latched its registers at launch, so only
  // the VMEM it moves at completion may conflict.
  for (int w = 0; w < n; w++) {
    const OpInfo &op = *g.nodes[w].op;
    if (op.opClass != OpClass::DmaWait)
      continue;
    bool local = false;
    for (int d = 0; d < w; d++)
      if (isLaunch(g.nodes[d]) && g.nodes[d].op->channel == op.channel)
        local = true;
    if (local)
      continue;
    for (int k = w + 1; k < n; k++) {
      bool guarded = isLaunch(g.nodes[k]);
      for (const Access &a : g.footprints[k].accesses)
        if (a.res == Res::Vmem)
          guarded = true;
      if (guarded)
        add(w, k, 1, EdgeKind::Order, op.name + " guards a transfer started in an earlier block");
    }
  }
  g.in.assign(n, {});
  g.out.assign(n, {});
  for (int e = 0; e < static_cast<int>(g.edges.size()); e++) {
    g.out[g.edges[e].from].push_back(e);
    g.in[g.edges[e].to].push_back(e);
  }
  return result;
}

FailureOr<MachineTiming> mlir::atlas::machineTimingOf(
    const TimingProvider &provider, const std::string &name, Operation *at) {
  auto instruction = [&](int rd, int rs1, int rs2) {
    timing::Instr in;
    in.op = timing::findOp(name);
    assert(in.op && "the timing model knows every machine operation");
    in.rd = rd;
    in.rs1 = rs1;
    in.rs2 = rs2;
    return in;
  };
  // Two instances that share no register, slot, BF16 pair or VMEM bank.
  timing::Instr first = instruction(0, 2, 4), second = instruction(6, 8, 10);
  switch (first.op->opClass) {
  case timing::OpClass::WeightPush:
  case timing::OpClass::AccPushFp8:
  case timing::OpClass::AccPushBf16:
    // A push names its slot and source pair; its second field is unused.
    first = instruction(0, 0, 0);
    second = instruction(1, 2, 0);
    break;
  case timing::OpClass::MatMul:
  case timing::OpClass::MatMulAcc:
    // MXU slot operands are 0 or 1.
    first = instruction(0, 0, 0);
    second = instruction(1, 2, 1);
    break;
  case timing::OpClass::PopFp8:
  case timing::OpClass::PopBf16:
    // A pop names its destination pair, scale register and accumulator slot.
    first = instruction(0, 0, 0);
    second = instruction(2, 0, 1);
    break;
  default:
    break;
  }
  timing::RegValues regs = timing::unknownRegs();
  regs[2] = 0;
  regs[8] = timing::kVmemBankBytes / 4;
  timing::Footprint firstFootprint = provider.footprint(first, regs);
  timing::Footprint secondFootprint = provider.footprint(second, regs);
  for (const timing::Footprint &f : {firstFootprint, secondFootprint})
    if (!f.error.empty())
      return at->emitOpError("virtual cost of ") << name << ": " << f.error;
  auto table = provider.createReservations();
  if (!table.error.empty() || !table.value)
    return at->emitOpError("virtual cost of ") << name << ": "
           << (table.error.empty() ? "timing provider returned no reservation state" : table.error);
  std::string why = table.value->reserve(first, firstFootprint, 0);
  int interval = 1;
  for (; why.empty(); ++interval) {
    auto conflict = table.value->conflict(second, secondFootprint, interval);
    why = conflict.error;
    if (why.empty() && conflict.value.empty())
      break;
  }
  if (!why.empty())
    return at->emitOpError("virtual cost of ") << name << ": " << why;
  return MachineTiming{firstFootprint.doneAge + 1, interval};
}

TimingProvider mlir::atlas::rtlTimingProvider(std::shared_ptr<const RTLEvidence> evidence) {
  TimingProvider p;
  p.id = kAtlasRTLTimingProviderId.str();
  p.redirectWaitsForDelay = false;
  p.concreteEntryContexts = true;
  p.validateScope = validateScope;
  p.footprint = [evidence](const Instr &in, const RegValues &regs) {
    return evidence->resolve(in, regs);
  };
  p.dependence = [](const Instr &a, const Footprint &fa, const Instr &b, const Footprint &fb) {
    Dependence d = dataDependence(a, fa, b, fb);
    // The LSU reports a VSTORE's source register as read until the VSTORE
    // drains; every writer but a VLOAD asserts on a pending read
    // (MregBankTracker, pairs/core_mxu_cross.yaml).
    if (a.op->opClass == OpClass::VStore && b.op->opClass != OpClass::VLoad) {
      int end = holdEnd(fa, Unit::VstorePath);
      for (int r : fa.mregReads)
        if (std::find(fb.mregWrites.begin(), fb.mregWrites.end(), r) != fb.mregWrites.end() &&
            end + 1 > d.distance)
          d = {end + 1, EdgeKind::WAR,
               "m" + std::to_string(r) + " reported read until the VSTORE drains at age " +
                   std::to_string(end)};
    }
    // Launches keep queue order, so a wait kept ahead of every later launch
    // keeps each launch's command slot free, as the scope requires.
    if (isLaunch(b) && a.op->opClass == OpClass::DmaWait && d.distance < 1)
      d = {1, EdgeKind::Order, "selected DMA launch stays behind an earlier wait (eight command slots)"};
    return TimingRuleResult<Dependence>{d, {}};
  };
  // Pending windows stay disjoint from each other and from VLS tiles; DRAM
  // spans may be shared only by reads. VLS, VPU, XLU and MXU commands kept
  // their isolated events beside a pending transfer in every swept run.
  p.dmaConflict = [](const Footprint &dma, const Footprint &f) {
    DMACompletionConflict result;
    result.conflict = conflictsAtCompletion(dma, f, result.kind);
    if (!result.conflict && isVls(f) && vlsOverlapsWindow(dma, f))
      result = {true, EdgeKind::Order};
    return TimingRuleResult<DMACompletionConflict>{result, {}};
  };
  p.issueGap = [](const Instr &in) { return TimingRuleResult<int>{naturalGap(in), {}}; };
  p.createReservations = [evidence] {
    TimingRuleResult<std::unique_ptr<TimingReservations>> result;
    result.value = std::make_unique<RTLReservations>(evidence);
    return result;
  };
  return p;
}

TimingRuleResult<TimingProvider> mlir::atlas::rtlTimingProviderFactory(ModuleOp module) {
  if (!module)
    return {{}, "the RTL timing provider requires a module carrying selected atlas.rtl_evidence"};
  auto evidence = loadSelectedRTLEvidence(module);
  if (!evidence)
    return {{}, llvm::toString(evidence.takeError())};
  if (!*evidence)
    return {{}, "the RTL timing provider requires selected evidence (--select-atlas-rtl-evidence)"};
  return {rtlTimingProvider(std::move(*evidence)), {}};
}

llvm::Expected<std::shared_ptr<RTLEvidence>>
mlir::atlas::loadSelectedRTLEvidence(ModuleOp module) {
  Attribute raw = module->getAttr(kAtlasRTLEvidence);
  if (!raw)
    return std::shared_ptr<RTLEvidence>();
  auto attr = dyn_cast<DictionaryAttr>(raw);
  if (!attr)
    return selectionError("atlas.rtl_evidence must be a selection dictionary");
  auto path = attr.getAs<StringAttr>("op_timing");
  auto hash = attr.getAs<StringAttr>("op_timing_sha256");
  auto hardware = attr.getAs<StringAttr>("hardware_ir_sha256");
  auto provider = attr.getAs<StringAttr>("provider");
  if (!path || !hash || !hardware || !provider)
    return selectionError("incomplete RTL evidence selection");
  bool dma = false;
  if (Attribute policy = attr.get("dma")) {
    auto text = dyn_cast<StringAttr>(policy);
    if (!text || text.getValue() != "wait")
      return selectionError("unsupported DMA policy; only dma=wait is defined");
    dma = true;
  }
  if (provider.getValue() != kAtlasRTLTimingProviderId)
    return selectionError("unsupported RTL evidence provider: " + provider.getValue().str());
  auto loaded = loadRTLTimingFacts(path.getValue(), hash.getValue(), dma);
  if (!loaded)
    return selectionError("RTL evidence selection failed: " + llvm::toString(loaded.takeError()));
  if (loaded->hardwareIRSha256() != hardware.getValue())
    return selectionError("selected facts describe a different hardware IR than recorded");
  auto pairs = attr.getAs<StringAttr>("op_pairs");
  auto pairsHash = attr.getAs<StringAttr>("op_pairs_sha256");
  if ((attr.get("op_pairs") || attr.get("op_pairs_sha256")) && !(pairs && pairsHash))
    return selectionError("incomplete RTL pair facts selection");
  if (pairs)
    if (llvm::Error error = loaded->selectPairFacts(pairs.getValue(), pairsHash.getValue()))
      return selectionError("RTL evidence selection failed: " + llvm::toString(std::move(error)));
  return std::make_shared<RTLEvidence>(std::move(*loaded));
}

FailureOr<std::shared_ptr<RTLEvidence>>
mlir::atlas::getSelectedRTLEvidence(ModuleOp module) {
  auto evidence = loadSelectedRTLEvidence(module);
  if (!evidence)
    return module.emitError(llvm::toString(evidence.takeError()));
  return std::move(*evidence);
}

namespace {
struct SelectAtlasRTLEvidencePass
    : PassWrapper<SelectAtlasRTLEvidencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectAtlasRTLEvidencePass)
  SelectAtlasRTLEvidencePass() = default;
  SelectAtlasRTLEvidencePass(const SelectAtlasRTLEvidencePass &other)
      : PassWrapper(other) {}
  Option<std::string> opTiming{*this, "op-timing",
      llvm::cl::desc("RTL-computed operation timing facts (merlin.op_timing.v1)")};
  Option<std::string> opTimingHash{*this, "op-timing-sha256",
      llvm::cl::desc("Expected SHA-256 of the facts file")};
  Option<std::string> dmaPolicy{*this, "dma",
      llvm::cl::desc("Optional DMA policy: wait")};
  Option<std::string> opPairs{*this, "op-pairs",
      llvm::cl::desc("Optional pair facts (merlin.op_pairs.v1) derived from the same facts")};
  Option<std::string> opPairsHash{*this, "op-pairs-sha256",
      llvm::cl::desc("Expected SHA-256 of the pair facts file")};
  StringRef getArgument() const final { return "select-atlas-rtl-evidence"; }
  StringRef getDescription() const final {
    return "Select SHA-256-identified RTL-computed timing facts for the RTL timing provider";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!dmaPolicy.empty() && dmaPolicy != "wait") {
      module.emitError("unsupported DMA policy; only dma=wait is defined");
      signalPassFailure();
      return;
    }
    auto loaded = loadRTLTimingFacts(opTiming, opTimingHash, dmaPolicy == "wait");
    if (!loaded) {
      module.emitError("RTL evidence selection failed: ") << llvm::toString(loaded.takeError());
      signalPassFailure();
      return;
    }
    if (opPairs.empty() != opPairsHash.empty()) {
      module.emitError("op-pairs and op-pairs-sha256 are selected together");
      signalPassFailure();
      return;
    }
    if (!opPairs.empty())
      if (llvm::Error error = loaded->selectPairFacts(opPairs, opPairsHash)) {
        module.emitError("RTL evidence selection failed: ") << llvm::toString(std::move(error));
        signalPassFailure();
        return;
      }
    if (auto retained = module->getAttrOfType<StringAttr>(kAtlasTimingProvider);
        retained && retained.getValue() != kAtlasRTLTimingProviderId) {
      module.emitError("selected RTL evidence contradicts the retained timing provider ")
          << retained.getValue() << "; time the stream again after selection or drop the selection";
      signalPassFailure();
      return;
    }
    Builder builder(module.getContext());
    NamedAttrList fields;
    fields.set("op_timing", builder.getStringAttr(opTiming));
    fields.set("op_timing_sha256", builder.getStringAttr(opTimingHash));
    fields.set("hardware_ir_sha256", builder.getStringAttr(loaded->hardwareIRSha256()));
    if (loaded->dmaWait())
      fields.set("dma", builder.getStringAttr("wait"));
    if (!opPairs.empty()) {
      fields.set("op_pairs", builder.getStringAttr(opPairs));
      fields.set("op_pairs_sha256", builder.getStringAttr(opPairsHash));
    }
    fields.set("provider", builder.getStringAttr(kAtlasRTLTimingProviderId));
    module->setAttr(kAtlasRTLEvidence, fields.getDictionary(module.getContext()));
  }
};
} // namespace

void mlir::atlas::registerSelectAtlasRTLEvidencePass() {
  PassRegistration<SelectAtlasRTLEvidencePass>();
}

namespace {
// Ordering uses the compiler's DMA estimate; selected completion remains unknown.
int schedulerDMACycles(const Footprint &f) {
  for (const Access &a : f.accesses)
    if (a.res == Res::Vmem && a.atCompletion)
      return dmaTransferCycles(static_cast<long long>(a.count) * kLineBytes);
  return dmaTransferCycles(32LL * kLineBytes);
}

int primaryOccupancy(const Footprint &f, int &unit) {
  int length = 0;
  unit = -1;
  for (const Hold &h : f.holds) {
    if (h.unit == Unit::ScalarLoad || h.unit == Unit::ScalarWriteback)
      continue;
    if (h.to + 1 > length) {
      length = h.to + 1;
      unit = static_cast<int>(h.unit) * 64 + h.index;
    }
  }
  return length;
}

struct RTLOrderCandidate {
  std::vector<size_t> order;
  std::vector<DelayInsertion> before;
  int tail = 0;
  long long cost = 0;
};

// Waits stall the frontend in this cost model. The same-bank penalty accounts
// for the LSU taking priority over pending DMA VMEM accesses. Neither number
// supplies timing legality: reservations and the common verifier do that.
TimingRuleResult<long long> rtlOrderCost(
    const AtlasStream &s, size_t block, const TimingProvider &p,
    const RegValues &entry, const RTLOrderCandidate &candidate) {
  long long cycle = 0, last = 0;
  std::array<long long, 8> completion{}, penalty{};
  std::array<int, 8> bank;
  bank.fill(-1);
  std::array<bool, 8> pending{};
  RegValues regs = entry;
  for (size_t i : candidate.order) {
    const DelayInsertion &d = candidate.before[i];
    for (uint32_t v : d.delays)
      cycle += static_cast<long long>(v) + 1;
    cycle += d.guard;
    const Instr &in = s.instrs[i];
    Footprint f = p.footprint(in, regs);
    if (!f.error.empty())
      return {0, f.error};
    if (in.op->opClass == OpClass::DmaWait) {
      int ch = in.op->channel;
      cycle = std::max(cycle, completion[ch] + penalty[ch]);
      pending[ch] = false;
    }
    if (isLaunch(in)) {
      int ch = in.op->channel;
      completion[ch] = cycle + schedulerDMACycles(f);
      penalty[ch] = 0;
      pending[ch] = true;
      bank[ch] = -1;
      for (const Access &a : f.accesses)
        if (a.res == Res::Vmem && a.atCompletion && !a.anywhere)
          bank[ch] = a.first / (kVmemBankBytes / kLineBytes);
    }
    if (isVls(f)) {
      for (const Access &a : f.accesses) {
        if (a.res != Res::Vmem || a.anywhere)
          continue;
        int vb = a.first / (kVmemBankBytes / kLineBytes);
        for (int ch = 0; ch < 8; ++ch)
          if (pending[ch] && bank[ch] == vb)
            penalty[ch] += 32;
      }
    }
    applyScalar(in, regs);
    last = cycle;
    auto gap = p.issueGap(in);
    if (!gap.error.empty() || gap.value <= 0)
      return {0, gap.error.empty() ? "nonpositive issue gap" : gap.error};
    cycle += gap.value;
  }
  if (s.endsInHalt(block))
    return {last, {}};
  if (s.endsInBranch(block))
    return {last + 1, {}};
  return {cycle + candidate.tail, {}};
}
} // namespace

void mlir::atlas::estimateRTLGraphDMA(DepGraph &graph) {
  for (size_t i = 0; i < graph.nodes.size(); ++i)
    if (isLaunch(graph.nodes[i]))
      graph.footprints[i].dmaCycles = schedulerDMACycles(graph.footprints[i]);
}

std::vector<int> mlir::atlas::rtlContentionPriorities(
    const DepGraph &graph, const std::vector<int> &issued, int bodyCount) {
  std::vector<int> priority = criticalHeights(graph);
  std::vector<int> units(bodyCount, -1);
  // Remaining occupancy plus the shortest tail bounds completion on a busy
  // unit. Scalar operations retain their critical-height priority.
  std::map<int, std::pair<int, int>> backlog;
  for (int i = 0; i < bodyCount; ++i) {
    int occupancy = primaryOccupancy(graph.footprints[i], units[i]);
    if (issued[i] >= 0 || units[i] < 0)
      continue;
    auto &entry = backlog.try_emplace(units[i], 0, INT_MAX).first->second;
    entry.first += occupancy;
    entry.second = std::min(entry.second, std::max(0, priority[i] - occupancy));
  }
  for (int i = 0; i < bodyCount; ++i) {
    auto found = backlog.find(units[i]);
    if (found != backlog.end())
      priority[i] = std::max(priority[i], found->second.first + found->second.second);
  }
  return priority;
}

LogicalResult mlir::atlas::selectRTLBlockSchedule(
    const AtlasStream &s, size_t block, const TimingProvider &p,
    bool dmaEstimate, const RTLBlockScheduler &schedule,
    std::vector<size_t> &order, std::vector<DelayInsertion> &before, int &tail) {
  const auto groups = atlasBlockContexts(s, block, p);
  size_t begin = s.starts[block], end = s.blockEnd(block);
  std::vector<DepGraph> graphs;
  std::vector<Instr> nodes(s.instrs.begin() + begin, s.instrs.begin() + end);
  for (const AtlasContextGroup &group : groups) {
    auto built = buildTimingGraph(nodes, group.entries.front(), p);
    if (!built.error.empty())
      return emitAtlasFootprintError(s, s.ops[begin], built.error, &group.entries.front());
    graphs.push_back(std::move(built.value));
  }
  auto validOrder = [&](const std::vector<size_t> &candidate) {
    std::vector<size_t> positions(end - begin);
    for (size_t k = 0; k < candidate.size(); ++k)
      positions[candidate[k] - begin] = k;
    for (const DepGraph &graph : graphs)
      for (const Edge &e : graph.edges)
        if (positions[e.from] >= positions[e.to])
          return false;
    return true;
  };
  std::optional<RTLOrderCandidate> selected;
  auto consider = [&](RTLOrderCandidate candidate, bool keepTie = false) -> LogicalResult {
    candidate.cost = 0;
    for (const AtlasContextGroup &group : groups) {
      auto cost = rtlOrderCost(s, block, p, group.entries.front(), candidate);
      if (!cost.error.empty())
        return emitAtlasFootprintError(s, s.ops[begin], cost.error, &group.entries.front());
      candidate.cost = std::max(candidate.cost, cost.value);
    }
    // Keep the measured list-order policy for single-context blocks. A
    // multi-context source fallback wins a tie: reordering must offer a gain.
    if (!selected || candidate.cost < selected->cost ||
        (keepTie && candidate.cost == selected->cost))
      selected = std::move(candidate);
    return success();
  };
  // Different contexts often produce the same list order. Its fixpoint and
  // worst-context cost are identical, so reconcile that order only once.
  std::set<std::vector<size_t>> reconciled;
  for (const AtlasContextGroup &group : groups) {
    for (int variant = 0; variant < (dmaEstimate ? 2 : 1); ++variant) {
      bool estimate = dmaEstimate && variant == 0;
      for (bool contention : {false, true}) {
        RTLOrderCandidate candidate;
        candidate.before.resize(s.ops.size());
        {
          // A speculative order can exceed a bounded search or fail to
          // converge across contexts. Keep the source/legacy alternatives;
          // only a failure of their required rules is a pass error.
          ScopedDiagnosticHandler discard(s.ops[begin]->getContext(),
                                           [](Diagnostic &) { return success(); });
          if (failed(schedule(group.entries.front(), {estimate, contention},
                              candidate.order, candidate.before, candidate.tail)))
            continue;
          if (!validOrder(candidate.order))
            continue;
          if (groups.size() > 1) {
            if (!reconciled.insert(candidate.order).second)
              continue;
            DelayInsertion after;
            if (failed(atlasOrderedBlockDelays(s, block, p, candidate.order,
                                               candidate.before, after, candidate.tail)))
              continue;
          }
        }
        if (failed(consider(std::move(candidate))))
          return failure();
      }
    }
  }
  RTLOrderCandidate source;
  source.before.resize(s.ops.size());
  DelayInsertion after;
  if (failed(atlasInOrderBlockDelays(s, block, p, source.before, after, source.tail)))
    return failure();
  for (size_t i = begin; i < end; ++i)
    source.order.push_back(i);
  if (failed(consider(std::move(source), groups.size() > 1)))
    return failure();
  for (size_t i = begin; i < end; ++i)
    before[i] = std::move(selected->before[i]);
  order.insert(order.end(), selected->order.begin(), selected->order.end());
  tail = selected->tail;
  return success();
}
