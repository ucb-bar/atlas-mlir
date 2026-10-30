// Pre-allocation scheduling of virtual Atlas operations. Each block is list
// scheduled onto the functional units its operations occupy, keeping every
// SSA and implicit-state dependence. Costs are rough on purpose: registers
// and delays do not exist until the virtual-to-machine lowering, nor delay
// slots until the stream scheduler, which times the real instructions.

#include "Atlas/AtlasVirtualScheduling.h"
#include "Atlas/AtlasImmutableWeightReuse.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasSpilling.h"
#include "Atlas/AtlasTarget.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasTiming.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <tuple>

using namespace mlir;
using namespace mlir::atlas;

namespace {

// Ordering obligations the IR does not express as SSA, as pseudo-registers.
enum Pseudo : unsigned {
  kUnit0, // all of MXU0 or MXU1, which a legacy matmul takes
  kUnit1,
  kPack,    // the pack's VMEM scratch and relayout registers
  kBarrier, // read by every operation, written by unknown ones
  kNumPseudo
};

constexpr uint32_t bit(unsigned r) { return 1u << r; }

struct Effects {
  uint32_t reads = 0, writes = 0;
};

unsigned unitOf(Type type) {
  if (auto weight = dyn_cast<VirtualMXUWeightType>(type))
    return weight.getUnit();
  return cast<VirtualMXUAccType>(type).getUnit();
}

// Atomic virtual DMA: a load returns its tile, a store completes in place.
bool isDMA(Operation *op) {
  return isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op, VirtualDMAStoreFP8Op,
             VirtualDMAStoreBF16Op>(op);
}

// Boundary tiles move memory through helper DMA, beside which the generated
// stream admits no explicit transfer: explicit DMA keeps its source order
// around them, and insert-atlas-dma-waits completes every transfer before one.
// A PACK is no such barrier; transfers stay in flight across its relayout.
bool isImplicitMemory(Operation *op) {
  return isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op>(op);
}

// An explicit table, finer than the ops' declared memory effects: a PACK
// writes only its scratch and a legacy matmul only its unit, so transfers and
// other work can move across them. DMA ordering and slot limits are edges of
// their own (buildGraph, SlotPool). An operation missing from the table is a
// full barrier.
Effects effectsOf(Operation *op) {
  Effects e;
  e.reads = bit(kBarrier);
  if (isa<VirtualPackFP8Op, VirtualPackRoundTripBF16Op>(op))
    e.writes = bit(kPack);
  else if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op))
    e.reads |= bit(kUnit0 + unitOf(load.getWeight().getType()));
  else if (isa<VirtualMXULoadAccFP8Op, VirtualMXULoadAccBF16Op,
               VirtualMXUResetOp>(op))
    e.reads |= bit(kUnit0 + unitOf(op->getResult(1).getType()));
  else if (auto acc = dyn_cast<VirtualMXUAccumulateOp>(op))
    e.reads |= bit(kUnit0 + unitOf(acc.getAcc().getType()));
  else if (Value acc = virtualMXUReadoutAcc(op))
    e.reads |= bit(kUnit0 + unitOf(acc.getType()));
  else if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op))
    e.writes = bit(kUnit0 + matmul.getUnit());
  else if (!isDMA(op) &&
           !isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op,
                VirtualVPUUnaryOp, VirtualVPUBinaryOp, VirtualVPUReduceOp,
                VirtualVPUSplatOp, VirtualXLUTransposeOp,
                VirtualScaleConstantOp, VirtualSlotOp,
                arith::ConstantOp, arith::AddIOp, arith::CmpIOp>(op))
    e.writes = bit(kNumPseudo) - 1;
  return e;
}

// A resource with a few interchangeable slots: the weights or accumulators
// of one MXU unit. Along the source
// order each holder takes the slot freed longest ago, and the next holder of
// a slot waits for every last use of the one before. An order keeping these
// edges never needs more slots than there are.
class SlotPool {
public:
  explicit SlotPool(unsigned count) : slots(count) {}

  // `node` takes a slot for a holder last used by `lastUses`, all after it.
  // False when every slot is still held at `node`.
  template <typename Edge>
  bool acquire(unsigned node, ArrayRef<unsigned> lastUses, Edge &&edge) {
    Slot *oldest = nullptr;
    for (Slot &slot : slots)
      if (slot.freeAfter < static_cast<int>(node) &&
          (!oldest || slot.freeAfter < oldest->freeAfter))
        oldest = &slot;
    if (!oldest)
      return false;
    for (unsigned use : oldest->lastUses)
      edge(use, node);
    oldest->lastUses.assign(lastUses.begin(), lastUses.end());
    oldest->freeAfter = node;
    for (unsigned use : lastUses)
      oldest->freeAfter = std::max(oldest->freeAfter, static_cast<int>(use));
    return true;
  }

private:
  struct Slot {
    int freeAfter = -1;
    SmallVector<unsigned, 2> lastUses;
  };
  SmallVector<Slot, 2> slots;
};

// The dependence graph of one block's free operations. Every edge runs from
// an earlier to a later source position.
struct BlockGraph {
  SmallVector<Operation *> nodes;
  std::vector<SmallVector<unsigned, 4>> preds, succs;
};

// The operations a schedule may move: all but the start and the terminator.
SmallVector<Operation *> freeOperations(Block &block) {
  SmallVector<Operation *> ops;
  for (Operation &op : block)
    if (!isa<VirtualStartOp>(op) && !op.hasTrait<OpTrait::IsTerminator>())
      ops.push_back(&op);
  return ops;
}

// Fails, with an error, on a block whose transfers or MXU chains break the
// rules the verifier enforces, since no order of it could be checked.
FailureOr<BlockGraph> buildGraph(Block &block) {
  BlockGraph g;
  g.nodes = freeOperations(block);
  unsigned n = g.nodes.size();
  g.preds.assign(n, {});
  g.succs.assign(n, {});
  llvm::DenseMap<Operation *, unsigned> index;
  for (unsigned i = 0; i < n; ++i)
    index[g.nodes[i]] = i;
  llvm::DenseSet<std::pair<unsigned, unsigned>> seen;
  auto edge = [&](unsigned from, unsigned to) {
    assert(from < to && "dependences follow source order");
    if (seen.insert({from, to}).second) {
      g.succs[from].push_back(to);
      g.preds[to].push_back(from);
    }
  };
  // Data, except the state token, which records source order only.
  for (unsigned i = 0; i < n; ++i)
    for (Value operand : g.nodes[i]->getOperands()) {
      if (isa<VirtualStateType>(operand.getType()))
        continue;
      auto found = index.find(operand.getDefiningOp());
      if (found != index.end())
        edge(found->second, i);
    }
  // Read-after-write, write-after-read, and write-after-write on each
  // pseudo-register. An operation that reads and writes one, as a barrier
  // does, orders after its last writer and every reader since.
  std::vector<Effects> effects;
  for (Operation *op : g.nodes)
    effects.push_back(effectsOf(op));
  for (unsigned r = 0; r < kNumPseudo; ++r) {
    std::optional<unsigned> writer;
    SmallVector<unsigned> readers;
    for (unsigned i = 0; i < n; ++i) {
      bool reads = effects[i].reads & bit(r);
      bool writes = effects[i].writes & bit(r);
      if (writer && (reads || writes))
        edge(*writer, i);
      if (writes) {
        for (unsigned reader : readers)
          edge(reader, i);
        readers.clear();
        writer = i;
      } else if (reads) {
        readers.push_back(i);
      }
    }
  }
  auto cannotOrder = [](Operation *op, const Twine &why) {
    return op->emitError("schedule-atlas-virtual cannot order this block: ")
           << why;
  };
  auto nodeOf = [&](Operation *op) -> std::optional<unsigned> {
    auto found = index.find(op);
    if (found == index.end())
      return std::nullopt;
    return found->second;
  };
  // Each user of `value`, all of which must be operations of this block.
  auto usersOf = [&](Value value) -> std::optional<SmallVector<unsigned, 2>> {
    SmallVector<unsigned, 2> users;
    for (Operation *user : value.getUsers()) {
      std::optional<unsigned> node = nodeOf(user);
      if (!node)
        return std::nullopt;
      users.push_back(*node);
    }
    return users;
  };
  // The readout ending the accumulator chain `acc` starts. Each version is
  // used once, by the accumulate that replaces it or the readout.
  auto chainEnd = [&](Value acc) -> std::optional<unsigned> {
    while (acc.hasOneUse()) {
      Operation *user = *acc.user_begin();
      if (auto next = dyn_cast<VirtualMXUAccumulateOp>(user)) {
        acc = next.getNextAcc();
        continue;
      }
      if (isa<VirtualMXUReadoutBF16Op, VirtualMXUReadoutFP8Op>(user))
        return nodeOf(user);
      return std::nullopt;
    }
    return std::nullopt;
  };

  // Atomic DMA keeps its state order around boundary tiles, and between two
  // operations that may touch the same DRAM bytes while one writes. Any
  // other DMA may move; insert-atlas-dma-waits completes it where needed.
  SmallVector<unsigned> transfers, implicit;
  for (unsigned i = 0; i < n; ++i) {
    if (isDMA(g.nodes[i]))
      transfers.push_back(i);
    else if (isImplicitMemory(g.nodes[i]))
      implicit.push_back(i);
  }
  for (unsigned m : implicit)
    for (unsigned d : transfers)
      edge(std::min(m, d), std::max(m, d));
  for (unsigned a = 0; a < transfers.size(); ++a)
    for (unsigned b = a + 1; b < transfers.size(); ++b)
      if (virtualDMAConflict(g.nodes[transfers[a]], g.nodes[transfers[b]]))
        edge(transfers[a], transfers[b]);

  // Slot limits. A weight is last used by its users, or by its load if it has
  // none, and an accumulator chain by the readout that ends it.
  std::array<SlotPool, 2> weights = {SlotPool(kVirtualMXUSlots),
                                     SlotPool(kVirtualMXUSlots)};
  std::array<SlotPool, 2> accumulators = {SlotPool(kVirtualMXUSlots),
                                          SlotPool(kVirtualMXUSlots)};
  for (unsigned i = 0; i < n; ++i) {
    Operation *op = g.nodes[i];
    if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op)) {
      std::optional<SmallVector<unsigned, 2>> users =
          usersOf(load.getWeight());
      if (!users)
        return cannotOrder(op, "its weight is used outside it");
      if (users->empty())
        users->push_back(i);
      if (!weights[unitOf(load.getWeight().getType())].acquire(i, *users,
                                                                edge))
        return cannotOrder(op, "it holds more weights at once than the "
                               "verifier allows");
    } else if (isa<VirtualMXULoadAccFP8Op, VirtualMXULoadAccBF16Op,
                   VirtualMXUResetOp>(op)) {
      Value acc = op->getResult(1);
      std::optional<unsigned> end = chainEnd(acc);
      if (!end)
        return cannotOrder(op, "its accumulator chain does not end in a "
                               "readout in it");
      if (!accumulators[unitOf(acc.getType())].acquire(i, {*end}, edge))
        return cannotOrder(op, "it holds more accumulators at once than the "
                               "verifier allows");
    }
  }
  return g;
}

// The functional units an operation can occupy. Each MXU pushes weights
// through its own stream, beside its array.
enum class Unit { Scalar, Lsu, Vpu, Xlu, Dma, Mxu0, Mxu1, Weights0, Weights1 };
constexpr unsigned kUnits = static_cast<unsigned>(Unit::Weights1) + 1;

// What an operation costs: it holds one functional unit for `busy` cycles,
// its results are ready after `latency`, and the in-order frontend issues the
// next operation `frontend` cycles after it starts.
struct Cost {
  Unit unit;
  int64_t busy;
  int64_t latency;
  int64_t frontend = 1;
  // An atomic DMA lowers to a launch the frontend issues without waiting for
  // the engine; its transfer (`busy` cycles) queues behind earlier ones after
  // `frontend` staging cycles, and `latency` counts from the transfer start.
  bool queuedDMA = false;
  // Every explicit transfer completes first (isImplicitMemory).
  bool drains = false;
};

// Explicit transfers in flight at once: one per DMA channel and staging window.
constexpr unsigned kMaxQueuedDMA = kAtlasDMAChannels;

// A tile is 32 x 32 elements: one M register of FP8 or a pair of BF16.
int registersOf(bool fp8) { return fp8 ? 1 : 2; }

// Work that holds the frontend from start to finish.
Cost blocking(Unit unit, int64_t cycles) {
  return {unit, cycles, cycles, cycles};
}

// A VMEM store or load of `halves` tile registers, back to back, as a spill
// save or reload issues them.
int64_t vmemCycles(unsigned halves, bool store,
                   const timing::TimingProvider &provider) {
  const auto &t = provider.machineTiming.at(store ? "vstore" : "vload");
  return int64_t(halves - 1) * t.interval + t.latency;
}

// Scalar slots lower to materializeScalar(byte address), then LW or SW.
// Before slot allocation the address is unknown, so include both LUI and
// ADDI; an allocated address uses the same zero-part elision as lowering.
// These are serial proposal costs, not additional overlap admissions.
int64_t scalarSlotCycles(bool store, int64_t word,
                         const timing::TimingProvider &provider) {
  if (provider.scalarSlotProposalCycles > 0)
    return provider.scalarSlotProposalCycles;
  uint32_t address = static_cast<uint32_t>(word) * 4;
  uint32_t upper = ((static_cast<uint64_t>(address) + 0x800) >> 12) & 0xfffff;
  bool unknown = word < 0;
  int64_t cycles = 0;
  auto append = [&](const char *name) {
    const auto &t = provider.machineTiming.at(name);
    cycles += std::max(t.latency, t.interval);
  };
  if (unknown || upper)
    append("lui");
  if (unknown || (address & 0xfff) || !upper)
    append("addi");
  append(store ? "sw" : "lw");
  return cycles;
}

Cost packCost(const timing::TimingProvider &provider) {
  // Pack, then relayout the FP8 tile through VMEM in the same one-row
  // scalar loop as lowerPack. Each store hides behind the next word's load,
  // so a row takes a load per load-path interval, then the last word's load
  // and store, and the back-edge, whose delay slot takes the destination step.
  const auto &pack = provider.machineTiming.at("vpack.bf16.fp8");
  const auto &vstore = provider.machineTiming.at("vstore");
  const auto &lw = provider.machineTiming.at("lw");
  const auto &sw = provider.machineTiming.at("sw");
  const auto &vload = provider.machineTiming.at("vload");
  const int rowWords = kAtlasMregRowBytes / 4;
  // Policy: this proposal omits general scalar setup and assumes the
  // destination increment can fill the slot when filling is enabled.
  // The physical scheduler separately proves an actual available filler.
  const int rowControl = provider.frontendSpacing.successorGap() +
      (timing::fillsDelaySlots(provider) ? 0 : provider.frontendSpacing.ordinaryGap);
  int64_t row = int64_t(rowWords - 1) * lw.interval + lw.latency + sw.latency + rowControl;
  return blocking(Unit::Lsu, int64_t(pack.latency) + vstore.latency + kAtlasMregRows * row + vload.latency);
}

// An operation costs the machine operations it lowers to, as they issue in
// order on the unit doing its work.
Cost costOf(Operation *op, const timing::TimingProvider &provider, int transfer) {
  const int registerWords = kAtlasRegisterBytes / 4;
  auto onUnit = [&](const std::string &name, Unit unit) {
    const auto &t = provider.machineTiming.at(name);
    return Cost{unit, t.interval, t.latency};
  };
  auto onMxu = [&](StringRef name, unsigned unit) {
    return onUnit((name + ".mxu" + Twine(unit)).str(),
                  unit ? Unit::Mxu1 : Unit::Mxu0);
  };
  auto onVpu = [&](const std::string &name) { return onUnit(name, Unit::Vpu); };
  // Boundary tiles move one register at a time, each transfer waited for
  // before the next instruction issues.
  if (isa<VirtualInputBF16Op, VirtualInputFP8Op>(op)) {
    const auto &vload = provider.machineTiming.at("vload");
    return blocking(Unit::Dma, registersOf(isa<VirtualInputFP8Op>(op)) * (int64_t(transfer) + vload.latency));
  }
  if (isa<VirtualOutputBF16Op>(op)) {
    const auto &vstore = provider.machineTiming.at("vstore");
    return blocking(Unit::Dma, registersOf(false) * (int64_t(vstore.latency) + transfer));
  }
  // An atomic load's tile is ready once its transfer lands and its VLOADs
  // (one per register) finish. A consumer cannot start before then, so the
  // list scheduler puts independent work between the load and its MXU, VPU,
  // pack or store user, where insert-atlas-dma-waits will put the DMA.WAIT.
  if (isa<VirtualDMALoadBF16Op, VirtualDMALoadFP8Op>(op)) {
    const auto &vload = provider.machineTiming.at("vload");
    int registers = registersOf(isa<VirtualDMALoadFP8Op>(op));
    int64_t loaded = int64_t(registers - 1) * vload.interval + vload.latency;
    return {Unit::Dma, transfer, int64_t(transfer) + loaded, 1, true};
  }
  if (isa<VirtualDMAStoreBF16Op, VirtualDMAStoreFP8Op>(op)) {
    const auto &vstore = provider.machineTiming.at("vstore");
    // VSTOREs into the staging window, then the transfer once they land.
    int registers = registersOf(isa<VirtualDMAStoreFP8Op>(op));
    int64_t staged = int64_t(registers - 1) * vstore.interval + vstore.latency;
    return {Unit::Dma, transfer, transfer, staged + 1, true};
  }
  if (isAtlasSpillStore(op) || isAtlasSpillLoad(op)) {
    Operation *writer = atlasSpillSaved(op).getDefiningOp();
    int64_t cycles = vmemCycles(
        atlasSpillWords(writer) / registerWords, isAtlasSpillStore(op), provider);
    return blocking(Unit::Dma, cycles);
  }
  if (isAtlasSlotAccess(op)) {
    const auto &vload = provider.machineTiming.at("vload");
    const auto &vstore = provider.machineTiming.at("vstore");
    auto slot = atlasAccessedSlot(op).getDefiningOp<VirtualSlotOp>();
    unsigned words = slot.getWords();
    int64_t cycles;
    if (words == 1) {
      cycles = scalarSlotCycles(isa<VirtualSaveOp>(op), slot.getSlotWordAttr().getInt(), provider);
      if (auto copy = dyn_cast<VirtualSlotCopyOp>(op)) {
        // A fixed slot proposal covers the whole copy; derived costs include both accesses.
        if (provider.scalarSlotProposalCycles == 0) {
          auto destination = copy.getDst().getDefiningOp<VirtualSlotOp>();
          cycles += scalarSlotCycles(true, destination.getSlotWordAttr().getInt(), provider);
        }
      }
    } else {
      cycles = int64_t(words / registerWords) * (int64_t(vload.latency) + vstore.latency);
    }
    return blocking(Unit::Dma, cycles);
  }
  if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op)) {
    unsigned unit = unitOf(load.getWeight().getType());
    Cost push = onMxu("vmatpush.weight", unit);
    push.unit = unit ? Unit::Weights1 : Unit::Weights0;
    return push;
  }
  if (isa<VirtualMXULoadAccFP8Op, VirtualMXULoadAccBF16Op>(op))
    return onMxu(isa<VirtualMXULoadAccFP8Op>(op) ? "vmatpush.acc.fp8"
                                                 : "vmatpush.acc.bf16",
                 unitOf(op->getResult(1).getType()));
  if (auto reset = dyn_cast<VirtualMXUResetOp>(op))
    return onMxu("vmatmul", unitOf(reset.getAcc().getType()));
  if (auto acc = dyn_cast<VirtualMXUAccumulateOp>(op))
    return onMxu("vmatmul.acc", unitOf(acc.getAcc().getType()));
  if (Value acc = virtualMXUReadoutAcc(op))
    return onMxu(isa<VirtualMXUReadoutBF16Op>(op) ? "vmatpop.bf16.acc"
                                                  : "vmatpop.fp8.acc",
                 unitOf(acc.getType()));
  if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op)) {
    // Push, multiply, and pop, each waiting for the one before.
    unsigned unit = matmul.getUnit();
    Cost push = onMxu("vmatpush.weight", unit);
    Cost multiply = onMxu("vmatmul", unit);
    Cost pop = onMxu("vmatpop.bf16.acc", unit);
    return blocking(unit ? Unit::Mxu1 : Unit::Mxu0, int64_t(push.latency) + multiply.latency + pop.latency);
  }
  if (isa<VirtualVPUUnaryOp, VirtualVPUBinaryOp, VirtualVPUReduceOp>(op))
    return onVpu(
        timing::vpuMnemonic(op->getAttrOfType<StringAttr>("kind").str()));
  if (isa<VirtualVPUSplatOp>(op))
    return onVpu("vli.all");
  if (isa<VirtualXLUTransposeOp>(op)) {
    return onUnit("vtrpose.xlu", Unit::Xlu);
  }
  if (isa<VirtualPackRoundTripBF16Op>(op)) {
    const auto &pack = provider.machineTiming.at("vpack.bf16.fp8");
    const auto &unpack = provider.machineTiming.at("vunpack.fp8.bf16");
    const auto &scale = provider.machineTiming.at("seli");
    return blocking(Unit::Vpu,
                    int64_t(pack.latency) + unpack.latency + 2 * scale.latency);
  }
  if (isa<VirtualPackFP8Op>(op))
    return packCost(provider);
  // Scalar control, a DMA wait, or nothing at all.
  const auto &scalar = provider.machineTiming.at("addi");
  return {Unit::Scalar, 1, scalar.latency};
}

// Register kinds as the allocator colors them; counts are indexed by kind.
constexpr std::array<RegisterKind, 3> kKinds = {
    RegisterKind::BF16, RegisterKind::FP8, RegisterKind::Scalar};
using KindCounts = std::array<int, kKinds.size()>;

unsigned indexOf(RegisterKind kind) { return static_cast<unsigned>(kind); }

// One validated set of provider operands serves scheduling and graph metadata.
FailureOr<std::vector<Cost>> costsOf(const BlockGraph &g, const timing::TimingProvider &provider) {
  std::vector<Cost> costs;
  int64_t cycleBound = 0;
  for (Operation *op : g.nodes) {
    int transfer = 0;
    if (isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op,
            VirtualDMALoadBF16Op, VirtualDMALoadFP8Op,
            VirtualDMAStoreBF16Op, VirtualDMAStoreFP8Op>(op)) {
      if (!provider.dmaEstimate)
        return op->emitOpError("timing provider has no DMA proposal estimator");
      bool store = isa<VirtualOutputBF16Op, VirtualDMAStoreBF16Op, VirtualDMAStoreFP8Op>(op);
      int registers = isa<VirtualDMALoadBF16Op, VirtualDMAStoreBF16Op>(op) ? registersOf(false) : 1;
      auto estimate = provider.dmaEstimate(int64_t(registers) * kAtlasRegisterBytes, store);
      if (!estimate.error.empty()) return op->emitOpError(estimate.error);
      if (estimate.value < 1) return op->emitOpError("provider returned a nonpositive DMA proposal estimate");
      transfer = estimate.value;
    }
    Cost cost = costOf(op, provider, transfer);
    // Bound heights and cycle arithmetic by conservatively serializing all cost fields.
    cycleBound += cost.busy + cost.latency + cost.frontend;
    if (cycleBound > std::numeric_limits<int>::max())
      return op->emitOpError("virtual block cost exceeds nonnegative cycle domain");
    cost.drains = isImplicitMemory(op);
    costs.push_back(cost);
  }
  return costs;
}

// What spilling one value of `kind` costs: a save and `reloads` reloads, as
// costOf charges the operations --spill-atlas-virtual inserts.
struct SpillCycles {
  std::array<int64_t, kKinds.size()> save, reload;
  int64_t of(RegisterKind kind, int reloads) const {
    return save[indexOf(kind)] + int64_t(reloads) * reload[indexOf(kind)];
  }
};

SpillCycles spillCycles(const timing::TimingProvider &provider) {
  SpillCycles spill{};
  for (RegisterKind kind : kKinds) {
    unsigned k = indexOf(kind);
    auto cycles = [&](bool store) {
      return kind == RegisterKind::Scalar ? scalarSlotCycles(store, -1, provider)
          : vmemCycles(registersOf(kind == RegisterKind::FP8), store, provider);
    };
    spill.save[k] = cycles(true);
    spill.reload[k] = cycles(false);
  }
  return spill;
}

SmallVector<Value, 4> distinctOperands(Operation *op) {
  SmallVector<Value, 4> values;
  for (Value operand : op->getOperands())
    if (!llvm::is_contained(values, operand))
      values.push_back(operand);
  return values;
}

// Live values of each kind as a block's operations are placed, counted as
// the allocator's interference graph counts them: an operation's results
// interfere with its operands, and a loaded tile holds registers from its
// first use, where lowering stages it. Values beyond `capacity` are spilled.
class Pressure {
public:
  Pressure(Block &block, ArrayRef<Operation *> nodes, const Liveness &liveness,
           const KindCounts &capacity, const SpillCycles &spill)
      : capacity(capacity), spill(spill) {
    for (auto [index, op] : llvm::enumerate(nodes))
      for (Value operand : distinctOperands(op)) {
        ++remaining[operand];
        lastUse[operand] = index;
      }
    // Values the block hands on stay live throughout.
    kept.insert(liveness.getLiveOut(&block).begin(),
                liveness.getLiveOut(&block).end());
    for (Value operand : block.getTerminator()->getOperands())
      kept.insert(operand);
    llvm::DenseSet<Value> entry(liveness.getLiveIn(&block).begin(),
                                liveness.getLiveIn(&block).end());
    entry.insert(block.args_begin(), block.args_end());
    for (Value value : entry)
      if (needed(value))
        enter(value);
  }

  // Live values while `op` runs, if it is placed next.
  KindCounts during(Operation *op) const {
    KindCounts counts = live;
    for (Value result : op->getResults())
      if (!isVirtualDMALoadedTile(result))
        count(counts, result, 1);
    for (Value operand : distinctOperands(op))
      if (staging.contains(operand))
        count(counts, operand, 1);
    return counts;
  }

  // Live values once `op` has run, if it is placed next.
  KindCounts after(Operation *op) const {
    KindCounts counts = live;
    for (Value operand : distinctOperands(op)) {
      bool dies = remaining.lookup(operand) == 1 && !kept.contains(operand);
      if (staging.contains(operand) != dies)
        count(counts, operand, dies ? -1 : 1);
    }
    for (Value result : op->getResults())
      if (needed(result) && !isVirtualDMALoadedTile(result))
        count(counts, result, 1);
    return counts;
  }

  // The spill cycles placing `op` next adds: for each value live while it
  // runs beyond both the registers and this order's peak so far (whose
  // spills are already counted), --spill-atlas-virtual saves the
  // longest-lived value and reloads it before each of its uses. The live
  // value of that kind used last in source order stands in for it.
  int64_t spillCost(Operation *op) const {
    KindCounts counts = during(op);
    int64_t cycles = 0;
    for (unsigned k = 0; k < kKinds.size(); ++k) {
      int spills = counts[k] - std::max(capacity[k], peak[k]);
      if (spills <= 0)
        continue;
      Value victim;
      for (Value value : values)
        if (registerKindOf(value) == kKinds[k] &&
            (!victim || lastUse.lookup(value) > lastUse.lookup(victim)))
          victim = value;
      int reloads = victim ? std::max(1u, remaining.lookup(victim)) : 1;
      cycles += spills * spill.of(kKinds[k], reloads);
    }
    return cycles;
  }

  void place(Operation *op) {
    spilled += spillCost(op);
    KindCounts counts = during(op);
    for (unsigned k = 0; k < kKinds.size(); ++k)
      peak[k] = std::max(peak[k], counts[k]);
    live = after(op);
    for (Value operand : distinctOperands(op)) {
      if (remaining.lookup(operand) == 1 && !kept.contains(operand))
        values.erase(operand);
      else if (staging.contains(operand))
        values.insert(operand);
      staging.erase(operand);
    }
    for (Value result : op->getResults())
      if (needed(result) && isVirtualDMALoadedTile(result))
        staging.insert(result);
      else if (needed(result) && hasRegister(result))
        values.insert(result);
    for (Value operand : distinctOperands(op))
      --remaining[operand];
  }

  // The spill cycles of the operations placed so far.
  int64_t spilled = 0;

private:
  bool needed(Value value) const {
    return kept.contains(value) || remaining.lookup(value) > 0;
  }

  static bool hasRegister(Value value) { return registerKindOf(value).has_value(); }

  static void count(KindCounts &counts, Value value, int delta) {
    if (std::optional<RegisterKind> kind = registerKindOf(value))
      counts[indexOf(*kind)] += delta;
  }

  void enter(Value value) {
    count(live, value, 1);
    if (hasRegister(value))
      values.insert(value);
  }

  KindCounts capacity;
  SpillCycles spill;
  llvm::DenseMap<Value, unsigned> remaining;
  // Each value's last user, by its index in source order.
  llvm::DenseMap<Value, unsigned> lastUse;
  llvm::DenseSet<Value> kept;
  // The values `live` counts, and the loaded tiles not yet used.
  llvm::DenseSet<Value> values, staging;
  KindCounts live{};
  KindCounts peak{};
};

// In-order issue onto functional units: an operation starts once the one
// before it has issued, everything it depends on has finished, and its unit
// is free.
class Machine {
public:
  Machine(const BlockGraph &g, ArrayRef<Cost> costs)
      : g(g), costs(costs), finished(g.nodes.size(), 0) {}

  int start(unsigned node) const {
    const Cost &cost = costs[node];
    int cycle = cost.queuedDMA ? frontend : std::max(frontend, unitFree[unit(node)]);
    if (cost.drains)
      for (int end : transfers)
        cycle = std::max(cycle, end);
    for (unsigned pred : g.preds[node])
      cycle = std::max(cycle, finished[pred]);
    if (cost.queuedDMA) {
      // With every channel busy, a launch issues once the earliest ends.
      SmallVector<int, kMaxQueuedDMA> busy;
      for (int end : transfers)
        if (end > cycle)
          busy.push_back(end);
      if (busy.size() >= kMaxQueuedDMA) {
        llvm::sort(busy);
        cycle = busy[busy.size() - kMaxQueuedDMA];
      }
    }
    return cycle;
  }

  // When the frontend can issue again if `node` starts at `cycle`.
  int frontendAfter(unsigned node, int cycle) const {
    return cycle + costs[node].frontend;
  }

  void place(unsigned node) {
    int cycle = start(node);
    const Cost &cost = costs[node];
    // A queued transfer starts after its staging and every earlier transfer.
    int work = cost.queuedDMA
                   ? std::max<int64_t>(cycle + cost.frontend - 1, unitFree[unit(node)])
                   : cycle;
    finished[node] = work + cost.latency;
    unitFree[unit(node)] = work + cost.busy;
    if (cost.queuedDMA) {
      llvm::erase_if(transfers, [&](int end) { return end <= cycle; });
      transfers.push_back(work + cost.busy);
    }
    frontend = frontendAfter(node, cycle);
    makespan = std::max(makespan, finished[node]);
  }

  int makespan = 0;

private:
  unsigned unit(unsigned node) const {
    return static_cast<unsigned>(costs[node].unit);
  }

  const BlockGraph &g;
  ArrayRef<Cost> costs;
  std::vector<int> finished;
  std::array<int, kUnits> unitFree{};
  // When each explicit transfer placed so far completes.
  SmallVector<int> transfers;
  int frontend = 0;
};

class BlockScheduler {
public:
  BlockScheduler(Block &block, const BlockGraph &g, const Liveness &liveness,
                 std::vector<Cost> nodeCosts, const SpillCycles &spill)
      : block(block), g(g), liveness(liveness), costs(std::move(nodeCosts)), spill(spill) {
    unsigned n = g.nodes.size();
    // The earliest output each value reaches through non-state operands, or n
    // for none. Among equal starts, prefer it: shared operands load for their
    // first tile, and an available tile finishes before another tile's first
    // layer advances.
    firstOutput.assign(n, n);
    llvm::DenseMap<Operation *, unsigned> index;
    for (unsigned i = 0; i < n; ++i)
      index[g.nodes[i]] = i;
    for (unsigned i = n; i-- > 0;) {
      Operation *op = g.nodes[i];
      if (isa<VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op,
              VirtualOutputBF16Op>(op))
        firstOutput[i] = i;
      for (Value operand : op->getOperands()) {
        if (isa<VirtualStateType>(operand.getType()))
          continue;
        auto found = index.find(operand.getDefiningOp());
        if (found != index.end())
          firstOutput[found->second] =
              std::min(firstOutput[found->second], firstOutput[i]);
      }
    }
    // The longest path from each node to the block's end, counting its own
    // work. Edges point forward, so successors are done first.
    heights.assign(n, 0);
    for (unsigned i = n; i-- > 0;) {
      int below = 0;
      for (unsigned s : g.succs[i])
        below = std::max(below, heights[s]);
      heights[i] = costs[i].latency + below;
    }
  }

  int makespan(ArrayRef<unsigned> order) const {
    Machine machine(g, costs);
    for (unsigned node : order)
      machine.place(node);
    return machine.makespan;
  }

  // The spill cycles `order` needs (Pressure::spillCost).
  int64_t spillCost(ArrayRef<Operation *> order, const KindCounts &capacity) const {
    Pressure pressure(block, g.nodes, liveness, capacity, spill);
    for (Operation *op : order)
      pressure.place(op);
    return pressure.spilled;
  }

  // Of the operations whose dependences are placed, take the one that starts
  // first, counting the spills it adds as delay: a value beyond `capacity`
  // costs VMEM saves and reloads, worth it only when the alternatives wait
  // longer (Pressure::spillCost). Then the one feeding the earliest output;
  // then the one that frees the frontend first, so a blocking transfer waits
  // for work that can run beside it; then the one with the longest path to the
  // block's end; then the earliest in source order.
  // A load starts as soon as the frontend is free, so with `lookahead` loaded
  // tiles awaiting their first use, another load waits while anything else
  // is ready: a boundary tile must not drain transfers whose users all come
  // after it, and each holds a staging window until its first use.
  std::vector<unsigned> listSchedule(const KindCounts &capacity,
                                     unsigned lookahead) const {
    Pressure pressure(block, g.nodes, liveness, capacity, spill);
    llvm::SmallDenseSet<Value> unused;
    auto isLoad = [](Operation *op) {
      return isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op>(op);
    };
    return build([&](const Machine &machine, ArrayRef<unsigned> ready) {
      auto held = [&](unsigned node) {
        return unused.size() >= lookahead && isLoad(g.nodes[node]);
      };
      SmallVector<unsigned> eligible;
      llvm::copy_if(ready, std::back_inserter(eligible),
                    [&](unsigned node) { return !held(node); });
      if (eligible.empty())
        eligible.assign(ready.begin(), ready.end());
      auto key = [&](unsigned node) {
        Operation *op = g.nodes[node];
        int start = machine.start(node);
        return std::tuple(start + pressure.spillCost(op), firstOutput[node],
                          machine.frontendAfter(node, start), -heights[node],
                          node);
      };
      unsigned best = eligible.front();
      auto bestKey = key(best);
      for (unsigned node : ArrayRef(eligible).drop_front()) {
        auto nodeKey = key(node);
        if (nodeKey < bestKey) {
          best = node;
          bestKey = nodeKey;
        }
      }
      Operation *op = g.nodes[best];
      for (Value operand : op->getOperands())
        unused.erase(operand);
      if (isLoad(op))
        unused.insert(op->getResult(1));
      pressure.place(op);
      return best;
    });
  }

  // A uniformly random legal order, for soundness testing.
  std::vector<unsigned> randomOrder(unsigned seed) const {
    std::mt19937 rng(seed);
    return build([&](const Machine &, ArrayRef<unsigned> ready) {
      return ready[std::uniform_int_distribution<size_t>(0, ready.size() - 1)(
          rng)];
    });
  }

private:
  // Place every node in the order `pick` chooses from the ready ones.
  template <typename Pick>
  std::vector<unsigned> build(Pick pick) const {
    unsigned n = g.nodes.size();
    std::vector<unsigned> waiting(n), order;
    SmallVector<unsigned> ready;
    for (unsigned i = 0; i < n; ++i)
      if ((waiting[i] = g.preds[i].size()) == 0)
        ready.push_back(i);
    Machine machine(g, costs);
    while (!ready.empty()) {
      unsigned node = pick(machine, ready);
      ready.erase(llvm::find(ready, node));
      machine.place(node);
      order.push_back(node);
      for (unsigned s : g.succs[node])
        if (--waiting[s] == 0)
          ready.push_back(s);
    }
    assert(order.size() == n && "the dependence graph is acyclic");
    return order;
  }

  Block &block;
  const BlockGraph &g;
  const Liveness &liveness;
  std::vector<Cost> costs;
  SpillCycles spill;
  std::vector<int> heights;
  std::vector<unsigned> firstOutput;
};

// Move `ops` before the terminator in order, and rethread the state chain
// through the new order.
void reorder(Block &block, ArrayRef<Operation *> ops) {
  Operation *terminator = block.getTerminator();
  for (Operation *op : ops)
    op->moveBefore(terminator);
  Value state;
  if (!block.isEntryBlock())
    state = block.getArgument(0);
  for (Operation &op : block) {
    for (OpOperand &operand : op.getOpOperands())
      if (isa<VirtualStateType>(operand.get().getType()))
        operand.set(state);
    for (Value result : op.getResults())
      if (isa<VirtualStateType>(result.getType()))
        state = result;
  }
}

// Reorder one block when the list schedule is estimated to finish before the
// source order, spills included: --spill-atlas-virtual recovers pressure
// beyond the registers, at a cost. The schedule is the best list schedule
// over a few load lookaheads. A random order is taken as it is.
LogicalResult scheduleBlock(Block &block, unsigned index,
                            const Liveness &liveness,
                            const KindCounts &capacity,
                            std::optional<unsigned> randomSeed,
                            const timing::TimingProvider &provider) {
  FailureOr<BlockGraph> graph = buildGraph(block);
  if (failed(graph))
    return failure();
  const BlockGraph &g = *graph;
  auto costs = costsOf(g, provider);
  if (failed(costs)) return failure();
  BlockScheduler scheduler(block, g, liveness, std::move(*costs),
                           spillCycles(provider));
  auto opsOf = [&](ArrayRef<unsigned> order) {
    SmallVector<Operation *> ops;
    for (unsigned node : order)
      ops.push_back(g.nodes[node]);
    return ops;
  };
  if (randomSeed) {
    reorder(block, opsOf(scheduler.randomOrder(*randomSeed + index)));
    return success();
  }

  auto cycles = [&](ArrayRef<unsigned> order) {
    return scheduler.makespan(order) + scheduler.spillCost(opsOf(order), capacity);
  };
  std::vector<unsigned> order;
  int64_t best = 0;
  for (unsigned lookahead : {1u, 2u, 4u, kMaxQueuedDMA}) {
    std::vector<unsigned> candidate = scheduler.listSchedule(capacity, lookahead);
    int64_t estimate = cycles(candidate);
    if (order.empty() || estimate < best) {
      order = std::move(candidate);
      best = estimate;
    }
  }
  std::vector<unsigned> source(g.nodes.size());
  std::iota(source.begin(), source.end(), 0u);
  if (best < cycles(source))
    reorder(block, opsOf(order));
  return success();
}

struct ScheduleAtlasVirtualPass
    : PassWrapper<ScheduleAtlasVirtualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleAtlasVirtualPass)

  ScheduleAtlasVirtualPass() = default;
  ScheduleAtlasVirtualPass(const ScheduleAtlasVirtualPass &other)
      : PassWrapper(other), orderApplied(other.orderApplied),
        rememberOptions(other.rememberOptions) {}

  // Spilling and PACK repair expand an explicit order without changing it.
  // Its node IDs and graph output still refer to the prepared search graph.
  bool orderApplied = false;
  bool rememberOptions = false;

  static constexpr StringLiteral scheduleOptions = "atlas.virtual_schedule_options";

  LogicalResult restoreOptions(ModuleOp module) {
    auto options = module->getAttrOfType<DictionaryAttr>(scheduleOptions);
    if (!options)
      return success();
    auto savedProvider = options.getAs<StringAttr>("provider");
    auto savedSeed = options.getAs<IntegerAttr>("random_seed");
    auto savedOrder = options.getAs<StringAttr>("order_file");
    auto savedGraph = options.getAs<StringAttr>("graph_output");
    if (!savedProvider || !savedSeed || !savedOrder || !savedGraph)
      return module.emitError("virtual schedule options are incomplete");
    provider = savedProvider.str();
    randomSeed = savedSeed.getInt();
    orderFile = savedOrder.str();
    graphOutput = savedGraph.str();
    return success();
  }

  void saveOptions(ModuleOp module, StringRef selectedProvider) {
    Builder builder(module.getContext());
    module->setAttr(scheduleOptions, builder.getDictionaryAttr({
        builder.getNamedAttr("provider", builder.getStringAttr(selectedProvider)),
        builder.getNamedAttr("random_seed", builder.getI64IntegerAttr(randomSeed)),
        builder.getNamedAttr("order_file", builder.getStringAttr(orderFile)),
        builder.getNamedAttr("graph_output", builder.getStringAttr(graphOutput))}));
  }

  StringRef getArgument() const final { return "schedule-atlas-virtual"; }
  StringRef getDescription() const final {
    return "Reorder each virtual Atlas block onto its functional units before "
           "spilling and register allocation";
  }

  Option<int> randomSeed{
      *this, "random-seed",
      llvm::cl::desc("Pick a random legal order instead, for soundness "
                     "testing; negative disables"),
      llvm::cl::init(-1)};

  Option<std::string> provider{
      *this, "provider",
      llvm::cl::desc("Timing provider (default: atlas.timing_provider, else builtin)"),
      llvm::cl::init("")};

  Option<std::string> orderFile{
      *this, "order-file", llvm::cl::desc("Explicit per-block topological orders as JSON"),
      llvm::cl::init("")};

  Option<std::string> graphOutput{
      *this, "graph-output", llvm::cl::desc("Write dependence graphs and selected orders as JSON"),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (orderApplied && failed(restoreOptions(module)))
      return signalPassFailure();
    if (failed(verifyAtlasVirtualModule(module)))
      return signalPassFailure();
    auto refuse = [&](const Twine &why) { module.emitError(why); signalPassFailure(); };
    FailureOr<timing::TimingProvider> selected = selectAtlasTimingProvider(module, provider.getValue());
    if (failed(selected))
      return signalPassFailure();
    if (!orderFile.empty() && randomSeed >= 0)
      return refuse("order-file excludes random-seed");
    std::optional<llvm::json::Value> requested;
    if (!orderFile.empty() && !orderApplied) {
      auto file = llvm::MemoryBuffer::getFile(orderFile);
      if (!file)
        return refuse(Twine("cannot read order-file: ") + file.getError().message());
      auto parsed = llvm::json::parse((*file)->getBuffer());
      if (!parsed)
        return refuse(Twine("invalid order-file JSON: ") + llvm::toString(parsed.takeError()));
      requested = std::move(*parsed);
      if (!requested->getAsArray())
        return refuse("order-file must contain an array of block permutations");
    }
    llvm::json::Array graphs;
    const bool exportGraph = !graphOutput.empty() && !orderApplied;
    unsigned blockIndex = 0;
    std::optional<unsigned> seed;
    if (randomSeed >= 0)
      seed = static_cast<unsigned>(randomSeed);
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      reuseAtlasImmutableWeights(function);
      KindCounts capacity;
      RegisterCapacities capacities = registerCapacities(function);
      for (RegisterKind kind : kKinds)
        capacity[indexOf(kind)] = capacities[indexOf(kind)];
      Liveness liveness(function);
      unsigned index = 0;
      for (Block &block : function.getBody()) {
        std::optional<BlockGraph> graph;
        if (requested || exportGraph) {
          auto built = buildGraph(block);
          if (failed(built))
            return signalPassFailure();
          graph = std::move(*built);
        }
        if (requested) {
          const auto &blocks = *requested->getAsArray();
          const auto *order = blockIndex < blocks.size() ? blocks[blockIndex].getAsArray() : nullptr;
          if (!order || order->size() != graph->nodes.size())
            return refuse("order-file must contain one complete permutation per block");
          SmallVector<Operation *> ops;
          std::vector<bool> placed(graph->nodes.size(), false);
          for (const auto &value : *order) {
            auto node = value.getAsInteger();
            if (!node || *node < 0 || static_cast<uint64_t>(*node) >= placed.size() || placed[*node])
              return refuse("order-file contains an invalid or duplicate node");
            for (unsigned pred : graph->preds[*node])
              if (!placed[pred])
                return refuse("order-file violates a dependence");
            placed[*node] = true;
            ops.push_back(graph->nodes[*node]);
          }
          reorder(block, ops);
        } else if (!(orderApplied && !orderFile.empty()) &&
                   failed(scheduleBlock(block, index, liveness,
                                        capacity, seed, *selected))) {
          return signalPassFailure();
        }
        if (exportGraph) {
          llvm::json::Array preds, names, order;
          llvm::DenseMap<Operation *, unsigned> ids;
          for (unsigned node = 0; node < graph->nodes.size(); ++node) {
            ids[graph->nodes[node]] = node;
            llvm::json::Array incoming;
            for (unsigned pred : graph->preds[node])
              incoming.push_back(pred);
            preds.push_back(std::move(incoming));
            names.push_back(graph->nodes[node]->getName().getStringRef());
          }
          for (Operation *op : freeOperations(block)) {
            unsigned node = ids.lookup(op);
            order.push_back(node);
          }
          graphs.push_back(llvm::json::Object{{"preds", std::move(preds)}, {"operations", std::move(names)},
              {"order", std::move(order)}});
        }
        ++index;
        ++blockIndex;
      }
    }
    if (requested && requested->getAsArray()->size() != blockIndex) {
      module.emitError("order-file has extra block permutations");
      return signalPassFailure();
    }
    // Every order the dependence graph allows must verify, so a failure here
    // is a dependence the graph is missing.
    if (failed(verifyAtlasVirtualModule(module))) {
      module.emitError("schedule-atlas-virtual produced an order that does "
                       "not verify");
      return signalPassFailure();
    }
    if (exportGraph) {
      std::error_code error;
      llvm::raw_fd_ostream output(graphOutput, error);
      if (error)
        return refuse(Twine("cannot write graph-output: ") + error.message());
      output << llvm::json::Value(std::move(graphs)) << "\n";
      output.flush();
      if (output.has_error())
        return refuse("cannot write graph-output");
    }
    if (rememberOptions)
      saveOptions(module, selected->id);
    if (orderApplied)
      module->removeAttr(scheduleOptions);
  }
};

} // namespace

void mlir::atlas::registerScheduleAtlasVirtualPass() {
  PassRegistration<ScheduleAtlasVirtualPass>();
}

std::unique_ptr<Pass> mlir::atlas::createScheduleAtlasVirtualPass(
    StringRef provider, int randomSeed, StringRef orderFile,
    StringRef graphOutput, bool orderApplied,
    bool rememberOptions) {
  auto pass = std::make_unique<ScheduleAtlasVirtualPass>();
  pass->provider = provider.str();
  pass->randomSeed = randomSeed;
  pass->orderFile = orderFile.str();
  pass->graphOutput = graphOutput.str();
  pass->orderApplied = orderApplied;
  pass->rememberOptions = rememberOptions;
  return pass;
}
