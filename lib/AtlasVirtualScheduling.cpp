// Pre-allocation scheduling of virtual Atlas operations. Each block is list
// scheduled onto the functional units its operations occupy, keeping every
// SSA and implicit-state dependence. Costs are rough on purpose: registers,
// delays, and delay slots do not exist until the virtual-to-machine lowering,
// and the machine passes after it time the real instructions.

#include "Atlas/AtlasVirtualScheduling.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasSpilling.h"
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
#include <algorithm>
#include <array>
#include <cassert>
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

// An explicit table: `Pure` is not trusted, because legacy virtual_mxu_matmul
// and virtual_pack_fp8 are declared Pure yet have ordering obligations. DMA
// ordering and slot limits are edges of their own (buildGraph, SlotPool).
// An operation missing from the table is a full barrier.
Effects effectsOf(Operation *op) {
  Effects e;
  e.reads = bit(kBarrier);
  if (isa<VirtualPackFP8Op>(op))
    e.writes = bit(kPack);
  else if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op))
    e.reads |= bit(kUnit0 + unitOf(load.getWeight().getType()));
  else if (isa<VirtualMXULoadAccFP8Op, VirtualMXULoadAccBF16Op,
               VirtualMXUResetOp>(op))
    e.reads |= bit(kUnit0 + unitOf(op->getResult(1).getType()));
  else if (auto acc = dyn_cast<VirtualMXUAccumulateOp>(op))
    e.reads |= bit(kUnit0 + unitOf(acc.getAcc().getType()));
  else if (isa<VirtualMXUReadoutBF16Op, VirtualMXUReadoutFP8Op>(op))
    e.reads |= bit(kUnit0 + unitOf(op->getOperand(1).getType()));
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
  int busy;
  int latency;
  int frontend = 1;
  // An atomic DMA lowers to a launch the frontend issues without waiting for
  // the engine; its transfer (`busy` cycles) queues behind earlier ones after
  // `frontend` staging cycles, and `latency` counts from the transfer start.
  bool queuedDMA = false;
  // Every explicit transfer completes first (isImplicitMemory).
  bool drains = false;
};

// Explicit transfers in flight at once: one per DMA channel and staging window.
constexpr unsigned kMaxQueuedDMA = 8;

// A machine operation's timing by npu_model's rules: cycles until its work
// is done, and until the same operation on other registers and slots can
// issue behind it on its unit.
struct MachineTiming {
  int latency;
  int interval;
};

MachineTiming timingOf(const std::string &name) {
  auto instruction = [&](int rd, int rs1, int rs2) {
    timing::Instr in;
    in.op = timing::findOp(name);
    assert(in.op && "the timing model knows every machine operation");
    in.rd = rd;
    in.rs1 = rs1;
    in.rs2 = rs2;
    return in;
  };
  // Two instances that share no register, slot, or BF16 pair.
  timing::Instr first = instruction(0, 2, 4), second = instruction(6, 8, 10);
  switch (first.op->opClass) {
  case timing::OpClass::WeightPush:
  case timing::OpClass::AccPushFp8:
  case timing::OpClass::AccPushBf16:
  case timing::OpClass::MatMul:
  case timing::OpClass::MatMulAcc:
  case timing::OpClass::PopFp8:
  case timing::OpClass::PopBf16:
    // MXU slot operands are 0 or 1.
    first = instruction(0, 0, 0);
    second = instruction(1, 2, 1);
    break;
  default:
    break;
  }
  timing::Footprint firstFootprint =
      timing::footprintOf(first, timing::unknownRegs());
  timing::Footprint secondFootprint =
      timing::footprintOf(second, timing::unknownRegs());
  timing::ReservationTable table;
  table.reserve(first, firstFootprint, 0);
  int interval = 1;
  while (!table.conflict(second, secondFootprint, interval).empty())
    ++interval;
  return {firstFootprint.doneAge + 1, interval};
}

// A tile is 32 x 32 elements: one M register of FP8 or a pair of BF16. An M
// register holds 32 lines.
int registersOf(bool fp8) { return fp8 ? 1 : 2; }
constexpr int kRegisterBytes = 32 * timing::kLineBytes;

Cost onMxu(StringRef name, unsigned unit) {
  MachineTiming t = timingOf((name + ".mxu" + Twine(unit)).str());
  return {unit ? Unit::Mxu1 : Unit::Mxu0, t.interval, t.latency};
}

Cost onVpu(const std::string &name) {
  MachineTiming t = timingOf(name);
  return {Unit::Vpu, t.interval, t.latency};
}

// Work that holds the frontend from start to finish.
Cost blocking(Unit unit, int cycles) { return {unit, cycles, cycles, cycles}; }

// An operation costs the machine operations it lowers to, as they issue in
// order on the unit doing its work.
Cost costOf(Operation *op) {
  static const MachineTiming vload = timingOf("vload");
  static const MachineTiming vstore = timingOf("vstore");
  static const int registerLoad = timing::dmaTransferCycles(kRegisterBytes, false);
  static const int registerStore = timing::dmaTransferCycles(kRegisterBytes, true);
  // Boundary tiles move one register at a time, each transfer waited for
  // before the next instruction issues.
  if (isa<VirtualInputBF16Op, VirtualInputFP8Op>(op))
    return blocking(Unit::Dma, registersOf(isa<VirtualInputFP8Op>(op)) *
                                   (registerLoad + vload.latency));
  if (isa<VirtualOutputBF16Op>(op))
    return blocking(Unit::Dma,
                    registersOf(false) * (vstore.latency + registerStore));
  // An atomic load's tile is ready once its transfer lands and its VLOADs
  // (one per register) finish. A consumer cannot start before then, so the
  // list scheduler puts independent work between the load and its MXU, VPU,
  // pack or store user, where insert-atlas-dma-waits will put the DMA.WAIT.
  if (isa<VirtualDMALoadBF16Op, VirtualDMALoadFP8Op>(op)) {
    int registers = registersOf(isa<VirtualDMALoadFP8Op>(op));
    int cycles = timing::dmaTransferCycles(registers * kRegisterBytes, false);
    int loaded = (registers - 1) * vload.interval + vload.latency;
    Cost cost{Unit::Dma, cycles, cycles + loaded};
    cost.queuedDMA = true;
    return cost;
  }
  if (isa<VirtualDMAStoreBF16Op, VirtualDMAStoreFP8Op>(op)) {
    // VSTOREs into the staging window, then the transfer once they land.
    int registers = registersOf(isa<VirtualDMAStoreFP8Op>(op));
    int staged = (registers - 1) * vstore.interval + vstore.latency;
    int cycles = timing::dmaTransferCycles(registers * kRegisterBytes, true);
    Cost cost{Unit::Dma, cycles, cycles, staged + 1};
    cost.queuedDMA = true;
    return cost;
  }
  if (isAtlasSpillStore(op) || isAtlasSpillLoad(op)) {
    Operation *writer = isAtlasSpillStore(op) ? op : op->getOperand(1).getDefiningOp();
    unsigned halves = atlasSpillWords(writer) / 256;
    auto timing = isAtlasSpillStore(op) ? vstore : vload;
    return blocking(Unit::Dma, (halves - 1) * timing.interval + timing.latency);
  }
  if (isAtlasSlotAccess(op)) {
    auto slot = op->getOperand(1).getDefiningOp<VirtualSlotOp>();
    unsigned words = slot.getWords();
    unsigned cycles = words == 1 ? 8 : (words / 256) * (vload.latency + vstore.latency);
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
  if (isa<VirtualMXUReadoutBF16Op, VirtualMXUReadoutFP8Op>(op))
    return onMxu(isa<VirtualMXUReadoutBF16Op>(op) ? "vmatpop.bf16.acc"
                                                  : "vmatpop.fp8.acc",
                 unitOf(op->getOperand(1).getType()));
  if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op)) {
    // Push, multiply, and pop, each waiting for the one before.
    unsigned unit = matmul.getUnit();
    return blocking(unit ? Unit::Mxu1 : Unit::Mxu0,
                    onMxu("vmatpush.weight", unit).latency +
                        onMxu("vmatmul", unit).latency +
                        onMxu("vmatpop.bf16.acc", unit).latency);
  }
  if (isa<VirtualVPUUnaryOp, VirtualVPUBinaryOp, VirtualVPUReduceOp>(op))
    return onVpu(
        timing::vpuMnemonic(op->getAttrOfType<StringAttr>("kind").str()));
  if (isa<VirtualVPUSplatOp>(op))
    return onVpu("vli.all");
  if (isa<VirtualXLUTransposeOp>(op)) {
    MachineTiming t = timingOf("vtrpose.xlu");
    return {Unit::Xlu, t.interval, t.latency};
  }
  if (isa<VirtualPackFP8Op>(op)) {
    // Pack, then relayout the FP8 tile through VMEM in a scalar loop of eight
    // words a row (lowerPack). Each store hides behind the next word's load,
    // so a row takes a load per load-path interval, then the last word's load
    // and store, the destination step, and the back-edge with its delay slot.
    static const MachineTiming lw = timingOf("lw"), sw = timingOf("sw");
    constexpr int rowWords = kPackRelayoutRowWords, rowControl = 3;
    int rows = registersOf(true) * kRegisterBytes / 4 / rowWords;
    int row = (rowWords - 1) * lw.interval + lw.latency + sw.latency + rowControl;
    return blocking(Unit::Lsu, timingOf("vpack.bf16.fp8").latency +
                                   vstore.latency + rows * row + vload.latency);
  }
  // Scalar control, a DMA wait, or nothing at all.
  return {Unit::Scalar, 1, timingOf("addi").latency};
}

// Register kinds as the allocator colors them; counts are indexed by kind.
constexpr std::array<RegisterKind, 3> kKinds = {
    RegisterKind::BF16, RegisterKind::FP8, RegisterKind::Scalar};
using KindCounts = std::array<int, kKinds.size()>;

unsigned indexOf(RegisterKind kind) { return static_cast<unsigned>(kind); }

std::optional<RegisterKind> kindOf(Type type) {
  if (isa<VirtualBF16Type>(type))
    return RegisterKind::BF16;
  if (isa<VirtualFP8Type>(type))
    return RegisterKind::FP8;
  if (type.isInteger(1) || type.isInteger(32))
    return RegisterKind::Scalar;
  return std::nullopt;
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
// interfere with its operands.
class Pressure {
public:
  Pressure(Block &block, ArrayRef<Operation *> nodes,
           const Liveness &liveness) {
    for (Operation *op : nodes)
      for (Value operand : distinctOperands(op))
        ++remaining[operand];
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
        count(live, value, 1);
  }

  // Live values while `op` runs, if it is placed next.
  KindCounts during(Operation *op) const {
    KindCounts counts = live;
    for (Value result : op->getResults())
      count(counts, result, 1);
    return counts;
  }

  // Live values once `op` has run, if it is placed next.
  KindCounts after(Operation *op) const {
    KindCounts counts = live;
    for (Value operand : distinctOperands(op))
      if (remaining.lookup(operand) == 1 && !kept.contains(operand))
        count(counts, operand, -1);
    for (Value result : op->getResults())
      if (needed(result))
        count(counts, result, 1);
    return counts;
  }

  void place(Operation *op) {
    KindCounts counts = during(op);
    for (unsigned k = 0; k < kKinds.size(); ++k)
      peak[k] = std::max(peak[k], counts[k]);
    live = after(op);
    for (Value operand : distinctOperands(op))
      --remaining[operand];
  }

  KindCounts peak{};

private:
  bool needed(Value value) const {
    return kept.contains(value) || remaining.lookup(value) > 0;
  }

  // A DMA immediate takes no register: each launch builds it.
  static void count(KindCounts &counts, Value value, int delta) {
    if (isVirtualDMAImmediate(value))
      return;
    if (std::optional<RegisterKind> kind = kindOf(value.getType()))
      counts[indexOf(*kind)] += delta;
  }

  llvm::DenseMap<Value, unsigned> remaining;
  llvm::DenseSet<Value> kept;
  KindCounts live{};
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
                   ? std::max(cycle + cost.frontend - 1, unitFree[unit(node)])
                   : cycle;
    finished[node] = work + cost.latency;
    unitFree[unit(node)] = work + cost.busy;
    if (cost.queuedDMA)
      transfers.push_back(work + cost.busy);
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
  BlockScheduler(Block &block, const BlockGraph &g, const Liveness &liveness)
      : block(block), g(g), liveness(liveness) {
    unsigned n = g.nodes.size();
    for (Operation *op : g.nodes) {
      Cost cost = costOf(op);
      cost.drains = isImplicitMemory(op);
      costs.push_back(cost);
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

  KindCounts peakPressure(ArrayRef<Operation *> order) const {
    Pressure pressure(block, g.nodes, liveness);
    for (Operation *op : order)
      pressure.place(op);
    return pressure.peak;
  }

  // Of the operations whose dependences are placed, take the one that keeps
  // the registers in use within `capacity` and leaves one of each kind free,
  // or overshoots least: every operation defines at most one register value,
  // so a free register lets the next one run. Then take the one that starts
  // first; then the one that frees the frontend first, so a blocking
  // transfer waits for work that can run beside it; then the one with the
  // longest path to the block's end; then the earliest in source order.
  // A load starts as soon as the frontend is free, so with `lookahead` loaded
  // tiles awaiting their first use, another load waits while anything else
  // is ready: a boundary tile must not drain transfers whose users all come
  // after it, and each holds a staging window until its first use.
  std::vector<unsigned> listSchedule(const KindCounts &capacity,
                                     unsigned lookahead) const {
    Pressure pressure(block, g.nodes, liveness);
    auto over = [&](const KindCounts &counts, int slack) {
      int total = 0;
      for (unsigned k = 0; k < kKinds.size(); ++k)
        total += std::max(0, counts[k] - (capacity[k] - slack));
      return total;
    };
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
        return std::tuple(over(pressure.during(op), 0),
                          over(pressure.after(op), 1), start,
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
  std::vector<int> heights;
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

// Reorder one block when the schedule fits the allocator's capacity and the
// source order does not, or when it is faster without giving up a fit the
// source order has. The schedule is the fastest fitting list schedule over a
// few load lookaheads. A random order is taken as it is.
LogicalResult scheduleBlock(Block &block, unsigned index,
                            const Liveness &liveness,
                            const KindCounts &capacity,
                            std::optional<unsigned> randomSeed) {
  FailureOr<BlockGraph> graph = buildGraph(block);
  if (failed(graph))
    return failure();
  const BlockGraph &g = *graph;
  BlockScheduler scheduler(block, g, liveness);
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

  std::vector<unsigned> source(g.nodes.size());
  std::iota(source.begin(), source.end(), 0u);
  auto fits = [&](ArrayRef<Operation *> candidate) {
    KindCounts peak = scheduler.peakPressure(candidate);
    for (unsigned k = 0; k < kKinds.size(); ++k)
      if (peak[k] > capacity[k])
        return false;
    return true;
  };
  std::vector<unsigned> order;
  SmallVector<Operation *> ops;
  bool scheduleFits = false;
  int span = 0;
  for (unsigned lookahead : {1u, 2u, 4u, kMaxQueuedDMA}) {
    std::vector<unsigned> candidate = scheduler.listSchedule(capacity, lookahead);
    SmallVector<Operation *> candidateOps = opsOf(candidate);
    bool candidateFits = fits(candidateOps);
    int candidateSpan = scheduler.makespan(candidate);
    if (order.empty() || std::pair(!candidateFits, candidateSpan) <
                             std::pair(!scheduleFits, span)) {
      order = std::move(candidate);
      ops = std::move(candidateOps);
      scheduleFits = candidateFits;
      span = candidateSpan;
    }
  }
  bool sourceFits = fits(g.nodes);
  bool faster = span < scheduler.makespan(source);
  if (sourceFits ? scheduleFits && faster : scheduleFits || faster)
    reorder(block, ops);
  return success();
}

// Whether the register allocator succeeds on `function` as it stands; `why`
// receives its first error.
bool allocates(func::FuncOp function, std::string &why) {
  ScopedDiagnosticHandler handler(function.getContext(), [&](Diagnostic &d) {
    if (d.getSeverity() == DiagnosticSeverity::Error && why.empty())
      why = d.str();
    return success();
  });
  VirtualAllocationPlan plan;
  return succeeded(plan.allocate(function)) && succeeded(plan.verify());
}

struct ScheduleAtlasVirtualPass
    : PassWrapper<ScheduleAtlasVirtualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleAtlasVirtualPass)

  ScheduleAtlasVirtualPass() = default;
  ScheduleAtlasVirtualPass(const ScheduleAtlasVirtualPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "schedule-atlas-virtual"; }
  StringRef getDescription() const final {
    return "Reorder each virtual Atlas block onto its functional units before "
           "register allocation";
  }

  Option<int> randomSeed{
      *this, "random-seed",
      llvm::cl::desc("Pick a random legal order instead, for soundness "
                     "testing; negative disables"),
      llvm::cl::init(-1)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyAtlasVirtualModule(module)))
      return signalPassFailure();
    std::optional<unsigned> seed;
    if (randomSeed >= 0)
      seed = static_cast<unsigned>(randomSeed);
    // Each block's free operations in their current order, to put back.
    using BlockOrders =
        SmallVector<std::pair<Block *, SmallVector<Operation *>>>;
    auto ordersOf = [](func::FuncOp function) {
      BlockOrders orders;
      for (Block &block : function.getBody())
        orders.push_back({&block, freeOperations(block)});
      return orders;
    };
    auto restore = [](const BlockOrders &orders) {
      for (const auto &[block, ops] : orders)
        reorder(*block, ops);
    };
    SmallVector<std::pair<func::FuncOp, BlockOrders>> sources;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      sources.push_back({function, ordersOf(function)});
      KindCounts capacity;
      RegisterCapacities capacities = registerCapacities(function);
      for (RegisterKind kind : kKinds)
        capacity[indexOf(kind)] = capacities[indexOf(kind)];
      Liveness liveness(function);
      unsigned index = 0;
      for (Block &block : function.getBody())
        if (failed(scheduleBlock(block, index++, liveness, capacity, seed)))
          return signalPassFailure();
    }
    // Every order the dependence graph allows must verify, so a failure here
    // is a dependence the graph is missing.
    if (failed(verifyAtlasVirtualModule(module))) {
      module.emitError("schedule-atlas-virtual produced an order that does "
                       "not verify");
      return signalPassFailure();
    }
    if (seed)
      return;
    // Scheduling keeps pressure within capacity where it can, but that does
    // not guarantee the allocator's greedy coloring succeeds. Allocation is
    // function-wide, so a function whose schedule does not allocate goes back
    // to its source order when that does.
    for (auto &[function, source] : sources) {
      std::string why, ignored;
      if (allocates(function, why))
        continue;
      BlockOrders scheduled = ordersOf(function);
      restore(source);
      if (!allocates(function, ignored)) {
        // Neither order allocates; the lowering reports why.
        restore(scheduled);
        continue;
      }
      function.emitRemark() << "virtual schedule kept source order for @"
                            << function.getSymName() << ": " << why;
    }
  }
};

} // namespace

void mlir::atlas::registerScheduleAtlasVirtualPass() {
  PassRegistration<ScheduleAtlasVirtualPass>();
}
