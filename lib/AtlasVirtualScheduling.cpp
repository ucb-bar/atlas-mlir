// Pre-allocation scheduling of virtual Atlas operations. Each block is list
// scheduled onto the functional units its operations occupy, keeping every
// SSA and implicit-state dependence. Costs are rough on purpose: registers,
// delays, and delay slots do not exist until the virtual-to-machine lowering,
// and the machine passes after it time the real instructions.

#include "Atlas/AtlasVirtualScheduling.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTiming.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
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

bool isLaunch(Operation *op) {
  return isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op, VirtualDMAStoreFP8Op,
             VirtualDMAStoreBF16Op>(op);
}

bool isStore(Operation *op) {
  return isa<VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op);
}

bool isCompletion(Operation *op) {
  return isa<VirtualDMAAwaitFP8Op, VirtualDMAAwaitBF16Op, VirtualDMAWaitOp>(op);
}

// Boundary tiles and the pack move memory without a transfer handle, so they
// run with no explicit transfer pending.
bool isImplicitMemory(Operation *op) {
  return isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op,
             VirtualPackFP8Op>(op);
}

// The DRAM bytes a transfer moves, [first, last), when its address and size
// are constants.
using DRAMRange = std::pair<uint64_t, uint64_t>;
std::optional<DRAMRange> dramRange(Operation *launch) {
  auto [address, size] =
      llvm::TypeSwitch<Operation *, std::pair<Value, Value>>(launch)
          .Case<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op,
                VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>([](auto op) {
            return std::pair{op.getDramByte(), op.getSizeBytes()};
          })
          .Default([](Operation *) -> std::pair<Value, Value> {
            llvm_unreachable("not a DMA launch");
          });
  APInt first, bytes;
  if (!matchPattern(address, m_ConstantInt(&first)) ||
      !matchPattern(size, m_ConstantInt(&bytes)))
    return std::nullopt;
  return DRAMRange{first.getZExtValue(),
                   first.getZExtValue() + bytes.getZExtValue()};
}

// A range that is not a constant may touch anything.
bool mayOverlap(const std::optional<DRAMRange> &a,
                const std::optional<DRAMRange> &b) {
  return !a || !b || (a->first < b->second && b->first < a->second);
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
  else if (!isLaunch(op) && !isCompletion(op) &&
           !isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op,
                VirtualVPUUnaryOp, VirtualVPUBinaryOp, VirtualScaleConstantOp,
                arith::ConstantOp, arith::AddIOp, arith::CmpIOp>(op))
    e.writes = bit(kNumPseudo) - 1;
  return e;
}

// A resource with a few interchangeable slots: explicit DMA transfers in
// flight, or the weights or accumulators of one MXU unit. Along the source
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
  // The completion of the transfer `launch` starts: its only user.
  auto completionOf = [&](unsigned launch) -> std::optional<unsigned> {
    Value transfer = g.nodes[launch]->getResult(1);
    if (!transfer.hasOneUse())
      return std::nullopt;
    std::optional<unsigned> node = nodeOf(*transfer.user_begin());
    if (!node || !isCompletion(g.nodes[*node]))
      return std::nullopt;
    return node;
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

  // Explicit transfers run at the same time unless one is a store and they
  // may touch the same DRAM bytes; then the later one launches after the
  // earlier completes, as a verified source already orders them. Implicit
  // memory work keeps its place among the transfers, so nothing is pending
  // when it runs.
  SmallVector<unsigned> launches, completions, transfersAndCompletions,
      implicit;
  for (unsigned i = 0; i < n; ++i) {
    Operation *op = g.nodes[i];
    if (isLaunch(op)) {
      std::optional<unsigned> completion = completionOf(i);
      if (!completion)
        return cannotOrder(op, "its transfer is not completed once in it");
      launches.push_back(i);
      completions.push_back(*completion);
    }
    if (isLaunch(op) || isCompletion(op))
      transfersAndCompletions.push_back(i);
    else if (isImplicitMemory(op))
      implicit.push_back(i);
  }
  for (unsigned m : implicit)
    for (unsigned d : transfersAndCompletions)
      edge(std::min(m, d), std::max(m, d));
  std::vector<std::optional<DRAMRange>> ranges;
  for (unsigned launch : launches)
    ranges.push_back(dramRange(g.nodes[launch]));
  for (unsigned a = 0; a < launches.size(); ++a)
    for (unsigned b = a + 1; b < launches.size(); ++b) {
      if (!isStore(g.nodes[launches[a]]) && !isStore(g.nodes[launches[b]]))
        continue;
      if (!mayOverlap(ranges[a], ranges[b]))
        continue;
      // A source that already overlaps them keeps their launch order.
      edge(completions[a] < launches[b] ? completions[a] : launches[a],
           launches[b]);
    }

  // Slot limits. A transfer is last used by its completion, a weight by its
  // users, or by its load if it has none, and an accumulator chain by the
  // readout that ends it.
  SlotPool transfers(kMaxPendingVirtualDMA);
  std::array<SlotPool, 2> weights = {SlotPool(kVirtualMXUSlots),
                                     SlotPool(kVirtualMXUSlots)};
  std::array<SlotPool, 2> accumulators = {SlotPool(kVirtualMXUSlots),
                                          SlotPool(kVirtualMXUSlots)};
  for (unsigned i = 0, t = 0; i < n; ++i) {
    Operation *op = g.nodes[i];
    if (isLaunch(op)) {
      if (!transfers.acquire(i, {completions[t++]}, edge))
        return cannotOrder(op, "it has more DMA transfers pending at once "
                               "than the verifier allows");
    } else if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op)) {
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
enum class Unit { Scalar, Lsu, Vpu, Dma, Mxu0, Mxu1, Weights0, Weights1 };
constexpr unsigned kUnits = static_cast<unsigned>(Unit::Weights1) + 1;

// What an operation costs: it holds one functional unit for `busy` cycles,
// its results are ready after `latency`, and the in-order frontend issues the
// next operation `frontend` cycles after it starts.
struct Cost {
  Unit unit;
  int busy;
  int latency;
  int frontend = 1;
};

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

// Elementwise kinds share one footprint, so a kind the timing model does not
// name costs as a move.
Cost onVpu(StringRef kind) {
  std::string name = kind == "mov" ? "vmov" : ("v" + kind + ".bf16").str();
  MachineTiming t = timingOf(timing::findOp(name) ? name : "vmov");
  return {Unit::Vpu, t.interval, t.latency};
}

// Work that holds the frontend from start to finish.
Cost blocking(Unit unit, int cycles) { return {unit, cycles, cycles, cycles}; }

// An operation costs the machine operations it lowers to, as they issue in
// order on the unit doing its work.
Cost costOf(Operation *op) {
  static const MachineTiming vload = timingOf("vload");
  static const MachineTiming vstore = timingOf("vstore");
  static const int transfer = timing::dmaTransferCycles(kRegisterBytes);
  // Boundary tiles move one register at a time, each transfer waited for
  // before the next instruction issues.
  if (isa<VirtualInputBF16Op, VirtualInputFP8Op>(op))
    return blocking(Unit::Dma, registersOf(isa<VirtualInputFP8Op>(op)) *
                                   (transfer + vload.latency));
  if (isa<VirtualOutputBF16Op>(op))
    return blocking(Unit::Dma,
                    registersOf(false) * (vstore.latency + transfer));
  if (isa<VirtualDMALoadBF16Op, VirtualDMALoadFP8Op>(op)) {
    int registers = registersOf(isa<VirtualDMALoadFP8Op>(op));
    int cycles = timing::dmaTransferCycles(registers * kRegisterBytes);
    return {Unit::Dma, cycles, cycles};
  }
  if (isa<VirtualDMAAwaitBF16Op, VirtualDMAAwaitFP8Op>(op)) {
    // One VLOAD per register; each after the first waits for the load path.
    int behind =
        (registersOf(isa<VirtualDMAAwaitFP8Op>(op)) - 1) * vload.interval;
    return {Unit::Lsu, behind + vload.interval, behind + vload.latency,
            behind + 1};
  }
  if (isa<VirtualDMAStoreBF16Op, VirtualDMAStoreFP8Op>(op)) {
    // VSTOREs into the staging window, then the transfer once they land.
    int registers = registersOf(isa<VirtualDMAStoreFP8Op>(op));
    int staged = (registers - 1) * vstore.interval + vstore.latency;
    int cycles = staged + timing::dmaTransferCycles(registers * kRegisterBytes);
    return {Unit::Dma, cycles, cycles, staged + 1};
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
  if (auto unary = dyn_cast<VirtualVPUUnaryOp>(op))
    return onVpu(unary.getKind());
  if (auto binary = dyn_cast<VirtualVPUBinaryOp>(op))
    return onVpu(binary.getKind());
  if (isa<VirtualPackFP8Op>(op)) {
    // Pack, then relayout the FP8 tile through VMEM one word at a time in a
    // scalar loop.
    int words = registersOf(true) * kRegisterBytes / 4;
    return blocking(Unit::Lsu,
                    timingOf("vpack.bf16.fp8").latency + vstore.latency +
                        words * (timingOf("lw").latency +
                                 timingOf("sw").latency) +
                        vload.latency);
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

  static void count(KindCounts &counts, Value value, int delta) {
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
    int cycle = std::max(frontend, unitFree[unit(node)]);
    for (unsigned pred : g.preds[node])
      cycle = std::max(cycle, finished[pred]);
    return cycle;
  }

  // When the frontend can issue again if `node` starts at `cycle`.
  int frontendAfter(unsigned node, int cycle) const {
    return cycle + costs[node].frontend;
  }

  void place(unsigned node) {
    int cycle = start(node);
    finished[node] = cycle + costs[node].latency;
    unitFree[unit(node)] = cycle + costs[node].busy;
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
  int frontend = 0;
};

class BlockScheduler {
public:
  BlockScheduler(Block &block, const BlockGraph &g, const Liveness &liveness)
      : block(block), g(g), liveness(liveness) {
    unsigned n = g.nodes.size();
    for (Operation *op : g.nodes)
      costs.push_back(costOf(op));
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
  std::vector<unsigned> listSchedule(const KindCounts &capacity) const {
    Pressure pressure(block, g.nodes, liveness);
    auto over = [&](const KindCounts &counts, int slack) {
      int total = 0;
      for (unsigned k = 0; k < kKinds.size(); ++k)
        total += std::max(0, counts[k] - (capacity[k] - slack));
      return total;
    };
    return build([&](const Machine &machine, ArrayRef<unsigned> ready) {
      auto key = [&](unsigned node) {
        Operation *op = g.nodes[node];
        int start = machine.start(node);
        return std::tuple(over(pressure.during(op), 0),
                          over(pressure.after(op), 1), start,
                          machine.frontendAfter(node, start), -heights[node],
                          node);
      };
      unsigned best = ready.front();
      auto bestKey = key(best);
      for (unsigned node : ready.drop_front()) {
        auto nodeKey = key(node);
        if (nodeKey < bestKey) {
          best = node;
          bestKey = nodeKey;
        }
      }
      pressure.place(g.nodes[best]);
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
// source order has. A random order is taken as it is.
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
  std::vector<unsigned> order = scheduler.listSchedule(capacity);
  SmallVector<Operation *> ops = opsOf(order);
  auto fits = [&](ArrayRef<Operation *> candidate) {
    KindCounts peak = scheduler.peakPressure(candidate);
    for (unsigned k = 0; k < kKinds.size(); ++k)
      if (peak[k] > capacity[k])
        return false;
    return true;
  };
  bool sourceFits = fits(g.nodes), scheduleFits = fits(ops);
  bool faster = scheduler.makespan(order) < scheduler.makespan(source);
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
  return succeeded(plan.allocate(function));
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
    struct SourceOrder {
      func::FuncOp function;
      bool allocates;
      SmallVector<std::pair<Block *, SmallVector<Operation *>>> blocks;
    };
    SmallVector<SourceOrder> sources;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      std::string ignored;
      SourceOrder &source = sources.emplace_back();
      source.function = function;
      source.allocates = !seed && allocates(function, ignored);
      for (Block &block : function.getBody())
        source.blocks.push_back({&block, freeOperations(block)});

      KindCounts capacity;
      for (RegisterKind kind : kKinds)
        capacity[indexOf(kind)] = registerCapacity(function, kind);
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
    // Scheduling keeps pressure within capacity where it can, but that does
    // not guarantee the allocator's greedy coloring succeeds. Allocation is
    // function-wide, so a function that allocated before and no longer does
    // keeps its source order.
    for (SourceOrder &source : sources) {
      std::string why;
      if (!source.allocates || allocates(source.function, why))
        continue;
      for (auto &[block, ops] : source.blocks)
        reorder(*block, ops);
      source.function.emitRemark()
          << "virtual schedule kept source order for @"
          << source.function.getSymName() << ": " << why;
    }
  }
};

} // namespace

void mlir::atlas::registerScheduleAtlasVirtualPass() {
  PassRegistration<ScheduleAtlasVirtualPass>();
}
