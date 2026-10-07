// Pre-allocation list scheduling of virtual Atlas operations. The design is
// docs/virtual-scheduler-plan.md; section numbers below refer to it.

#include "Atlas/AtlasVirtualScheduling.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasTiming.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasVirtualScheduleTrace.h"
#include "Atlas/AtlasVirtualToMachine.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <tuple>

using namespace mlir;
using namespace mlir::atlas;

namespace {

// Register kinds as the allocator colors them (§7.1). Arrays of counts are
// indexed by RegisterKind.
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

const char *kindName(RegisterKind kind) {
  switch (kind) {
  case RegisterKind::BF16:
    return "BF16";
  case RegisterKind::FP8:
    return "FP8";
  case RegisterKind::Scalar:
    return "scalar";
  }
  llvm_unreachable("unknown register kind");
}

// The allocator's capacity for each kind in this function.
KindCounts capsFor(bool mixedFp8, bool hasPack) {
  KindCounts caps;
  for (RegisterKind kind : kKinds)
    caps[indexOf(kind)] = registerBudget(kind, mixedFp8, hasPack).count;
  return caps;
}

// Ordering obligations the IR does not express as SSA, as pseudo-registers
// (§5.2, E2).
enum Pseudo : unsigned {
  kDma,     // the single explicit-DMA pending slot
  kWeight0, // the current weight of MXU0 and MXU1
  kWeight1,
  kAcc0, // the accumulator occupancy of MXU0 and MXU1
  kAcc1,
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

// An explicit table: `Pure` is not trusted, because legacy virtual_mxu_matmul
// and virtual_pack_fp8 are declared Pure yet have ordering obligations.
Effects effectsOf(Operation *op) {
  Effects e;
  if (isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op>(op)) {
    e.reads = bit(kDma);
  } else if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op,
                 VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op,
                 VirtualDMAAwaitFP8Op, VirtualDMAAwaitBF16Op,
                 VirtualDMAWaitOp>(op)) {
    e.writes = bit(kDma);
  } else if (isa<VirtualPackFP8Op>(op)) {
    e.reads = bit(kDma);
    e.writes = bit(kPack);
  } else if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op)) {
    e.writes = bit(kWeight0 + unitOf(load.getWeight().getType()));
  } else if (auto reset = dyn_cast<VirtualMXUResetOp>(op)) {
    unsigned unit = unitOf(reset.getAcc().getType());
    e.reads = bit(kWeight0 + unit);
    e.writes = bit(kAcc0 + unit);
  } else if (auto acc = dyn_cast<VirtualMXUAccumulateOp>(op)) {
    unsigned unit = unitOf(acc.getAcc().getType());
    e.reads = bit(kWeight0 + unit);
    e.writes = bit(kAcc0 + unit);
  } else if (isa<VirtualMXULoadAccFP8Op, VirtualMXULoadAccBF16Op>(op)) {
    e.writes = bit(kAcc0 + unitOf(op->getResult(1).getType()));
  } else if (isa<VirtualMXUReadoutBF16Op, VirtualMXUReadoutFP8Op>(op)) {
    e.writes = bit(kAcc0 + unitOf(op->getOperand(1).getType()));
  } else if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(op)) {
    unsigned unit = matmul.getUnit();
    e.writes = bit(kWeight0 + unit) | bit(kAcc0 + unit);
  } else if (!isa<VirtualVPUUnaryOp, VirtualVPUBinaryOp, VirtualScaleConstantOp,
                  arith::ConstantOp, arith::AddIOp, arith::CmpIOp>(op)) {
    e.writes = bit(kNumPseudo) - 1;
  }
  e.reads |= bit(kBarrier);
  return e;
}

// The dependence graph of one block's free operations (§5). Every edge runs
// from an earlier to a later source position.
struct BlockGraph {
  Block *block = nullptr;
  SmallVector<Operation *> nodes;
  std::vector<SmallVector<unsigned, 4>> preds, succs;
};

BlockGraph buildGraph(Block &block) {
  BlockGraph g;
  g.block = &block;
  for (Operation &op : block)
    if (!isa<VirtualStartOp>(op) && !op.hasTrait<OpTrait::IsTerminator>())
      g.nodes.push_back(&op);
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
  // E1: data, except the state token, which records source order only.
  for (unsigned i = 0; i < n; ++i)
    for (Value operand : g.nodes[i]->getOperands()) {
      if (isa<VirtualStateType>(operand.getType()))
        continue;
      auto found = index.find(operand.getDefiningOp());
      if (found != index.end())
        edge(found->second, i);
    }
  // E2: read-after-write, write-after-read, and write-after-write on each
  // pseudo-register.
  std::vector<Effects> effects;
  for (Operation *op : g.nodes)
    effects.push_back(effectsOf(op));
  for (unsigned r = 0; r < kNumPseudo; ++r) {
    std::optional<unsigned> writer;
    SmallVector<unsigned> readers;
    for (unsigned i = 0; i < n; ++i) {
      if (effects[i].reads & bit(r)) {
        if (writer)
          edge(*writer, i);
        readers.push_back(i);
      }
      if (effects[i].writes & bit(r)) {
        if (writer)
          edge(*writer, i);
        for (unsigned reader : readers)
          edge(reader, i);
        readers.clear();
        writer = i;
      }
    }
  }
  return g;
}

// Live values per register kind, counted as the allocator's interference
// graph counts them (§7.2, §7.3).
class Pressure {
public:
  Pressure(const BlockGraph &g, const Liveness &liveness) {
    for (Operation *op : g.nodes) {
      SmallVector<Value, 4> uses;
      for (Value operand : op->getOperands())
        if (kindOf(operand.getType()) && !llvm::is_contained(uses, operand))
          uses.push_back(operand);
      for (Value use : uses)
        ++remaining[use];
      operands.push_back(uses);
      SmallVector<Value, 2> defs;
      for (Value result : op->getResults())
        if (kindOf(result.getType()))
          defs.push_back(result);
      results.push_back(defs);
    }
    // Live at the end: live-out values and every terminator operand (§6.2).
    Block *block = g.block;
    llvm::DenseSet<Value> exitSet(liveness.getLiveOut(block).begin(),
                                  liveness.getLiveOut(block).end());
    for (Value operand : block->getTerminator()->getOperands())
      exitSet.insert(operand);
    for (Value value : exitSet)
      if (auto kind = kindOf(value.getType())) {
        pinned.insert(value);
        ++exitCounts[indexOf(*kind)];
      }
    // Live at the start: live-in values and every block argument, used or not.
    llvm::DenseSet<Value> entrySet(liveness.getLiveIn(block).begin(),
                                   liveness.getLiveIn(block).end());
    for (BlockArgument arg : block->getArguments())
      entrySet.insert(arg);
    for (Value value : entrySet)
      if (auto kind = kindOf(value.getType())) {
        ++entryCounts[indexOf(*kind)];
        if (remaining.lookup(value) > 0 || pinned.contains(value))
          live[indexOf(*kind)].insert(value);
      }
    for (unsigned k = 0; k < kKinds.size(); ++k)
      peak[k] = std::max(entryCounts[k], exitCounts[k]);
  }

  // Live values while `node` executes: its operands interfere with its
  // results, so both count (§7.2).
  KindCounts at(unsigned node) const {
    KindCounts counts = current();
    for (Value result : results[node]) {
      unsigned k = indexOf(*kindOf(result.getType()));
      if (!live[k].contains(result))
        ++counts[k];
    }
    return counts;
  }

  bool fits(unsigned node, const KindCounts &caps) const {
    KindCounts counts = at(node);
    for (unsigned k = 0; k < kKinds.size(); ++k)
      if (counts[k] > caps[k])
        return false;
    return true;
  }

  // The net change in live values if `node` were placed now.
  int delta(unsigned node) const {
    int change = 0;
    for (Value result : results[node])
      if (remaining.lookup(result) > 0 || pinned.contains(result))
        ++change;
    for (Value operand : operands[node])
      if (remaining.lookup(operand) == 1 && !pinned.contains(operand))
        --change;
    return change;
  }

  void place(unsigned node) {
    KindCounts counts = at(node);
    for (unsigned k = 0; k < kKinds.size(); ++k)
      peak[k] = std::max(peak[k], counts[k]);
    for (Value operand : operands[node])
      if (--remaining[operand] == 0 && !pinned.contains(operand))
        live[indexOf(*kindOf(operand.getType()))].erase(operand);
    for (Value result : results[node])
      if (remaining.lookup(result) > 0 || pinned.contains(result))
        live[indexOf(*kindOf(result.getType()))].insert(result);
  }

  KindCounts current() const {
    KindCounts counts;
    for (unsigned k = 0; k < kKinds.size(); ++k)
      counts[k] = live[k].size();
    return counts;
  }

  KindCounts entryCounts{}, exitCounts{}, peak{};

private:
  std::array<llvm::DenseSet<Value>, kKinds.size()> live;
  llvm::DenseMap<Value, int> remaining;
  llvm::DenseSet<Value> pinned;
  std::vector<SmallVector<Value, 4>> operands;
  std::vector<SmallVector<Value, 2>> results;
};

// Registers for values that have none before allocation (§8.2). A value
// takes the next location of its kind's budget, in rotation, when the first
// operation touching it is placed, so values defined close together never
// share a register and the timing model sees data dependences exactly.
// Assignment depends only on placement order, so placing the same order
// again reproduces the same costs.
class PlaceholderRegisters {
public:
  PlaceholderRegisters(bool mixedFp8, bool hasPack) {
    for (RegisterKind kind : kKinds)
      budgets[indexOf(kind)] = registerBudget(kind, mixedFp8, hasPack);
  }

  // Give each of `values` without a register one: into this state, or, for
  // a probe, into `pending` only.
  void assign(ArrayRef<Value> values, llvm::DenseMap<Value, unsigned> *pending) {
    std::array<unsigned, kKinds.size()> probeNext = next;
    for (Value value : values) {
      std::optional<RegisterKind> kind = kindOf(value.getType());
      if (!kind || regs.count(value) || (pending && pending->count(value)))
        continue;
      unsigned k = indexOf(*kind);
      unsigned &n = pending ? probeNext[k] : next[k];
      unsigned reg = budgets[k].reg(n++ % budgets[k].count);
      (pending ? *pending : regs)[value] = reg;
    }
  }

  std::optional<unsigned> find(Value value) const {
    auto found = regs.find(value);
    if (found == regs.end())
      return std::nullopt;
    return found->second;
  }

private:
  std::array<RegisterBudget, kKinds.size()> budgets;
  std::array<unsigned, kKinds.size()> next{};
  llvm::DenseMap<Value, unsigned> regs;
};

// The lowering's view of placeholder registers, with the MXU, DMA, and fixed
// placements of the real allocation policy.
class PlaceholderPlacement : public VirtualPlacement {
public:
  PlaceholderPlacement(PlaceholderRegisters &registers,
                       const VirtualAllocationPlan &resources,
                       ArrayRef<Value> values, bool commit)
      : registers(registers), resources(resources) {
    registers.assign(values, commit ? nullptr : &pending);
  }

  unsigned tile(Value value) const override { return reg(value); }
  unsigned fp8(Value value) const override { return reg(value); }
  unsigned scalar(Value value) const override { return reg(value); }
  MXUPlacement mxu(Value value) const override { return resources.mxu(value); }
  const DMATransferPlacement &dma(Value value) const override {
    return resources.dma(value);
  }
  const FixedResourcePlacement &fixed() const override {
    return resources.fixed();
  }

private:
  unsigned reg(Value value) const {
    auto found = pending.find(value);
    if (found != pending.end())
      return found->second;
    if (std::optional<unsigned> reg = registers.find(value))
      return *reg;
    llvm_unreachable("the lowering reads only the values it was given");
  }

  PlaceholderRegisters &registers;
  const VirtualAllocationPlan &resources;
  llvm::DenseMap<Value, unsigned> pending;
};

// The timing model's instructions for one virtual operation, in the order
// the lowering emits them. A redirect's target is an instruction index.
struct MachineTemplate {
  std::vector<timing::Instr> instrs;
  std::vector<std::optional<size_t>> targets;
};

SmallVector<Value> valuesOf(Operation *op) {
  SmallVector<Value> values(op->getOperands());
  llvm::append_range(values, op->getResults());
  return values;
}

// Times virtual operations with the instructions the lowering emits for
// them, without its fixed diagnostic delays: the stream phase 2 hands to
// --insert-atlas-delays (§8.1).
class LoweringModel : public MachineStepSink {
public:
  // Fails, with the lowering's first diagnostic in `why`, for a function the
  // lowering cannot handle: no ABI, or an operation without a lowering.
  static std::unique_ptr<LoweringModel> create(func::FuncOp function,
                                               std::string &why);

  bool mixedFp8() const { return resources.fp8Present(); }
  bool hasPack() const { return resources.packPresent(); }
  // Registers the prologue sets and no other code writes, known everywhere.
  const timing::RegValues &entryRegs() const { return entry; }

  std::optional<MachineTemplate> build(Operation *op,
                                       PlaceholderRegisters &registers,
                                       bool commit) {
    PlaceholderPlacement placement(registers, resources, valuesOf(op), commit);
    resetSteps();
    if (failed(lowerVirtualOperation(*op, placement, abi, options, *this)))
      return std::nullopt;
    return convert();
  }

  void add(MachineStep step) override { steps.push_back(std::move(step)); }
  unsigned newLabel() override { return nextLabel++; }
  void mark(unsigned label) override { labels[label] = steps.size(); }
  unsigned labelFor(Block *) override {
    llvm_unreachable("terminators are never scheduled");
  }

private:
  explicit LoweringModel(func::FuncOp function)
      : scratch(ModuleOp::create(function.getLoc())) {
    OpBuilder builder = OpBuilder::atBlockEnd(scratch->getBody());
    OperationState start(function.getLoc(), "atlas.start");
    start.addTypes(StateType::get(function.getContext()));
    state = builder.create(start)->getResult(0);
  }

  void resetSteps() {
    steps.clear();
    labels.clear();
    nextLabel = 0;
  }

  // Convert the steps with the converter the machine passes use, through
  // machine operations built in a scratch module.
  std::optional<MachineTemplate> convert() {
    MachineTemplate t;
    OpBuilder builder = OpBuilder::atBlockEnd(scratch->getBody());
    for (auto [pc, step] : llvm::enumerate(steps)) {
      OperationState machine(step.loc, step.name);
      machine.addOperands(state);
      machine.addTypes(StateType::get(builder.getContext()));
      machine.addAttributes(step.attrs);
      std::optional<size_t> target;
      if (step.targetLabel) {
        target = labels.lookup(*step.targetLabel);
        int64_t offset = 2 * (static_cast<int64_t>(*target) -
                              static_cast<int64_t>(pc));
        machine.addAttribute(step.name == "atlas.branch" ? "offset_bytes"
                                                         : "offset",
                             builder.getI32IntegerAttr(offset));
      }
      Operation *op = builder.create(machine);
      FailureOr<timing::Instr> in = toTimingInstr(op);
      op->erase();
      if (failed(in))
        return std::nullopt;
      t.instrs.push_back(*in);
      t.targets.push_back(target);
    }
    return t;
  }

  VirtualAllocationPlan resources;
  VirtualLoweringABI abi;
  VirtualLoweringOptions options{/*diagnosticDelays=*/false};
  OwningOpRef<ModuleOp> scratch;
  Value state;
  std::vector<MachineStep> steps;
  llvm::DenseMap<unsigned, size_t> labels;
  unsigned nextLabel = 0;
  timing::RegValues entry = timing::unknownRegs();
};

// Run `fn` with diagnostics captured instead of printed; `firstError`
// receives the first error.
template <typename Fn>
auto quietly(MLIRContext *context, std::string &firstError, Fn &&fn) {
  ScopedDiagnosticHandler handler(context, [&](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error && firstError.empty())
      firstError = diag.str();
    return success();
  });
  return fn();
}

std::unique_ptr<LoweringModel> LoweringModel::create(func::FuncOp function,
                                                     std::string &why) {
  return quietly(function.getContext(), why,
                 [&]() -> std::unique_ptr<LoweringModel> {
    std::unique_ptr<LoweringModel> model(new LoweringModel(function));
    if (failed(model->resources.placeResources(function)))
      return nullptr;
    FailureOr<VirtualLoweringABI> abi =
        readVirtualLoweringABI(function, model->resources.fixed());
    if (failed(abi))
      return nullptr;
    model->abi = *abi;

    PlaceholderRegisters registers(model->mixedFp8(), model->hasPack());
    SmallVector<Value> arguments(function.getArguments());
    PlaceholderPlacement placement(registers, model->resources, arguments,
                                   /*commit=*/true);
    model->resetSteps();
    lowerVirtualPrologue(function, placement, model->abi, model->options,
                         *model);
    std::optional<MachineTemplate> prologue = model->convert();
    if (!prologue)
      return nullptr;
    for (const timing::Instr &in : prologue->instrs)
      timing::applyScalar(in, model->entry);

    // Every operation must lower before any order is timed.
    for (Block &block : function.getBody())
      for (Operation &op : block)
        if (!isa<VirtualStartOp>(op) &&
            !op.hasTrait<OpTrait::IsTerminator>() &&
            !model->build(&op, registers, /*commit=*/false))
          return nullptr;
    return model;
  });
}

const char *engineName(timing::Engine engine) {
  switch (engine) {
  case timing::Engine::Scalar:
    return "Scalar";
  case timing::Engine::Lsu:
    return "LSU";
  case timing::Engine::Mxu0:
    return "MXU0";
  case timing::Engine::Mxu1:
    return "MXU1";
  case timing::Engine::Vpu:
    return "VPU";
  case timing::Engine::Xlu:
    return "XLU";
  case timing::Engine::Dma:
    return "DMA";
  }
  llvm_unreachable("unknown engine");
}

// Where a template landed: its first issue, when the frontend can issue
// again, and when its own work ends.
struct Placement {
  int first;
  int frontendFree;
  int done;
};

// The block's modeled run: templates issue in order through the in-order
// issue model insert-atlas-delays uses, with DMA waits released when their
// modeled transfers end (§8.3).
class Timeline {
public:
  explicit Timeline(const timing::RegValues &entry)
      : issue(entry, timing::WaitRelease::Modeled) {}

  std::optional<Placement>
  place(const MachineTemplate &t, unsigned node,
        std::vector<ScheduleTimeline::Instruction> *trace = nullptr);

  // Where `t` would land, leaving this timeline unchanged.
  std::optional<Placement> project(const MachineTemplate &t) const {
    Timeline copy = *this;
    return copy.place(t, 0);
  }

  int finish() const { return std::max(issue.nextFree(), issue.drained()); }

private:
  std::optional<std::pair<int, int>>
  issueNext(const timing::Instr &in, int notBefore, unsigned node,
            std::vector<ScheduleTimeline::Instruction> *trace);

  timing::InOrderIssue issue;
};

std::optional<std::pair<int, int>>
Timeline::issueNext(const timing::Instr &in, int notBefore, unsigned node,
                    std::vector<ScheduleTimeline::Instruction> *trace) {
  timing::Footprint f = timing::footprintOf(in, issue.regs());
  int from = std::max(issue.earliest(in, f, issue.nextFree()), notBefore);
  std::string why;
  std::optional<int> cycle = timing::firstFit(
      from, [&](int c) { return issue.table().conflict(in, f, c); }, why);
  if (!cycle)
    return std::nullopt;
  issue.issue(in, f, *cycle);
  int end = f.dmaCycles > 0 ? issue.dmaRelease(in.op->channel)
                            : *cycle + f.doneAge + 1;
  if (trace)
    trace->push_back(
        {node, in.op->name, engineName(in.op->engine), *cycle, end});
  return std::pair{*cycle, end};
}

std::optional<Placement>
Timeline::place(const MachineTemplate &t, unsigned node,
                std::vector<ScheduleTimeline::Instruction> *trace) {
  // The lowering's loops have trip counts the model evaluates; this bounds a
  // loop whose count it cannot.
  constexpr unsigned kMaxSteps = 1u << 16;
  std::optional<int> first;
  int done = issue.nextFree();
  unsigned steps = 0;
  for (size_t pc = 0; pc < t.instrs.size();) {
    if (++steps > kMaxSteps)
      return std::nullopt;
    const timing::Instr &in = t.instrs[pc];
    bool redirect = timing::isControlFlow(*in.op);
    std::optional<bool> taken =
        in.op->opClass == timing::OpClass::Jump
            ? std::optional<bool>(true)
            : timing::branchTaken(in, issue.regs());
    // A branch whose outcome depends on runtime data cannot be timed.
    if (redirect && !taken)
      return std::nullopt;
    // As insert-atlas-delays places a redirect: two cycles before the work
    // in flight drains, since its successors start drained, with its delay
    // slot next.
    auto issued =
        issueNext(in, redirect ? issue.drained() - 2 : INT_MIN, node, trace);
    if (!issued)
      return std::nullopt;
    first = first.value_or(issued->first);
    done = std::max(done, issued->second);
    if (!redirect) {
      ++pc;
      continue;
    }
    if (pc + 1 >= t.instrs.size())
      return std::nullopt;
    auto slot = issueNext(t.instrs[pc + 1], issued->first + 1, node, trace);
    if (!slot)
      return std::nullopt;
    done = std::max(done, slot->second);
    pc = *taken ? *t.targets[pc] : pc + 2;
  }
  issue.retire();
  return Placement{first.value_or(issue.nextFree()), issue.nextFree(), done};
}

enum class Priority {
  // Minimize the bound on the block's finish (§9.2).
  Latency,
  // Minimize live values, for blocks the latency order cannot fit.
  Pressure,
};

// Orders and evaluates one block.
class BlockScheduler {
public:
  BlockScheduler(const BlockGraph &g, const Liveness &liveness,
                 LoweringModel *model, KindCounts caps)
      : g(g), liveness(liveness), model(model), caps(caps) {}

  // Block-boundary pressure no order can change (§6.2).
  std::optional<std::string> boundaryViolation() const {
    Pressure pressure(g, liveness);
    for (RegisterKind kind : kKinds) {
      unsigned k = indexOf(kind);
      for (auto [where, count] : {std::pair{"entry", pressure.entryCounts[k]},
                                  std::pair{"exit", pressure.exitCounts[k]}})
        if (count > caps[k])
          return std::string(where) + " " + kindName(kind) + " pressure " +
                 std::to_string(count) + " exceeds cap " +
                 std::to_string(caps[k]);
    }
    return std::nullopt;
  }

  std::optional<ScheduleTimeline> evaluate(ArrayRef<unsigned> order);
  std::optional<std::vector<unsigned>> listSchedule(Priority priority,
                                                    std::string &why);
  std::vector<unsigned> randomOrder(unsigned seed) const;

private:
  bool computeHeights();

  const BlockGraph &g;
  const Liveness &liveness;
  LoweringModel *model;
  KindCounts caps;
  std::vector<int> heights;
};

std::optional<ScheduleTimeline>
BlockScheduler::evaluate(ArrayRef<unsigned> order) {
  ScheduleTimeline e;
  PlaceholderRegisters registers(model->mixedFp8(), model->hasPack());
  Timeline timeline(model->entryRegs());
  Pressure pressure(g, liveness);
  for (unsigned node : order) {
    std::optional<MachineTemplate> t =
        model->build(g.nodes[node], registers, /*commit=*/true);
    std::optional<Placement> placed =
        t ? timeline.place(*t, node, &e.instructions) : std::nullopt;
    if (!placed)
      return std::nullopt;
    e.spans.push_back({node, placed->first, placed->done});
    pressure.place(node);
    e.live.push_back(pressure.current());
  }
  e.order.assign(order.begin(), order.end());
  e.cycles = timeline.finish();
  e.peak = pressure.peak;
  return e;
}

// The longest modeled path from each node to the end of the block, counting
// its own work. Edge latencies come from placing the two templates alone, so
// they do not depend on the order being built.
bool BlockScheduler::computeHeights() {
  unsigned n = g.nodes.size();
  heights.assign(n, 0);
  for (unsigned i = n; i-- > 0;) {
    PlaceholderRegisters alone(model->mixedFp8(), model->hasPack());
    Timeline timeline(model->entryRegs());
    std::optional<MachineTemplate> t =
        model->build(g.nodes[i], alone, /*commit=*/true);
    std::optional<Placement> own = t ? timeline.place(*t, i) : std::nullopt;
    if (!own)
      return false;
    int height = own->done - own->first;
    for (unsigned s : g.succs[i]) {
      PlaceholderRegisters pair(model->mixedFp8(), model->hasPack());
      Timeline both(model->entryRegs());
      std::optional<MachineTemplate> from =
          model->build(g.nodes[i], pair, /*commit=*/true);
      std::optional<Placement> start =
          from ? both.place(*from, i) : std::nullopt;
      std::optional<MachineTemplate> to =
          start ? model->build(g.nodes[s], pair, /*commit=*/true)
                : std::nullopt;
      std::optional<Placement> next = to ? both.project(*to) : std::nullopt;
      if (!next)
        return false;
      height = std::max(height, next->first - start->first + heights[s]);
    }
    heights[i] = height;
  }
  return true;
}

std::optional<std::vector<unsigned>>
BlockScheduler::listSchedule(Priority priority, std::string &why) {
  if (heights.empty() && !computeHeights()) {
    why = "the timing model cannot time this block";
    return std::nullopt;
  }
  unsigned n = g.nodes.size();
  std::vector<unsigned> waiting(n);
  for (unsigned i = 0; i < n; ++i)
    waiting[i] = g.preds[i].size();
  std::vector<bool> placed(n, false);
  std::vector<unsigned> order;
  PlaceholderRegisters registers(model->mixedFp8(), model->hasPack());
  Timeline timeline(model->entryRegs());
  Pressure pressure(g, liveness);
  while (order.size() < n) {
    SmallVector<unsigned> fits;
    for (unsigned i = 0; i < n; ++i)
      if (!placed[i] && waiting[i] == 0 && pressure.fits(i, caps))
        fits.push_back(i);
    if (fits.empty()) {
      why = "no order fits the register caps";
      return std::nullopt;
    }

    // Each candidate is ranked by a key; ties go to the lower source index.
    using Key = std::tuple<int, int, int>;
    std::optional<std::pair<Key, unsigned>> best;
    auto consider = [&](Key key, unsigned i) {
      if (!best || key < best->first)
        best = {key, i};
    };
    if (priority == Priority::Pressure) {
      for (unsigned i : fits)
        consider({pressure.delta(i), -heights[i], 0}, i);
    } else {
      // Every unplaced node issues after the frontend frees up, so placing
      // `c` bounds the block's finish below by max(start(c) + height(c),
      // frontendFree(c) + the tallest other height). Prefer the smallest
      // bound: a template that holds the frontend, like a DMA wait, delays
      // everything else, and a critical template should start early.
      std::array<std::pair<int, unsigned>, 2> tallest = {
          std::pair{INT_MIN, n}, std::pair{INT_MIN, n}};
      for (unsigned i = 0; i < n; ++i)
        if (!placed[i]) {
          std::pair<int, unsigned> h{heights[i], i};
          if (h.first > tallest[0].first)
            tallest = {h, tallest[0]};
          else if (h.first > tallest[1].first)
            tallest[1] = h;
        }
      for (unsigned c : fits) {
        int other = tallest[0].second == c ? tallest[1].first : tallest[0].first;
        std::optional<MachineTemplate> t =
            model->build(g.nodes[c], registers, /*commit=*/false);
        std::optional<Placement> p = t ? timeline.project(*t) : std::nullopt;
        if (!p) {
          why = "the timing model cannot time this block";
          return std::nullopt;
        }
        int bound = std::max(p->first + heights[c],
                             other == INT_MIN ? INT_MIN
                                              : p->frontendFree + other);
        consider({bound, -heights[c], pressure.delta(c)}, c);
      }
    }

    unsigned pick = best->second;
    std::optional<MachineTemplate> t =
        model->build(g.nodes[pick], registers, /*commit=*/true);
    if (!t || !timeline.place(*t, pick)) {
      why = "the timing model cannot time this block";
      return std::nullopt;
    }
    pressure.place(pick);
    placed[pick] = true;
    order.push_back(pick);
    for (unsigned s : g.succs[pick])
      --waiting[s];
  }
  return order;
}

// A uniformly random legal order, for soundness testing (§11.3).
std::vector<unsigned> BlockScheduler::randomOrder(unsigned seed) const {
  unsigned n = g.nodes.size();
  std::vector<unsigned> waiting(n);
  for (unsigned i = 0; i < n; ++i)
    waiting[i] = g.preds[i].size();
  std::vector<bool> placed(n, false);
  std::vector<unsigned> order;
  std::mt19937 rng(seed);
  while (order.size() < n) {
    SmallVector<unsigned> ready;
    for (unsigned i = 0; i < n; ++i)
      if (!placed[i] && waiting[i] == 0)
        ready.push_back(i);
    unsigned pick =
        ready[std::uniform_int_distribution<size_t>(0, ready.size() - 1)(rng)];
    placed[pick] = true;
    order.push_back(pick);
    for (unsigned s : g.succs[pick])
      --waiting[s];
  }
  return order;
}

// Move `ops` before the terminator in order, and rethread the state chain
// through the new order (§5.4).
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

bool lowers(ModuleOp module, std::string &why) {
  OwningOpRef<ModuleOp> clone(module.clone());
  return quietly(module.getContext(), why,
                 [&] { return succeeded(lowerAtlasVirtualModule(*clone)); });
}

struct ScheduleOptions {
  bool report = false;
  bool strict = false;
  std::optional<unsigned> randomSeed;
};

// Schedules one function as a transaction (§10): every block is planned on
// the source IR, the plans are applied, and the result must verify and,
// when the source lowers, lower too.
class FunctionScheduler {
public:
  FunctionScheduler(func::FuncOp function, ScheduleOptions options)
      : function(function), options(options) {}

  LogicalResult run(ModuleOp module, bool trialLowering,
                    FunctionScheduleRecord &record);

private:
  struct BlockPlan {
    BlockGraph graph;
    std::vector<Operation *> scheduledOps;
    int gain = 0;
    BlockScheduleRecord record;
  };

  void plan(BlockPlan &p, const Liveness &liveness, LoweringModel *model,
            const KindCounts &caps);
  void chooseOrder(BlockScheduler &scheduler, const KindCounts &caps,
                   std::vector<unsigned> &order, BlockScheduleRecord &r);
  void restore(BlockPlan &p, std::string reason) {
    reorder(*p.graph.block, p.graph.nodes);
    p.scheduledOps.assign(p.graph.nodes.begin(), p.graph.nodes.end());
    p.record.keptReason = std::move(reason);
    p.record.strategy = "source";
    p.record.scheduled = p.record.source;
    p.gain = 0;
  }
  void remark(BlockPlan &p);

  func::FuncOp function;
  ScheduleOptions options;
  std::vector<BlockPlan> plans;
};

void FunctionScheduler::plan(BlockPlan &p, const Liveness &liveness,
                             LoweringModel *model, const KindCounts &caps) {
  BlockGraph &g = p.graph;
  BlockScheduleRecord &r = p.record;
  for (Operation *op : g.nodes)
    r.labels.push_back(describeVirtualOperation(op));
  for (const auto &preds : g.preds)
    r.predecessors.emplace_back(preds.begin(), preds.end());
  r.caps = caps;

  std::vector<unsigned> order(g.nodes.size());
  std::iota(order.begin(), order.end(), 0u);
  if (!g.nodes.empty()) {
    BlockScheduler scheduler(g, liveness, model, caps);
    if (options.randomSeed) {
      order = scheduler.randomOrder(*options.randomSeed + r.index);
      r.strategy = "random";
    } else {
      chooseOrder(scheduler, caps, order, r);
      if (r.source && r.scheduled)
        p.gain = r.source->cycles - r.scheduled->cycles;
    }
  }
  p.scheduledOps.clear();
  for (unsigned node : order)
    p.scheduledOps.push_back(g.nodes[node]);
}

// The list schedules are heuristics, but the model times whole orders
// exactly, so keep the fastest candidate. A schedule never models slower than
// a source order that fits the caps, and ties keep the source.
void FunctionScheduler::chooseOrder(BlockScheduler &scheduler,
                                    const KindCounts &caps,
                                    std::vector<unsigned> &order,
                                    BlockScheduleRecord &r) {
  if (std::optional<std::string> why = scheduler.boundaryViolation()) {
    r.keptReason = *why;
    return;
  }
  r.source = scheduler.evaluate(order);
  if (!r.source) {
    r.keptReason = "the timing model cannot time this block";
    return;
  }
  r.scheduled = r.source;
  bool sourceFits = true;
  for (unsigned k = 0; k < kKinds.size(); ++k)
    sourceFits &= r.source->peak[k] <= caps[k];
  std::optional<int> best;
  if (sourceFits)
    best = r.source->cycles;
  std::string why;
  for (auto [priority, name] : {std::pair{Priority::Latency, "latency"},
                                std::pair{Priority::Pressure, "pressure"}}) {
    std::optional<std::vector<unsigned>> candidate =
        scheduler.listSchedule(priority, why);
    if (!candidate)
      continue;
    std::optional<ScheduleTimeline> timeline = scheduler.evaluate(*candidate);
    if (timeline && (!best || timeline->cycles < *best)) {
      best = timeline->cycles;
      order = *candidate;
      r.scheduled = std::move(timeline);
      r.strategy = name;
    }
  }
  if (!best)
    r.keptReason = why;
}

void FunctionScheduler::remark(BlockPlan &p) {
  BlockScheduleRecord &r = p.record;
  Operation *where = p.graph.block->getTerminator();
  std::string block = "block " + std::to_string(r.index);
  if (r.keptReason)
    where->emitRemark("virtual schedule kept source order for " + block +
                      ": " + *r.keptReason);
  if (!options.report || !r.source || !r.scheduled)
    return;
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "virtual schedule " << block << ": modeled " << r.source->cycles
     << " cycles in source order, " << r.scheduled->cycles
     << " scheduled; peak pressure";
  for (RegisterKind kind : kKinds) {
    unsigned k = indexOf(kind);
    os << (k ? "," : "") << " " << kindName(kind) << " "
       << r.scheduled->peak[k] << "/" << r.caps[k] << " (source "
       << r.source->peak[k] << ")";
  }
  where->emitRemark(text);
}

LogicalResult FunctionScheduler::run(ModuleOp module, bool trialLowering,
                                     FunctionScheduleRecord &record) {
  record.name = function.getSymName().str();
  std::unique_ptr<LoweringModel> model;
  KindCounts caps{};
  if (!options.randomSeed) {
    std::string why;
    model = LoweringModel::create(function, why);
    if (!model) {
      record.keptReason = "the lowering cannot handle it: " + why;
      function.emitRemark("virtual schedule kept source order for @")
          << record.name << ": " << *record.keptReason;
      return success();
    }
    caps = capsFor(model->mixedFp8(), model->hasPack());
  }

  {
    Liveness liveness(function);
    unsigned index = 0;
    for (Block &block : function.getBody()) {
      BlockPlan &p = plans.emplace_back();
      p.graph = buildGraph(block);
      p.record.index = index++;
      plan(p, liveness, model.get(), caps);
    }
  }
  bool changed = false;
  for (BlockPlan &p : plans)
    if (!llvm::equal(p.scheduledOps, p.graph.nodes)) {
      reorder(*p.graph.block, p.scheduledOps);
      changed = true;
    }

  if (changed) {
    std::string why;
    if (!quietly(module.getContext(), why, [&] {
          return succeeded(verifyAtlasVirtualModule(module));
        })) {
      for (BlockPlan &p : plans)
        restore(p, "invalid order reverted");
      if (options.strict)
        return function.emitError("virtual schedule produced an invalid "
                                  "order for @")
               << record.name << ": " << why;
      function.emitWarning("virtual schedule produced an invalid order for @")
          << record.name << "; reverted: " << why;
    } else if (trialLowering) {
      // Allocation and the word limit are function-wide, so no block can be
      // blamed alone. Give back the smallest gains first until the function
      // lowers again; the source order is known to lower.
      SmallVector<BlockPlan *> scheduled;
      for (BlockPlan &p : plans)
        if (!llvm::equal(p.scheduledOps, p.graph.nodes))
          scheduled.push_back(&p);
      llvm::stable_sort(scheduled, [](BlockPlan *a, BlockPlan *b) {
        return a->gain < b->gain;
      });
      for (BlockPlan *p : scheduled) {
        std::string failure;
        if (lowers(module, failure))
          break;
        restore(*p, "lowering failed after scheduling: " + failure);
      }
    }
  }

  for (BlockPlan &p : plans) {
    remark(p);
    record.blocks.push_back(std::move(p.record));
  }
  return success();
}

struct ScheduleAtlasVirtualPass
    : PassWrapper<ScheduleAtlasVirtualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleAtlasVirtualPass)

  ScheduleAtlasVirtualPass() = default;
  ScheduleAtlasVirtualPass(const ScheduleAtlasVirtualPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "schedule-atlas-virtual"; }
  StringRef getDescription() const final {
    return "Reorder each virtual Atlas block before register allocation";
  }

  Option<bool> report{*this, "report",
                      llvm::cl::desc("Remark modeled cycles and peak pressure "
                                     "for each block"),
                      llvm::cl::init(false)};
  Option<bool> strict{*this, "strict",
                      llvm::cl::desc("Fail instead of reverting when the "
                                     "scheduled order does not verify"),
                      llvm::cl::init(false)};
  Option<int> randomSeed{
      *this, "random-seed",
      llvm::cl::desc("Pick a random legal order, for soundness testing; "
                     "negative disables"),
      llvm::cl::init(-1)};
  Option<std::string> traceFile{
      *this, "trace-file",
      llvm::cl::desc("Write the modeled source and scheduled runs as JSON"),
      llvm::cl::init("")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyAtlasVirtualModule(module)))
      return signalPassFailure();
    ScheduleOptions options;
    options.report = report;
    options.strict = strict;
    if (randomSeed >= 0)
      options.randomSeed = static_cast<unsigned>(randomSeed);

    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    // A trial lowering can only blame the schedule if the source lowers, and
    // the lowering takes exactly one function.
    std::string ignored;
    bool trialLowering = functions.size() == 1 && !options.randomSeed &&
                         lowers(module, ignored);
    std::vector<FunctionScheduleRecord> records;
    for (func::FuncOp function : functions) {
      FunctionScheduleRecord &record = records.emplace_back();
      if (failed(FunctionScheduler(function, options)
                     .run(module, trialLowering, record)))
        return signalPassFailure();
    }

    std::string path = traceFile;
    if (path.empty())
      return;
    std::error_code error;
    llvm::raw_fd_ostream os(path, error);
    if (error) {
      module.emitError("cannot write trace file ")
          << path << ": " << error.message();
      return signalPassFailure();
    }
    writeScheduleTrace(os, records);
  }
};

} // namespace

void mlir::atlas::registerScheduleAtlasVirtualPass() {
  PassRegistration<ScheduleAtlasVirtualPass>();
}
