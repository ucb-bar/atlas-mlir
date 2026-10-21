#include "Atlas/AtlasSpilling.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;
using namespace mlir::atlas;

namespace {
// Restores MXU state of `handle`'s type from its saved tile.
Operation *createMXURestore(OpBuilder &builder, Location loc, Value state,
                            Value tile, Type handle) {
  Operation *load;
  if (auto acc = dyn_cast<VirtualMXUAccType>(handle))
    load = VirtualMXULoadAccBF16Op::create(builder, loc, state.getType(), handle,
                                           state, tile, acc.getUnit());
  else
    load = VirtualMXULoadWeightOp::create(
        builder, loc, state.getType(), handle, state, tile,
        cast<VirtualMXUWeightType>(handle).getUnit());
  load->setAttr("atlas.mxu_spill", builder.getStringAttr("restore"));
  return load;
}

// Saves a completed accumulator as the BF16 tile it holds. A BF16 pop copies
// its raw halves, unlike an FP8 readout, which would quantize them.
Operation *createMXUSave(OpBuilder &builder, Location loc, Value state,
                         Value acc) {
  Operation *readout = VirtualMXUReadoutBF16Op::create(
      builder, loc, state.getType(), VirtualBF16Type::get(builder.getContext()),
      state, acc);
  readout->setAttr("atlas.mxu_spill", builder.getStringAttr("save"));
  return readout;
}

// Normalize explicit resource block arguments to ordinary tile edges. A BF16
// readout consumes the completed accumulator once, even when a conditional
// branch forwards it to both successors. The destination starts a new local
// handle. Tensor allocation/spilling owns transport after this boundary.
LogicalResult materializeMXUEdges(func::FuncOp function, unsigned limit,
                                 unsigned &spills) {
  unsigned work = 0;
  for (Block &block : function.getBody()) {
    for (BlockArgument arg : block.getArguments())
      work += isa<VirtualMXUWeightType, VirtualMXUAccType>(arg.getType());
    llvm::DenseSet<Value> snapshots;
    Operation *term = block.getTerminator();
    if (isa<cf::BranchOp, cf::CondBranchOp>(term))
      for (Value value : term->getOperands())
        if (isa<VirtualMXUAccType>(value.getType())) snapshots.insert(value);
    work += snapshots.size();
  }
  if (spills > limit || work > limit - spills)
    return function.emitOpError("MXU CFG materialization work limit exceeded");
  spills += work;
  for (Block &block : function.getBody()) {
    if (&block == &function.front()) continue;
    Value initial = block.getArgument(0), state = initial;
    SmallVector<OpOperand *> stateUses;
    for (OpOperand &use : initial.getUses()) stateUses.push_back(&use);
    OpBuilder builder(&block, block.begin());
    for (BlockArgument arg : block.getArguments().drop_front()) {
      Type resource = arg.getType();
      if (!isa<VirtualMXUWeightType, VirtualMXUAccType>(resource)) continue;
      bool acc = isa<VirtualMXUAccType>(resource);
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : arg.getUses()) uses.push_back(&use);
      arg.setType(acc ? Type(VirtualBF16Type::get(function.getContext()))
                      : Type(VirtualFP8Type::get(function.getContext())));
      Operation *load = createMXURestore(builder, function.getLoc(), state, arg, resource);
      state = load->getResult(0);
      for (OpOperand *use : uses) use->set(load->getResult(1));
    }
    for (OpOperand *use : stateUses) use->set(state);
  }
  for (Block &block : function.getBody()) {
    Operation *term = block.getTerminator();
    Value state;
    if (auto branch = dyn_cast<cf::BranchOp>(term))
      state = branch.getDestOperands().front();
    else if (auto branch = dyn_cast<cf::CondBranchOp>(term))
      state = branch.getTrueDestOperands().front();
    else continue;
    Value initial = state;
    OpBuilder builder(term);
    llvm::DenseMap<Value, Value> snapshots;
    for (OpOperand &operand : term->getOpOperands()) {
      Value handle = operand.get();
      if (!isa<VirtualMXUWeightType, VirtualMXUAccType>(handle.getType())) continue;
      Value &tile = snapshots[handle];
      if (!tile) {
        if (isa<VirtualMXUWeightType>(handle.getType())) {
          auto load = handle.getDefiningOp<VirtualMXULoadWeightOp>();
          if (!load) return term->emitOpError("MXU edge weight has no local source");
          tile = load.getSrc();
        } else {
          Operation *readout = createMXUSave(builder, term->getLoc(), state, handle);
          state = readout->getResult(0);
          tile = readout->getResult(1);
        }
      }
      operand.set(tile);
    }
    for (OpOperand &operand : term->getOpOperands())
      if (operand.get() == initial) operand.set(state);
  }
  return success();
}

struct Resource {
  Value resident;
  Value saved;
  Value source;
};

// Handles cannot cross blocks. Track original SSA identities independently of
// their current resident versions, so insertion never changes victim ordering.
class MXUSpiller {
public:
  MXUSpiller(Block &block, unsigned limit, unsigned &spills)
      : limit(limit), spills(spills) {
    for (Operation &op : block) {
      unsigned index = operations.size();
      operations.push_back(&op);
      for (Value operand : op.getOperands())
        if (isa<VirtualMXUWeightType, VirtualMXUAccType>(operand.getType()))
          uses[operand].push_back(index);
    }
  }

  LogicalResult run() {
    for (unsigned index = 0; index < operations.size(); ++index) {
      Operation *op = operations[index];
      // Weight handles are immutable and cease occupying a slot at last use.
      for (auto &unit : resident[0])
        llvm::erase_if(unit, [&](Value value) {
          if (nextUse(value, index) != never) return false;
          resources[value].resident = {};
          return true;
        });
      bool loadWeight = isa<VirtualMXULoadWeightOp>(op);
      bool startAcc = isa<VirtualMXULoadAccBF16Op, VirtualMXULoadAccFP8Op,
                          VirtualMXUResetOp>(op);
      bool consumeAcc = isa<VirtualMXUAccumulateOp, VirtualMXUReadoutBF16Op,
                            VirtualMXUReadoutFP8Op>(op);
      if (!loadWeight && !startAcc && !consumeAcc)
        continue;
      SmallVector<Value> original(op->getOperands());
      Value state = original[0];
      for (unsigned i = 1; i < original.size(); ++i) {
        Value handle = original[i];
        if (!uses.contains(handle)) continue;
        Resource &resource = resources[handle];
        if (!resource.resident) {
          if (failed(makeRoom(handle.getType(), index, op, state)))
            return failure();
          bool acc = isa<VirtualMXUAccType>(handle.getType());
          OpBuilder builder(op);
          Value tile = append(state, createAtlasReload(
              builder, op->getLoc(), state, resource.saved,
              acc ? Type(VirtualBF16Type::get(op->getContext()))
                  : Type(VirtualFP8Type::get(op->getContext()))));
          resource.resident = append(state, createMXURestore(
              builder, op->getLoc(), state, tile, handle.getType()));
          resident[acc][unit(handle.getType())].push_back(handle);
        }
        op->setOperand(i, resource.resident);
      }
      if (loadWeight || startAcc) {
        Value result = op->getResult(1);
        if (failed(makeRoom(result.getType(), index, op, state)))
          return failure();
        resources[result] = {result, {}, loadWeight ? original[1] : Value{}};
        resident[startAcc][unit(result.getType())].push_back(result);
      } else if (consumeAcc) {
        // Accumulation consumes one version and produces the next in its slot.
        Value acc = original[isa<VirtualMXUAccumulateOp>(op) ? 3 : 1];
        auto &live = resident[1][unit(acc.getType())];
        llvm::erase(live, acc);
        resources.erase(acc);
        if (isa<VirtualMXUAccumulateOp>(op)) {
          Value next = op->getResult(1);
          resources[next] = {next, {}, {}};
          live.push_back(next);
        }
      }
      op->setOperand(0, state);
    }
    return success();
  }

private:
  static constexpr unsigned never = std::numeric_limits<unsigned>::max();
  unsigned limit;
  unsigned &spills;
  SmallVector<Operation *> operations;
  llvm::DenseMap<Value, SmallVector<unsigned>> uses;
  llvm::DenseMap<Value, Resource> resources;
  SmallVector<Value> resident[2][2]; // weights/accumulators, then owning unit

  unsigned unit(Type type) {
    if (auto acc = dyn_cast<VirtualMXUAccType>(type)) return acc.getUnit();
    return cast<VirtualMXUWeightType>(type).getUnit();
  }

  unsigned nextUse(Value value, unsigned index) {
    for (unsigned use : uses[value])
      if (use >= index) return use;
    return never;
  }

  // Continues the state chain from `op`, returning its value.
  static Value append(Value &state, Operation *op) {
    state = op->getResult(0);
    return op->getResult(1);
  }

  LogicalResult makeRoom(Type type, unsigned index, Operation *before,
                         Value &state) {
    bool acc = isa<VirtualMXUAccType>(type);
    auto &live = resident[acc][unit(type)];
    if (live.size() < kVirtualMXUSlots) return success();
    Value victim;
    unsigned farthest = 0;
    for (Value value : live) {
      unsigned use = nextUse(value, index);
      // Never evict an operand needed by this very instruction.
      if (use > index && (!victim || use > farthest)) {
        victim = value;
        farthest = use;
      }
    }
    if (!victim || spills >= limit)
      return before->emitOpError("MXU spill rewrite work limit or working slots exhausted");
    Resource &resource = resources[victim];
    Value tile = resource.source;
    OpBuilder builder(before);
    // Selected MXU0/MXU1 completed buffers are BF16.
    if (acc)
      tile = append(state, createMXUSave(builder, before->getLoc(), state,
                                         resource.resident));
    if (acc || !resource.saved)
      resource.saved = append(state, createAtlasSpillStore(
          builder, before->getLoc(), state, tile,
          VirtualAllocationPlan().fixed().spillWord));
    resource.resident = {};
    llvm::erase(live, victim);
    ++spills;
    return success();
  }
};

struct SpillAtlasMXUStatePass
    : PassWrapper<SpillAtlasMXUStatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpillAtlasMXUStatePass)
  SpillAtlasMXUStatePass() = default;
  SpillAtlasMXUStatePass(const SpillAtlasMXUStatePass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "spill-atlas-mxu-state"; }
  StringRef getDescription() const final {
    return "Split MXU weight and accumulator lifetimes with exact tile saves";
  }
  Option<unsigned> maxSpills{*this, "max-spills", llvm::cl::init(256),
                            llvm::cl::desc("Bound MXU spill rewrites")};
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyAtlasVirtualModule(module, /*allowMXUPressure=*/true,
                                      /*allowMXUCFG=*/true)))
      return signalPassFailure();
    unsigned spills = 0;
    for (auto function : module.getOps<func::FuncOp>())
      if (failed(spillAtlasMXUState(function, maxSpills, spills)) ||
          failed(placeAtlasSpillSlots(function,
              VirtualAllocationPlan().fixed().spillWindowWords)))
        return signalPassFailure();
    if (failed(verifyAtlasVirtualModule(module)))
      return signalPassFailure();
  }
};
} // namespace

LogicalResult mlir::atlas::spillAtlasMXUState(func::FuncOp function,
                                            unsigned maxSpills,
                                            unsigned &spills) {
  if (failed(materializeMXUEdges(function, maxSpills, spills)))
    return failure();
  for (Block &block : function.getBody())
    if (failed(MXUSpiller(block, maxSpills, spills).run()))
      return failure();
  return success();
}

void mlir::atlas::registerSpillAtlasMXUStatePass() {
  PassRegistration<SpillAtlasMXUStatePass>();
}

std::unique_ptr<Pass> mlir::atlas::createSpillAtlasMXUStatePass() {
  return std::make_unique<SpillAtlasMXUStatePass>();
}
