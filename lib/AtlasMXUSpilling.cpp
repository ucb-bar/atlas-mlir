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
      unsigned unit = acc ? cast<VirtualMXUAccType>(resource).getUnit()
                          : cast<VirtualMXUWeightType>(resource).getUnit();
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : arg.getUses()) uses.push_back(&use);
      arg.setType(acc ? Type(VirtualBF16Type::get(function.getContext()))
                      : Type(VirtualFP8Type::get(function.getContext())));
      OperationState os(function.getLoc(), acc ? "atlas.virtual_mxu_load_acc_bf16"
                                               : "atlas.virtual_mxu_load_weight");
      os.addOperands({state, arg});
      os.addTypes({initial.getType(), resource});
      os.addAttribute("unit", builder.getI32IntegerAttr(unit));
      os.addAttribute("atlas.mxu_spill", builder.getStringAttr("restore"));
      Operation *load = builder.create(os);
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
          OperationState os(term->getLoc(), "atlas.virtual_mxu_readout_bf16");
          os.addOperands({state, handle});
          os.addTypes({state.getType(), VirtualBF16Type::get(function.getContext())});
          os.addAttribute("atlas.mxu_spill", builder.getStringAttr("save"));
          Operation *readout = builder.create(os);
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
          Value tile = append(op, state,
              acc ? "atlas.virtual_reload_bf16" : "atlas.virtual_reload_fp8",
              resource.saved, acc ? Type(VirtualBF16Type::get(op->getContext()))
                                  : Type(VirtualFP8Type::get(op->getContext())));
          resource.resident = append(op, state,
              acc ? "atlas.virtual_mxu_load_acc_bf16" : "atlas.virtual_mxu_load_weight",
              tile, handle.getType(), true);
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

  Value append(Operation *before, Value &state, StringRef name, Value operand,
               Type result, bool load = false) {
    OpBuilder builder(before);
    OperationState os(before->getLoc(), name);
    os.addOperands({state, operand});
    os.addTypes({state.getType(), result});
    if (load) {
      os.addAttribute("unit", builder.getI32IntegerAttr(unit(result)));
      os.addAttribute("atlas.mxu_spill", builder.getStringAttr("restore"));
    }
    Operation *op = builder.create(os);
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
    if (acc) {
      // Selected MXU0/MXU1 completed buffers are BF16. BF16 pop copies their
      // raw halves, unlike FP8 readout, which would introduce quantization.
      tile = append(before, state, "atlas.virtual_mxu_readout_bf16",
                    resource.resident, VirtualBF16Type::get(before->getContext()));
      tile.getDefiningOp()->setAttr("atlas.mxu_spill",
                                   StringAttr::get(before->getContext(), "save"));
    }
    if (acc || !resource.saved) {
      OpBuilder builder(before);
      OperationState os(before->getLoc(), acc ? "atlas.virtual_spill_bf16"
                                             : "atlas.virtual_spill_fp8");
      os.addOperands({state, tile});
      os.addTypes({state.getType(), VirtualSpillType::get(before->getContext())});
      os.addAttribute("slot_word", builder.getI32IntegerAttr(
          VirtualAllocationPlan().fixed().spillWord));
      Operation *store = builder.create(os);
      state = store->getResult(0);
      resource.saved = store->getResult(1);
    }
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
