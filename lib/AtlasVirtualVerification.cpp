#include "Atlas/AtlasVirtualVerification.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <iterator>

using namespace mlir;
using namespace mlir::atlas;

namespace {
// Track the explicit transfers in flight within a block. They may complete in
// any order.
struct VirtualDMATransfers {
  SmallVector<Value, kMaxPendingVirtualDMA> pending;

  LogicalResult verify(Operation &op) {
    if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op, VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op)) {
      if (pending.size() == kMaxPendingVirtualDMA)
        return op.emitOpError("must complete a pending DMA before another launch; at most ")
               << kMaxPendingVirtualDMA << " may be pending";
      pending.push_back(op.getResult(1));
      return success();
    }
    auto found = llvm::find(pending, op.getOperand(1));
    if (found == pending.end())
      return op.emitOpError("must consume a pending DMA transfer");
    pending.erase(found);
    return success();
  }

  LogicalResult verifyClosed(Operation *where) {
    if (!pending.empty())
      return where->emitOpError("must complete pending DMA before block exit");
    return success();
  }
};

// Track the weights resident on each unit and the current version of each
// live accumulator, at most one per hardware slot, within a block. A weight
// stays resident while it has uses left, so every use finds it.
struct VirtualMXUResources {
  SmallVector<Value, kVirtualMXUSlots> weights[2];
  SmallVector<Value, kVirtualMXUSlots> accumulators[2];
  llvm::DenseMap<Value, unsigned> remainingUses;

  // The resident weights of `unit` that something still uses.
  SmallVector<Value, kVirtualMXUSlots> &liveWeights(unsigned unit) {
    llvm::erase_if(weights[unit],
                   [&](Value weight) { return remainingUses[weight] == 0; });
    return weights[unit];
  }

  LogicalResult startAccumulator(Operation &op, unsigned unit, Value acc,
                                 StringRef verb) {
    if (accumulators[unit].size() == kVirtualMXUSlots)
      return op.emitOpError("cannot ")
             << verb << " a unit with " << kVirtualMXUSlots
             << " live accumulators";
    accumulators[unit].push_back(acc);
    return success();
  }

  // Replace the live accumulator version `acc` with `next`, or retire it.
  LogicalResult consumeAccumulator(Operation &op, unsigned unit, Value acc,
                                   Value next) {
    auto found = llvm::find(accumulators[unit], acc);
    if (found == accumulators[unit].end())
      return op.emitOpError("must consume the current accumulator version; "
                            "stale accumulator handle");
    if (next)
      *found = next;
    else
      accumulators[unit].erase(found);
    return success();
  }

  LogicalResult verify(Operation &op) {
    if (auto load = dyn_cast<VirtualMXULoadWeightOp>(op)) {
      Value weight = load.getWeight();
      unsigned unit = cast<VirtualMXUWeightType>(weight.getType()).getUnit();
      if (unit > 1)
        return op.emitOpError("MXU unit must be in [0, 1]");
      if (liveWeights(unit).size() == kVirtualMXUSlots)
        return op.emitOpError("cannot load a weight on a unit with ")
               << kVirtualMXUSlots << " live weights";
      weights[unit].push_back(weight);
      remainingUses[weight] =
          std::distance(weight.use_begin(), weight.use_end());
      return success();
    }
    if (isa<VirtualMXULoadAccFP8Op, VirtualMXULoadAccBF16Op>(op)) {
      Value acc = op.getResult(1);
      unsigned unit = cast<VirtualMXUAccType>(acc.getType()).getUnit();
      if (unit > 1)
        return op.emitOpError("MXU unit must be in [0, 1]");
      return startAccumulator(op, unit, acc, "load");
    }
    if (auto reset = dyn_cast<VirtualMXUResetOp>(op)) {
      unsigned unit = cast<VirtualMXUAccType>(reset.getAcc().getType()).getUnit();
      if (unit > 1)
        return op.emitOpError("MXU unit must be in [0, 1]");
      --remainingUses[reset.getWeight()];
      return startAccumulator(op, unit, reset.getAcc(), "reset");
    }
    if (auto accumulate = dyn_cast<VirtualMXUAccumulateOp>(op)) {
      unsigned unit =
          cast<VirtualMXUAccType>(accumulate.getAcc().getType()).getUnit();
      if (unit > 1)
        return op.emitOpError("MXU unit must be in [0, 1]");
      --remainingUses[accumulate.getWeight()];
      return consumeAccumulator(op, unit, accumulate.getAcc(),
                                accumulate.getNextAcc());
    }
    Value acc = op.getOperand(1);
    unsigned unit = cast<VirtualMXUAccType>(acc.getType()).getUnit();
    if (unit > 1)
      return op.emitOpError("MXU unit must be in [0, 1]");
    return consumeAccumulator(op, unit, acc, Value{});
  }

  LogicalResult verifyClosed(Operation *where) {
    for (unsigned unit = 0; unit < 2; ++unit)
      if (!accumulators[unit].empty())
        return where->emitOpError(
                   "must read out the live MXU accumulator before block exit; unit ")
               << unit;
    return success();
  }
};

// The virtual state is an SSA edge value. Every successor receives the current
// state as its first block argument; a conditional branch may pass that same
// state to both mutually exclusive successors. MLIR checks ordinary value
// dominance and the remaining block-argument types.
LogicalResult verifySuccessor(Operation &op, Block *dest, ValueRange operands,
                              Value state) {
  if (dest->getNumArguments() == 0 ||
      !isa<VirtualStateType>(dest->getArgument(0).getType()) ||
      operands.empty() || operands.front() != state) {
    op.emitOpError("must pass the current virtual state as successor argument 0");
    return failure();
  }
  return success();
}

LogicalResult verifyVirtualBlock(Block &block, bool entry, bool cfg,
                                 unsigned &starts, unsigned &outputs,
                                 llvm::DenseSet<int64_t> &outputIndices) {
  Value state;
  VirtualMXUResources mxu;
  VirtualDMATransfers dma;
  if (cfg) {
    if (entry) {
      for (BlockArgument arg : block.getArguments())
        if (!arg.getType().isInteger(1) && !arg.getType().isInteger(32)) {
          block.getParentOp()->emitOpError(
              "virtual entry arguments must be i1/i32 controls; tiles enter through virtual_input_bf16");
          return failure();
        }
    } else {
      if (block.getNumArguments() == 0 ||
          !isa<VirtualStateType>(block.getArgument(0).getType())) {
        block.getParentOp()->emitOpError(
            "non-entry virtual block requires a virtual_state argument 0");
        return failure();
      }
      state = block.getArgument(0);
      for (BlockArgument arg : block.getArguments().drop_front())
        if (!isa<VirtualBF16Type>(arg.getType()) &&
            !arg.getType().isInteger(1) && !arg.getType().isInteger(32)) {
          block.getParentOp()->emitOpError(
              "virtual block arguments must be BF16 tiles or i1/i32 controls");
          return failure();
        }
    }
  }

  for (Operation &operation : block.getOperations()) {
    // Dominance alone permits cross-block captures of resident handles.
    for (Value operand : operation.getOperands()) {
      if (isa<VirtualMXUWeightType, VirtualMXUAccType>(operand.getType()) &&
          operand.getParentBlock() != &block)
        return operation.emitOpError(
            "virtual MXU handles cannot cross CFG blocks");
      if (isa<VirtualDMALoadFP8Type, VirtualDMALoadBF16Type,
              VirtualDMAStoreType>(operand.getType()) &&
          operand.getParentBlock() != &block)
        return operation.emitOpError(
            "pending DMA handles cannot cross CFG blocks");
    }
    if (!dma.pending.empty() &&
        isa<VirtualInputBF16Op, VirtualInputFP8Op, VirtualOutputBF16Op,
            VirtualPackFP8Op>(operation))
      return operation.emitOpError(
          "must complete pending DMA before implicit memory operations");
    if (auto start = dyn_cast<VirtualStartOp>(operation)) {
      if (!entry || &operation != &block.front() || ++starts != 1 || state) {
        start.emitOpError("virtual_start must be the unique first entry operation");
        return failure();
      }
      state = start.getNext();
      continue;
    }
    if (auto input = dyn_cast<VirtualInputBF16Op>(operation)) {
      if (!state || input.getState() != state) {
        input.emitOpError("nonlinear virtual state chain");
        return failure();
      }
      state = input.getNext();
      continue;
    }
    if (auto input = dyn_cast<VirtualInputFP8Op>(operation)) {
      if (!state || input.getState() != state) {
        input.emitOpError("nonlinear virtual state chain");
        return failure();
      }
      state = input.getNext();
      continue;
    }
    if (auto output = dyn_cast<VirtualOutputBF16Op>(operation)) {
      if (!state || output.getState() != state) {
        output.emitOpError("nonlinear virtual state chain");
        return failure();
      }
      int64_t index = output.getIndexAttr().getValue().getSExtValue();
      if (!outputIndices.insert(index).second) {
        output.emitOpError("duplicate virtual output index ") << index;
        return failure();
      }
      state = output.getNext();
      ++outputs;
      continue;
    }
    if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op,
            VirtualDMAAwaitFP8Op, VirtualDMAAwaitBF16Op,
            VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op,
            VirtualDMAWaitOp>(operation)) {
      if (!state || operation.getOperand(0) != state)
        return operation.emitOpError("nonlinear virtual state chain");
      if (failed(dma.verify(operation)))
        return failure();
      state = operation.getResult(0);
      if (isa<VirtualDMAWaitOp>(operation))
        ++outputs;
      continue;
    }
    if (isa<VirtualMXULoadWeightOp, VirtualMXULoadAccFP8Op,
            VirtualMXULoadAccBF16Op, VirtualMXUResetOp, VirtualMXUAccumulateOp,
            VirtualMXUReadoutBF16Op, VirtualMXUReadoutFP8Op>(operation)) {
      if (!state || operation.getOperand(0) != state)
        return operation.emitOpError("nonlinear virtual state chain");
      if (failed(mxu.verify(operation)))
        return failure();
      state = operation.getResult(0);
      continue;
    }
    if (auto matmul = dyn_cast<VirtualMXUMatmulOp>(operation)) {
      unsigned unit = matmul.getUnit();
      if (unit > 1)
        return matmul.emitOpError("unit must be in [0, 1]");
      if (!mxu.accumulators[unit].empty())
        return matmul.emitOpError(
            "legacy virtual_mxu_matmul cannot overwrite a live accumulator");
      // Legacy matmul overwrites a weight slot and completes readout
      // internally.
      if (!mxu.liveWeights(unit).empty())
        return matmul.emitOpError(
            "legacy virtual_mxu_matmul cannot overwrite a live weight");
      continue;
    }
    if (isa<VirtualVPUUnaryOp, VirtualVPUBinaryOp, VirtualPackFP8Op,
            VirtualScaleConstantOp>(operation))
      continue;
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      Type type = constant.getResult().getType();
      if (!isa<IntegerAttr>(constant.getValue()) ||
          (!type.isInteger(1) && !type.isInteger(32)))
        return constant.emitOpError("virtual scalar constants require i1 or i32");
      continue;
    }
    if (auto add = dyn_cast<arith::AddIOp>(operation)) {
      if (!add.getResult().getType().isInteger(32))
        return add.emitOpError("virtual scalar addition requires i32");
      if (add.getOverflowFlags() != arith::IntegerOverflowFlags::none)
        return add.emitOpError("virtual scalar addition requires wrapping arithmetic");
      continue;
    }
    if (auto cmp = dyn_cast<arith::CmpIOp>(operation)) {
      if (!cmp.getLhs().getType().isInteger(32) ||
          !cmp.getRhs().getType().isInteger(32))
        return cmp.emitOpError("virtual scalar comparison requires i32 operands");
      continue;
    }
    if (cfg) {
      if (operation.hasTrait<OpTrait::IsTerminator>() &&
          (failed(mxu.verifyClosed(&operation)) ||
           failed(dma.verifyClosed(&operation))))
        return failure();
      if (auto branch = dyn_cast<cf::BranchOp>(operation))
        return verifySuccessor(operation, branch.getDest(),
                               branch.getDestOperands(), state);
      if (auto branch = dyn_cast<cf::CondBranchOp>(operation)) {
        if (failed(verifySuccessor(operation, branch.getTrueDest(),
                                   branch.getTrueDestOperands(), state)) ||
            failed(verifySuccessor(operation, branch.getFalseDest(),
                                   branch.getFalseDestOperands(), state)))
          return failure();
        return success();
      }
      if (auto ret = dyn_cast<func::ReturnOp>(operation)) {
        if (!state || ret.getNumOperands() != 1 || ret.getOperand(0) != state) {
          ret.emitOpError("must return the current virtual state");
          return failure();
        }
        return success();
      }
    }
    operation.emitOpError("operation is outside the virtual Atlas stage");
    return failure();
  }
  if (cfg) {
    block.getParentOp()->emitOpError("virtual CFG block has no terminator");
    return failure();
  }
  if (failed(mxu.verifyClosed(block.getParentOp())))
    return failure();
  return dma.verifyClosed(block.getParentOp());
}

LogicalResult verifyVirtualFunction(func::FuncOp function) {
  if (function.isExternal() || function.getBody().empty() ||
      function.getFunctionType().getNumResults() != 1 ||
      !isa<VirtualStateType>(function.getFunctionType().getResult(0))) {
    function.emitOpError("virtual function must have a body and return one virtual_state");
    return failure();
  }
  unsigned starts = 0;
  unsigned outputs = 0;
  llvm::DenseSet<int64_t> outputIndices;
  Block *returnBlock = nullptr;
  for (Block &block : function.getBody()) {
    if (failed(verifyVirtualBlock(block, &block == &function.getBody().front(),
                                  /*cfg=*/true, starts, outputs,
                                  outputIndices)))
      return failure();
    if (isa<func::ReturnOp>(block.getTerminator())) {
      if (returnBlock) {
        function.emitOpError("virtual CFG requires one return block");
        return failure();
      }
      returnBlock = &block;
    }
  }
  if (starts != 1 || outputs == 0) {
    function.emitOpError("virtual CFG requires one start and at least one output");
    return failure();
  }
  if (!returnBlock) {
    function.emitOpError("virtual CFG requires a return block");
    return failure();
  }
  // For this first CFG slice, all externally visible outputs live in the
  // unique return block. Thus every terminating path executes them exactly
  // once. More general path-sensitive boundary effects need a separate proof.
  for (Block &block : function.getBody())
    for (Operation &op : block)
      if (isa<VirtualOutputBF16Op, VirtualDMAStoreFP8Op,
              VirtualDMAStoreBF16Op, VirtualDMAWaitOp>(op) &&
          &block != returnBlock) {
        op.emitOpError("virtual CFG outputs must be in the return block");
        return failure();
      }
  llvm::DenseSet<Block *> reachable;
  llvm::SmallVector<Block *> pending{&function.getBody().front()};
  while (!pending.empty()) {
    Block *block = pending.pop_back_val();
    if (!reachable.insert(block).second)
      continue;
    for (Block *successor : block->getTerminator()->getSuccessors())
      pending.push_back(successor);
  }
  if (reachable.size() != function.getBody().getBlocks().size()) {
    function.emitOpError("virtual CFG contains an unreachable block");
    return failure();
  }
  return success();
}

struct VerifyAtlasVirtualStreamPass
    : PassWrapper<VerifyAtlasVirtualStreamPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAtlasVirtualStreamPass)

  StringRef getArgument() const final { return "verify-atlas-virtual-stream"; }
  StringRef getDescription() const final {
    return "Check virtual BF16 SSA values, CFG state edges, and stage isolation";
  }

  void runOnOperation() override {
    if (failed(mlir::atlas::verifyAtlasVirtualModule(getOperation())))
      return signalPassFailure();
  }
};
} // namespace

LogicalResult mlir::atlas::verifyAtlasVirtualModule(ModuleOp module) {
  bool hasFunctions = llvm::any_of(module.getBody()->getOperations(),
                                   [](Operation &op) { return isa<func::FuncOp>(op); });
  if (hasFunctions) {
    for (Operation &op : module.getBody()->getOperations()) {
      auto function = dyn_cast<func::FuncOp>(op);
      if (!function)
        return op.emitOpError(
            "virtual CFG module must contain only func.func operations");
      if (failed(verifyVirtualFunction(function)))
        return failure();
    }
    return success();
  }
  unsigned starts = 0;
  unsigned outputs = 0;
  llvm::DenseSet<int64_t> outputIndices;
  if (failed(verifyVirtualBlock(*module.getBody(), /*entry=*/true,
                                /*cfg=*/false, starts, outputs,
                                outputIndices)))
    return failure();
  if (starts != 1 || outputs == 0)
    return module.emitOpError(
        "virtual stream requires one start and at least one output");
  return success();
}

void mlir::atlas::registerVerifyAtlasVirtualStreamPass() {
  PassRegistration<VerifyAtlasVirtualStreamPass>();
}
