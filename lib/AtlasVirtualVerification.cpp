#include "Atlas/AtlasVirtualVerification.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::atlas;

namespace {
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
    if (isa<VirtualVPUUnaryOp, VirtualVPUBinaryOp>(operation))
      continue;
    if (cfg && isa<arith::ConstantOp, arith::AddIOp, arith::CmpIOp>(operation))
      continue;
    if (cfg) {
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
  return success();
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
      if (isa<VirtualOutputBF16Op>(op) && &block != returnBlock) {
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
