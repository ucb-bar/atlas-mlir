#include "Atlas/AtlasAddressAnalysis.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasSpilling.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>

using namespace mlir;
using namespace mlir::atlas;

namespace {
class SourceAddressProof {
public:
  SourceAddressProof(func::FuncOp function, Operation *use)
      : dominance(function), use(use) {}

  std::optional<I32AddressSpan> prove(Value value) {
    if (!value.getType().isInteger(32) || !active.insert(value).second)
      return std::nullopt;
    auto result = derive(value);
    active.erase(value);
    return result;
  }

private:
  DominanceInfo dominance;
  Operation *use;
  llvm::DenseSet<Value> active;

  // Recognize an increment without treating wrapping arithmetic as unbounded
  // integer arithmetic. The induction proof below bounds every observed value.
  std::optional<uint32_t> increment(Value next, Value current) {
    auto add = next.getDefiningOp<arith::AddIOp>();
    if (!add || add.getOverflowFlags() != arith::IntegerOverflowFlags::none)
      return std::nullopt;
    if (add.getLhs() == current) return provenI32(add.getRhs());
    if (add.getRhs() == current) return provenI32(add.getLhs());
    return std::nullopt;
  }

  std::optional<I32AddressSpan> derive(Value value) {
    if (auto constant = provenI32(value))
      return I32AddressSpan{*constant, *constant};
    if (auto add = value.getDefiningOp<arith::AddIOp>()) {
      if (add.getOverflowFlags() != arith::IntegerOverflowFlags::none)
        return std::nullopt;
      auto a = prove(add.getLhs()), b = prove(add.getRhs());
      if (!a || !b || uint64_t(a->last) + b->last > UINT32_MAX)
        return std::nullopt;
      return I32AddressSpan{a->first + b->first, a->last + b->last};
    }
    auto pointer = dyn_cast<BlockArgument>(value);
    if (!pointer) return std::nullopt;
    Block *header = pointer.getOwner();
    auto branch = dyn_cast<cf::CondBranchOp>(header->getTerminator());
    if (!branch || branch.getTrueDest() == branch.getFalseDest() ||
        branch.getTrueDest()->getSinglePredecessor() != header ||
        !dominance.dominates(branch.getTrueDest(), use->getBlock()))
      return std::nullopt;
    auto cmp = branch.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp || cmp->getBlock() != header || cmp.getPredicate() != arith::CmpIPredicate::ult)
      return std::nullopt;
    auto counter = dyn_cast<BlockArgument>(cmp.getLhs());
    auto count = provenI32(cmp.getRhs());
    if (!counter || counter.getOwner() != header || !count || !*count)
      return std::nullopt;
    SmallVector<Block *> predecessors(header->getPredecessors());
    if (predecessors.size() != 2 || predecessors[0] == predecessors[1])
      return std::nullopt;
    Block *back = nullptr, *entry = nullptr;
    for (Block *pred : predecessors) {
      if (dominance.dominates(header, pred)) back = pred;
      else entry = pred;
    }
    if (!back || !entry || !dominance.dominates(branch.getTrueDest(), back))
      return std::nullopt;
    auto initial = dyn_cast<cf::BranchOp>(entry->getTerminator());
    auto repeat = dyn_cast<cf::BranchOp>(back->getTerminator());
    unsigned p = pointer.getArgNumber(), c = counter.getArgNumber();
    if (!initial || !repeat || initial.getDest() != header || repeat.getDest() != header ||
        initial.getDestOperands().size() != header->getNumArguments() ||
        repeat.getDestOperands().size() != header->getNumArguments())
      return std::nullopt;
    auto zero = provenI32(initial.getDestOperands()[c]);
    auto one = increment(repeat.getDestOperands()[c], counter);
    auto stride = increment(repeat.getDestOperands()[p], pointer);
    if (!zero || *zero || !one || *one != 1 || !stride || !*stride || *stride % 32)
      return std::nullopt;
    auto base = prove(initial.getDestOperands()[p]);
    if (!base) return std::nullopt;
    uint64_t last = uint64_t(base->last) + uint64_t(*count - 1) * *stride;
    if (last > UINT32_MAX) return std::nullopt;
    return I32AddressSpan{base->first, uint32_t(last)};
  }
};
} // namespace

std::optional<I32AddressSpan> mlir::atlas::provenI32AddressSpan(Value value,
                                                            Operation *use) {
  auto function = use->getParentOfType<func::FuncOp>();
  if (!function) {
    auto constant = provenI32(value);
    return constant ? std::optional<I32AddressSpan>{{*constant, *constant}} : std::nullopt;
  }
  return SourceAddressProof(function, use).prove(value);
}

std::optional<int32_t> mlir::atlas::atlasCFGValueID(Value value) {
  auto function = value.getParentBlock()->getParentOp()->getParentOfType<func::FuncOp>();
  if (!function) function = dyn_cast<func::FuncOp>(value.getParentBlock()->getParentOp());
  if (!function) return std::nullopt;
  int32_t next = 0;
  auto visit = [&](Value v) {
    bool tracked = !isAtlasMemoryControl(v) && !isVirtualDMAImmediate(v) && (v.getType().isInteger(1) ||
        v.getType().isInteger(32) || isa<VirtualBF16Type, VirtualFP8Type>(v.getType()));
    if (!tracked) return std::optional<int32_t>{};
    int32_t id = next++;
    return v == value ? std::optional<int32_t>{id} : std::nullopt;
  };
  for (Block &block : function.getBody()) {
    for (Value arg : block.getArguments()) if (auto id = visit(arg)) return id;
    for (Operation &op : block) for (Value result : op.getResults())
      if (auto id = visit(result)) return id;
  }
  return std::nullopt;
}


LogicalResult mlir::atlas::verifyAtlasOutputBuffer(Operation *operation) {
  auto function = dyn_cast<func::FuncOp>(operation);
  if (!function || !function->hasAttr("atlas.output_dram_bytes")) return success();
  auto size = function->getAttrOfType<IntegerAttr>("atlas.output_dram_bytes");
  auto base = function->getAttrOfType<IntegerAttr>("atlas.output_dram_base");
  auto valid = [](IntegerAttr value) {
    return value && value.getType().isSignlessInteger(64) && value.getInt() > 0;
  };
  if (!valid(size) || !valid(base) || base.getInt() < 0x80000000LL ||
      base.getInt() > UINT32_MAX || base.getInt() % 32 || size.getInt() % 32 ||
      uint64_t(base.getInt()) + uint64_t(size.getInt()) > (1ULL << 32))
    return function.emitOpError("explicit output buffer requires aligned i64 base/bytes within selected DRAM");
  uint64_t first = base.getInt(), end = first + size.getInt();
  bool readsOutput = false;
  WalkResult result = function.walk([&](Operation *op) {
    if (isa<VirtualOutputBF16Op>(op)) {
      op->emitOpError("explicit output buffers require DMA stores, not indexed outputs");
      return WalkResult::interrupt();
    }
    bool load = isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op>(op);
    bool store = isa<VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op);
    if (load || store) {
      auto address = provenI32AddressSpan(op->getOperand(load ? 1 : 2),op);
      auto bytes = provenI32(op->getOperand(load ? 2 : 3));
      if (!address || !bytes || uint64_t(address->last)+*bytes > (1ULL << 32)) {
        op->emitOpError("explicit output buffer needs proven DMA access bounds");
        return WalkResult::interrupt();
      }
      uint64_t stop = uint64_t(address->last)+*bytes;
      if (store && (address->first < first || stop > end)) {
        op->emitOpError("DMA store exceeds the declared output buffer");
        return WalkResult::interrupt();
      }
      readsOutput |= load && address->first < end && first < stop;
    } else if (isa<VirtualInputFP8Op, VirtualInputBF16Op>(op)) {
      auto input = function->getAttrOfType<IntegerAttr>("atlas.input_dram_base");
      auto index = op->getAttrOfType<IntegerAttr>("index");
      if (!valid(input) || !index || index.getInt() < 0 ||
          uint64_t(index.getInt()) > UINT32_MAX / 2048) {
        op->emitOpError("indexed input has no checked DRAM range");
        return WalkResult::interrupt();
      }
      uint64_t start = uint64_t(input.getInt())+uint64_t(index.getInt())*2048;
      uint64_t stop = start+(isa<VirtualInputFP8Op>(op) ? 1024 : 2048);
      if (stop > (1ULL << 32) || (start < end && first < stop)) {
        op->emitOpError("indexed input overlaps the declared output buffer or overflows DRAM");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return failure();
  return readsOutput ? verifyAtlasOutputInitialization(function, first, end) : success();
}
