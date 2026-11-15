#include "Atlas/AtlasProofWork.h"
#include "Atlas/AtlasAddressAnalysis.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTarget.h"
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
  SourceAddressProof(func::FuncOp function, Operation *use, bool scalar = false)
      : dominance(function), use(use), scalar(scalar) {}

  std::optional<I32AddressSpan> prove(Value value) {
    atlasProofTransfer(AtlasProofKind::SourceAddress);
    if ((!value.getType().isInteger(32) && !(scalar && value.getType().isInteger(1))) || !active.insert(value).second)
      return std::nullopt;
    auto result = derive(value);
    active.erase(value);
    return result;
  }

private:
  DominanceInfo dominance;
  Operation *use;
  bool scalar;
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

  std::optional<I32AddressSpan> refine(Value value, Value condition, bool truth) {
    auto range = prove(value);
    if (auto extend = condition.getDefiningOp<arith::ExtUIOp>()) condition = extend.getIn();
    auto cmp = condition.getDefiningOp<arith::CmpIOp>();
    if (!cmp || cmp.getLhs() != value) return range;
    auto bound = provenI32(cmp.getRhs());
    if (!bound) return range;
    uint32_t first = range ? range->first : 0, last = range ? range->last : UINT32_MAX;
    switch (cmp.getPredicate()) {
    case arith::CmpIPredicate::ult:
      if (truth) { if (!*bound) return std::nullopt; last = std::min(last, *bound - 1); }
      else first = std::max(first, *bound);
      break;
    case arith::CmpIPredicate::ule:
      if (truth) last = std::min(last, *bound);
      else { if (*bound == UINT32_MAX) return std::nullopt; first = std::max(first, *bound + 1); }
      break;
    case arith::CmpIPredicate::eq:
      if (truth) { first = std::max(first, *bound); last = std::min(last, *bound); }
      break;
    default: break;
    }
    return first <= last ? std::optional<I32AddressSpan>{{first,last}} : std::nullopt;
  }

  // false XOR ((true XOR false) AND (0 - boolean)) is an exact i32 select.
  std::optional<I32AddressSpan> maskedSelect(Value value) {
    auto out = value.getDefiningOp<arith::XOrIOp>();
    if (!out) return std::nullopt;
    for (bool swapOuter : {false, true}) {
      Value no = swapOuter ? out.getRhs() : out.getLhs();
      auto chosen = (swapOuter ? out.getLhs() : out.getRhs()).getDefiningOp<arith::AndIOp>();
      if (!chosen) continue;
      for (bool swapMask : {false, true}) {
        auto mask = (swapMask ? chosen.getLhs() : chosen.getRhs()).getDefiningOp<arith::SubIOp>();
        auto delta = (swapMask ? chosen.getRhs() : chosen.getLhs()).getDefiningOp<arith::XOrIOp>();
        if (!mask || !delta || mask.getOverflowFlags() != arith::IntegerOverflowFlags::none || provenI32(mask.getLhs()) != 0u) continue;
        auto predicate = prove(mask.getRhs());
        if (!predicate || predicate->last > 1) continue;
        Value yes;
        if (delta.getLhs() == no) yes = delta.getRhs();
        else if (delta.getRhs() == no) yes = delta.getLhs();
        else continue;
        auto a = refine(yes, mask.getRhs(), true), b = refine(no, mask.getRhs(), false);
        if (a && b) return I32AddressSpan{std::min(a->first,b->first),std::max(a->last,b->last)};
      }
    }
    return std::nullopt;
  }

  std::optional<I32AddressSpan> derive(Value value) {
    if (auto constant = provenI32(value))
      return I32AddressSpan{*constant, *constant};
    if (!scalar) {
      if (auto shift = value.getDefiningOp<arith::ShLIOp>()) {
        // A bounded index scaled by at least one DMA granule is aligned for
        // every value, not merely at the endpoints of its range. Keep the
        // ordinary scalar range proof separate: its unscaled interior values
        // need not be aligned addresses.
        auto bits = provenI32(shift.getRhs());
        if (!bits || *bits < 5 || *bits >= 32 ||
            shift.getOverflowFlags() != arith::IntegerOverflowFlags::none)
          return std::nullopt;
        auto function = use->getParentOfType<func::FuncOp>();
        auto index = SourceAddressProof(function, use, /*scalar=*/true)
                         .prove(shift.getLhs());
        if (!index || (uint64_t(index->last) << *bits) > UINT32_MAX)
          return std::nullopt;
        return I32AddressSpan{uint32_t(index->first << *bits),
                              uint32_t(index->last << *bits)};
      }
    }
    if (scalar) {
      if (value.getType().isInteger(1)) return I32AddressSpan{0,1};
      if (auto select = maskedSelect(value)) return select;
      if (auto ext = value.getDefiningOp<arith::ExtUIOp>()) return prove(ext.getIn());
      if (auto bit = value.getDefiningOp<arith::AndIOp>()) {
        if (auto mask = provenI32(bit.getRhs())) return I32AddressSpan{0,*mask};
        if (auto mask = provenI32(bit.getLhs())) return I32AddressSpan{0,*mask};
        auto a = prove(bit.getLhs()), b = prove(bit.getRhs());
        if (!a && !b) return std::nullopt;
        return I32AddressSpan{0, a && b ? std::min(a->last,b->last) : a ? a->last : b->last};
      }
      if (isa_and_nonnull<arith::OrIOp,arith::XOrIOp>(value.getDefiningOp())) {
        auto a = prove(value.getDefiningOp()->getOperand(0)), b = prove(value.getDefiningOp()->getOperand(1));
        if (!a || !b) return std::nullopt;
        uint32_t last = std::max(a->last,b->last);
        last |= last >> 1; last |= last >> 2; last |= last >> 4; last |= last >> 8; last |= last >> 16;
        return I32AddressSpan{0,last};
      }
      if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
        if (sub.getOverflowFlags() != arith::IntegerOverflowFlags::none) return std::nullopt;
        auto a = prove(sub.getLhs()), b = prove(sub.getRhs());
        if (!a || !b || a->first < b->last) return std::nullopt;
        return I32AddressSpan{a->first-b->last,a->last-b->first};
      }
      if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
        auto a = prove(mul.getLhs()), b = prove(mul.getRhs());
        if (!a || !b || uint64_t(a->last)*b->last > UINT32_MAX) return std::nullopt;
        return I32AddressSpan{a->first*b->first,a->last*b->last};
      }
      if (isa_and_nonnull<arith::ShLIOp,arith::ShRUIOp>(value.getDefiningOp())) {
        auto *op = value.getDefiningOp();
        auto shift = prove(op->getOperand(1));
        if (!shift || shift->last >= 32) return std::nullopt;
        auto a = prove(op->getOperand(0));
        if (isa<arith::ShRUIOp>(op)) {
          if (!a) a = I32AddressSpan{0,UINT32_MAX};
          return I32AddressSpan{a->first >> shift->last,a->last >> shift->first};
        }
        if (!a || uint64_t(a->last) << shift->last > UINT32_MAX) return std::nullopt;
        return I32AddressSpan{uint32_t(a->first << shift->first),uint32_t(a->last << shift->last)};
      }
    }
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
    auto start = provenI32(initial.getDestOperands()[c]);
    auto step = increment(repeat.getDestOperands()[c], counter);
    auto stride = increment(repeat.getDestOperands()[p], pointer);
    if (!start || !step || !*step || !stride || !*stride || (!scalar && *stride % 32))
      return std::nullopt;
    uint64_t trips = *count;
    if (!*start) {
      // Retain the canonical zero-based, unit-counter contract.
      if (*step != 1) return std::nullopt;
    } else {
      // A pointer may itself guard the loop. Its final increment must not
      // wrap: unlike a companion pointer, it is observed by the exit test.
      if (*start >= *count) return std::nullopt;
      uint64_t distance = uint64_t(*count) - *start;
      trips = (distance + *step - 1) / *step;
      if (uint64_t(*start) + trips * *step > UINT32_MAX)
        return std::nullopt;
    }
    auto base = prove(initial.getDestOperands()[p]);
    if (!base) return std::nullopt;
    uint64_t last = uint64_t(base->last) + (trips - 1) * *stride;
    if (last > UINT32_MAX) return std::nullopt;
    return I32AddressSpan{base->first, uint32_t(last)};
  }
};
} // namespace

std::optional<I32AddressSpan> mlir::atlas::provenI32AddressSpan(Value value,
                                                            Operation *use) {
  atlasProofInvocation(AtlasProofKind::SourceAddress);
  auto function = use->getParentOfType<func::FuncOp>();
  if (!function) {
    auto constant = provenI32(value);
    return constant ? std::optional<I32AddressSpan>{{*constant, *constant}} : std::nullopt;
  }
  return SourceAddressProof(function, use).prove(value);
}

std::optional<I32AddressSpan> mlir::atlas::provenI32UnsignedRange(Value value, Operation *use) {
  atlasProofInvocation(AtlasProofKind::SourceAddress);
  auto function = use->getParentOfType<func::FuncOp>();
  if (!function) {
    auto constant = provenI32(value);
    return constant ? std::optional<I32AddressSpan>{{*constant,*constant}} : std::nullopt;
  }
  return SourceAddressProof(function,use, /*scalar=*/true).prove(value);
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
  if (!valid(size) || !valid(base) || !atlasDMAMemorySpan(uint64_t(base.getInt()), uint64_t(size.getInt())) ||
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
