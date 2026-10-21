#include "Atlas/AtlasAddressAnalysis.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <iterator>
#include <map>

using namespace mlir;
using namespace mlir::atlas;

namespace {
// A union of initialized byte intervals, independent of the allocation solver.
class InitializedBytes {
  std::map<uint64_t, uint64_t> intervals;
public:
  void write(uint64_t first, uint64_t end) {
    auto next = intervals.lower_bound(first);
    if (next != intervals.begin() && std::prev(next)->second >= first)
      next = std::prev(next);
    while (next != intervals.end() && next->first <= end) {
      first = std::min(first, next->first);
      end = std::max(end, next->second);
      next = intervals.erase(next);
    }
    intervals.emplace(first, end);
  }
  bool covers(uint64_t first, uint64_t end) const {
    auto next = intervals.upper_bound(first);
    return next != intervals.begin() && std::prev(next)->second >= end;
  }
};
} // namespace

LogicalResult mlir::atlas::verifyAtlasOutputInitialization(Operation *operation,
                                                          uint64_t first,
                                                          uint64_t end) {
  auto function = dyn_cast<func::FuncOp>(operation);
  if (!function || function.isDeclaration()) return failure();
  llvm::DenseMap<Value, uint32_t> values;
  InitializedBytes initialized;
  auto get = [&](Value v) -> std::optional<uint32_t> {
    auto found = values.find(v);
    return found == values.end() ? std::nullopt : std::optional<uint32_t>(found->second);
  };
  auto enter = [&](Block *destination, ValueRange operands) {
    SmallVector<std::optional<uint32_t>> incoming;
    for (Value v : operands) incoming.push_back(get(v));
    for (auto [arg, value] : llvm::zip_equal(destination->getArguments(), incoming)) {
      if (value) values[arg] = *value;
      else values.erase(arg);
    }
    return destination;
  };
  // This is a work bound, not an inferred loop bound or a hardware limit.
  // Unknown runtime branches and exhausted proofs remain compile failures.
  unsigned remaining = 1000000;
  Block *block = &function.getBody().front();
  while (block) {
    Block *next = nullptr;
    for (Operation &op : *block) {
      if (!remaining--)
        return op.emitOpError("output initialization proof exceeded its scalar work limit");
      // Revisited loop definitions must not retain an earlier iteration's fact.
      for (Value result : op.getResults()) values.erase(result);
      if (auto c = dyn_cast<arith::ConstantOp>(op)) {
        auto value = dyn_cast<IntegerAttr>(c.getValue());
        if (value && (c.getType().isInteger(32) || c.getType().isInteger(1)))
          values[c.getResult()] = value.getValue().getZExtValue();
      } else if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
        auto a = get(op.getOperand(0)), b = get(op.getOperand(1));
        if (a && b) {
          uint32_t value = isa<arith::AddIOp>(op) ? *a + *b :
                           isa<arith::SubIOp>(op) ? *a - *b : *a * *b;
          values[op.getResult(0)] = value;
        }
      } else if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
        auto a = get(cmp.getLhs()), b = get(cmp.getRhs());
        if (a && b) {
          bool result = false;
          switch (cmp.getPredicate()) {
          case arith::CmpIPredicate::eq: result = *a == *b; break;
          case arith::CmpIPredicate::ne: result = *a != *b; break;
          case arith::CmpIPredicate::ult: result = *a < *b; break;
          case arith::CmpIPredicate::ule: result = *a <= *b; break;
          case arith::CmpIPredicate::ugt: result = *a > *b; break;
          case arith::CmpIPredicate::uge: result = *a >= *b; break;
          case arith::CmpIPredicate::slt: result = int32_t(*a) < int32_t(*b); break;
          case arith::CmpIPredicate::sle: result = int32_t(*a) <= int32_t(*b); break;
          case arith::CmpIPredicate::sgt: result = int32_t(*a) > int32_t(*b); break;
          case arith::CmpIPredicate::sge: result = int32_t(*a) >= int32_t(*b); break;
          }
          values[cmp.getResult()] = result;
        }
      } else if (isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op,
                     VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op)) {
        bool load = isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op>(op);
        auto address = get(op.getOperand(load ? 1 : 2));
        auto bytes = get(op.getOperand(load ? 2 : 3));
        if (!address || !bytes || !*bytes || uint64_t(*address) + *bytes > (1ULL << 32))
          return op.emitOpError("output initialization needs a proven scalar DMA address and size");
        uint64_t start = *address, stop = start + *bytes;
        if (!load) {
          if (start < first || stop > end)
            return op.emitOpError("DMA store exceeds the declared output buffer");
          initialized.write(start, stop);
        } else if (start < end && first < stop &&
                   !initialized.covers(std::max(start, first), std::min(stop, end))) {
          return op.emitOpError("DMA read overlaps output bytes before they are initialized");
        }
      } else if (auto branch = dyn_cast<cf::BranchOp>(op)) {
        next = enter(branch.getDest(), branch.getDestOperands());
      } else if (auto branch = dyn_cast<cf::CondBranchOp>(op)) {
        auto condition = get(branch.getCondition());
        if (!condition)
          return op.emitOpError("output initialization cannot prove a runtime branch");
        next = *condition ? enter(branch.getTrueDest(), branch.getTrueDestOperands()) :
                            enter(branch.getFalseDest(), branch.getFalseDestOperands());
      } else if (isa<func::ReturnOp>(op)) {
        return success();
      }
    }
    if (!next) return block->getParentOp()->emitOpError("output initialization needs a supported CFG terminator");
    block = next;
  }
  return failure();
}
