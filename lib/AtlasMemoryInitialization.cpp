#include "Atlas/AtlasProofWork.h"
#include "Atlas/AtlasProofLimits.h"
#include "Atlas/AtlasAddressAnalysis.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <map>
#include <deque>
#include <vector>

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
  bool intersect(const InitializedBytes &other) {
    std::map<uint64_t, uint64_t> common;
    for (auto [first, end] : intervals)
      for (auto [rightFirst, rightEnd] : other.intervals) {
        auto begin = std::max(first, rightFirst), stop = std::min(end, rightEnd);
        if (begin < stop) common[begin] = stop;
      }
    bool changed = common != intervals;
    intervals = std::move(common);
    return changed;
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
  atlasProofInvocation(AtlasProofKind::SourceInitialization);
  auto function = dyn_cast<func::FuncOp>(operation);
  if (!function || function.isDeclaration()) return failure();
  Liveness liveness(function);
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
  unsigned remaining = kAtlasScalarProofWorkLimit;
  // Identify scalar values this interpreter can ever resolve. Runtime tensor
  // contents stay unknown; replaying their arithmetic cannot establish a DMA
  // address or choose a control edge. This is a may-be-known fixed point, so
  // loop arguments remain eligible if any incoming edge supplies a known value.
  llvm::DenseSet<Value> mayBeKnown;
  SmallVector<Value> knownWork;
  llvm::DenseMap<Value, SmallVector<Value>> forwarded;
  llvm::DenseMap<Operation *, unsigned> unknownOperands;
  auto discover = [&](Value value) {
    if (mayBeKnown.insert(value).second) knownWork.push_back(value);
  };
  auto forward = [&](Block *destination, ValueRange operands) {
    for (auto [argument, value] : llvm::zip_equal(destination->getArguments(), operands))
      forwarded[value].push_back(argument);
  };
  for (Block &body : function.getBody()) {
    for (Operation &op : body) {
      if (!remaining--) return op.emitOpError("output initialization proof exceeded its scalar work limit");
      atlasProofRecord(AtlasProofKind::SourceInitialization);
      if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        if (isa<IntegerAttr>(constant.getValue()) &&
            (constant.getType().isInteger(1) || constant.getType().isInteger(32)))
          discover(constant.getResult());
      } else if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp,
                     arith::AndIOp, arith::OrIOp, arith::XOrIOp,
                     arith::ShLIOp, arith::ShRUIOp,
                     arith::ExtUIOp, arith::CmpIOp>(op)) {
        unknownOperands[&op] = op.getNumOperands();
      } else if (auto branch = dyn_cast<cf::BranchOp>(op)) {
        forward(branch.getDest(), branch.getDestOperands());
      } else if (auto branch = dyn_cast<cf::CondBranchOp>(op)) {
        forward(branch.getTrueDest(), branch.getTrueDestOperands());
        forward(branch.getFalseDest(), branch.getFalseDestOperands());
      }
    }
  }
  while (!knownWork.empty()) {
    Value value = knownWork.pop_back_val();
    for (Value argument : forwarded.lookup(value)) {
      if (!remaining--) return operation->emitOpError("output initialization proof exceeded its scalar work limit");
      atlasProofRecord(AtlasProofKind::SourceInitialization);
      discover(argument);
    }
    for (OpOperand &use : value.getUses()) {
      if (!remaining--) return operation->emitOpError("output initialization proof exceeded its scalar work limit");
      atlasProofRecord(AtlasProofKind::SourceInitialization);
      auto found = unknownOperands.find(use.getOwner());
      if (found != unknownOperands.end() && --found->second == 0)
        for (Value result : use.getOwner()->getResults()) discover(result);
    }
  }
  // Slice once, before replaying loops. Tensor arithmetic and scalar values
  // used only to form tensor data cannot change initialization. Keep every
  // control-flow operand (including loop-carried values) so state-cycle and
  // termination checks retain the same scalar semantics.
  llvm::DenseSet<Value> needed;
  SmallVector<Value> worklist;
  auto require = [&](Value value) {
    if ((value.getType().isInteger(1) || value.getType().isInteger(32)) &&
        mayBeKnown.contains(value) && needed.insert(value).second)
      worklist.push_back(value);
  };
  for (Block &body : function.getBody()) {
    for (Operation &op : body) {
      if (!remaining--) return op.emitOpError("output initialization proof exceeded its scalar work limit");
      atlasProofRecord(AtlasProofKind::SourceInitialization);
      bool load = isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op>(op);
      bool store = isa<VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op);
      if (load || store) {
        require(op.getOperand(load ? 1 : 2));
        require(op.getOperand(load ? 2 : 3));
      }
      if (op.hasTrait<OpTrait::IsTerminator>())
        for (Value operand : op.getOperands()) require(operand);
    }
  }
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!remaining--) return operation->emitOpError("output initialization proof exceeded its scalar work limit");
    atlasProofRecord(AtlasProofKind::SourceInitialization);
    if (Operation *definition = value.getDefiningOp())
      for (Value operand : definition->getOperands()) require(operand);
  }
  llvm::DenseMap<Block *, SmallVector<Operation *>> proofOperations;
  for (Block &body : function.getBody()) {
    auto &selected = proofOperations[&body];
    for (Operation &op : body)
      if (op.hasTrait<OpTrait::IsTerminator>() ||
          isa<VirtualDMALoadFP8Op, VirtualDMALoadBF16Op,
              VirtualDMAStoreFP8Op, VirtualDMAStoreBF16Op>(op) ||
          llvm::any_of(op.getResults(), [&](Value value) { return needed.contains(value); }))
        selected.push_back(&op);
  }
  Block *block = &function.getBody().front();
  struct Pending {
    Block *block;
    llvm::DenseMap<Value, uint32_t> values;
    InitializedBytes initialized;
    std::optional<unsigned> predecessor;
  };
  std::deque<Pending> pending;
  std::map<std::vector<uint64_t>, InitializedBytes> entries;
  std::map<std::vector<uint64_t>, unsigned> stateIDs;
  AtlasProofStateGraph graph;
  pending.push_back({block, values, initialized, std::nullopt});
  while (!pending.empty()) {
    auto current = std::move(pending.front()); pending.pop_front();
    block = current.block; values = std::move(current.values); initialized = std::move(current.initialized);
    // Only facts that can be used after this block entry distinguish scalar
    // states. Dead branch-local facts otherwise duplicate every later proof
    // path, even after its initialized-byte sets have reached the same join.
    const auto &liveIn = liveness.getLiveIn(block);
    SmallVector<Value> dead;
    for (auto [value, bits] : values) {
      bool argument = isa<BlockArgument>(value) && cast<BlockArgument>(value).getOwner() == block;
      if (!argument && !liveIn.contains(value)) dead.push_back(value);
    }
    for (Value value : dead) values.erase(value);
    std::vector<std::pair<uint64_t, uint32_t>> scalar;
    for (auto [value, bits] : values)
      scalar.emplace_back(reinterpret_cast<uintptr_t>(value.getAsOpaquePointer()), bits);
    std::sort(scalar.begin(), scalar.end());
    std::vector<uint64_t> key{reinterpret_cast<uintptr_t>(block)};
    for (auto [value, bits] : scalar) { key.push_back(value); key.push_back(bits); }
    auto [state, newState] = stateIDs.emplace(key, stateIDs.size());
    if (newState) graph.addNode();
    unsigned stateID = state->second;
    if (current.predecessor) {
      auto edge = graph.addEdge(*current.predecessor, stateID, remaining);
      if (edge == AtlasProofStateGraph::EdgeResult::Cycle)
        return operation->emitOpError("output initialization proof encountered a nonterminating scalar-state cycle");
      if (edge == AtlasProofStateGraph::EdgeResult::Exhausted)
        return operation->emitOpError("output initialization proof exceeded its scalar work limit");
    }
    auto [entry, fresh] = entries.emplace(key, initialized);
    if (!fresh) {
      if (!entry->second.intersect(initialized)) continue;
      initialized = entry->second;
    }
    Block *next = nullptr;
    for (Operation *selected : proofOperations[block]) {
      Operation &op = *selected;
      atlasProofTransfer(AtlasProofKind::SourceInitialization);
      if (!remaining--)
        return op.emitOpError("output initialization proof exceeded its scalar work limit");
      // Revisited loop definitions must not retain an earlier iteration's fact.
      for (Value result : op.getResults()) values.erase(result);
      if (auto c = dyn_cast<arith::ConstantOp>(op)) {
        auto value = dyn_cast<IntegerAttr>(c.getValue());
        if (value && (c.getType().isInteger(32) || c.getType().isInteger(1)))
          values[c.getResult()] = value.getValue().getZExtValue();
      } else if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp,
                     arith::AndIOp, arith::OrIOp, arith::XOrIOp,
                     arith::ShLIOp, arith::ShRUIOp>(op)) {
        auto a = get(op.getOperand(0)), b = get(op.getOperand(1));
        if (a && b) {
          if (isa<arith::AddIOp>(op)) values[op.getResult(0)] = *a + *b;
          if (isa<arith::SubIOp>(op)) values[op.getResult(0)] = *a - *b;
          if (isa<arith::MulIOp>(op)) values[op.getResult(0)] = *a * *b;
          if (isa<arith::AndIOp>(op)) values[op.getResult(0)] = *a & *b;
          if (isa<arith::OrIOp>(op)) values[op.getResult(0)] = *a | *b;
          if (isa<arith::XOrIOp>(op)) values[op.getResult(0)] = *a ^ *b;
          if (isa<arith::ShLIOp>(op) && *b < 32) values[op.getResult(0)] = *a << *b;
          if (isa<arith::ShRUIOp>(op) && *b < 32) values[op.getResult(0)] = *a >> *b;
        }
      } else if (auto extend = dyn_cast<arith::ExtUIOp>(op)) {
        if (auto value = get(extend.getIn())) values[extend.getResult()] = *value & 1;
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
        if (!condition) {
          auto saved = values;
          Block *alternate = enter(branch.getFalseDest(), branch.getFalseDestOperands());
          pending.push_back({alternate, values, initialized, stateID});
          values = std::move(saved);
          next = enter(branch.getTrueDest(), branch.getTrueDestOperands());
        } else next = *condition ? enter(branch.getTrueDest(), branch.getTrueDestOperands()) :
                                   enter(branch.getFalseDest(), branch.getFalseDestOperands());
      } else if (isa<func::ReturnOp>(op)) {
        next = nullptr;
        break;
      }
    }
    if (next) pending.push_back({next, values, initialized, stateID});
    else if (!isa<func::ReturnOp>(block->getTerminator()))
      return block->getParentOp()->emitOpError("output initialization needs a supported CFG terminator");
  }
  return success();
}
