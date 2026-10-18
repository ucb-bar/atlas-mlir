#include "Atlas/AtlasSpilling.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;

bool mlir::atlas::isAtlasSpillStore(Operation *op) {
  return isa<VirtualSpillBF16Op, VirtualSpillFP8Op>(op);
}

bool mlir::atlas::isAtlasSpillLoad(Operation *op) {
  return isa<VirtualReloadBF16Op, VirtualReloadFP8Op>(op);
}

unsigned mlir::atlas::atlasSpillWords(Operation *store) {
  return isa<VirtualSpillBF16Op>(store) ? 512 : 256;
}

LogicalResult mlir::atlas::verifyAtlasSpillOperation(Operation *op) {
  if (isAtlasSpillStore(op)) {
    auto slot = op->getAttrOfType<IntegerAttr>("slot_word");
    auto fixed = VirtualAllocationPlan().fixed();
    unsigned words = atlasSpillWords(op);
    if (!slot || slot.getInt() < fixed.spillWord ||
        uint64_t(slot.getInt()) + words > uint64_t(fixed.spillWord) + fixed.spillWindowWords ||
        slot.getInt() % 256)
      return op->emitOpError("spill slot is unaligned or outside the reserved VMEM frame");
  } else {
    Operation *store = op->getOperand(1).getDefiningOp();
    bool bf16 = isa<VirtualReloadBF16Op>(op);
    if (!store || !isAtlasSpillStore(store) ||
        bf16 != isa<VirtualSpillBF16Op>(store) ||
        store->getBlock() != op->getBlock())
      return op->emitOpError("reload needs a matching spill writer in the same block");
  }
  return success();
}

namespace {
struct Slot {
  Operation *store;
  unsigned begin, end, words;
  uint32_t word;
};

SmallVector<Slot> slotsOf(func::FuncOp function) {
  llvm::DenseMap<Operation *, unsigned> order;
  unsigned next = 0;
  for (Operation &op : function.front())
    order[&op] = next++;
  SmallVector<Slot> slots;
  for (Operation &op : function.front()) {
    if (!isAtlasSpillStore(&op))
      continue;
    unsigned end = order.lookup(&op);
    for (Operation *user : op.getResult(1).getUsers())
      end = std::max(end, order.lookup(user));
    auto word = op.getAttrOfType<IntegerAttr>("slot_word");
    slots.push_back({&op, order.lookup(&op), end, atlasSpillWords(&op),
                     word ? uint32_t(word.getInt()) : 0});
  }
  return slots;
}

bool liveTogether(const Slot &a, const Slot &b) {
  return a.begin <= b.end && b.begin <= a.end;
}

bool overlap(const Slot &a, const Slot &b) {
  return uint64_t(a.word) < uint64_t(b.word) + b.words &&
         uint64_t(b.word) < uint64_t(a.word) + a.words;
}

// Earliest aligned placement in the compiler-owned frame. Unlike textual
// register reuse, a slot remains occupied through its last reload.
LogicalResult placeSlots(func::FuncOp function, unsigned limit) {
  auto fixed = VirtualAllocationPlan().fixed();
  auto slots = slotsOf(function);
  Builder builder(function.getContext());
  for (unsigned i = 0; i < slots.size(); ++i) {
    Slot &slot = slots[i];
    uint64_t word = fixed.spillWord;
    while (word + slot.words <= uint64_t(fixed.spillWord) + limit) {
      slot.word = word;
      bool occupied = false;
      for (unsigned j = 0; j < i; ++j) {
        if (!liveTogether(slot, slots[j]) || !overlap(slot, slots[j]))
          continue;
        word = (uint64_t(slots[j].word) + slots[j].words + 255) / 256 * 256;
        occupied = true;
        break;
      }
      if (!occupied)
        break;
    }
    if (word + slot.words > uint64_t(fixed.spillWord) + limit)
      return slot.store->emitOpError("VMEM spill frame exhausted");
    slot.store->setAttr("slot_word", builder.getI32IntegerAttr(word));
  }
  return success();
}

Value stateBefore(Operation *where) {
  Value state;
  for (Operation &op : *where->getBlock()) {
    if (&op == where)
      break;
    for (Value value : op.getResults())
      if (isa<VirtualStateType>(value.getType()))
        state = value;
  }
  return state;
}

Operation *insertMemoryOp(Operation *before, StringRef name, Value value,
                          Type result, bool store) {
  OpBuilder builder(before);
  Value state = stateBefore(before);
  OperationState os(before->getLoc(), name);
  os.addOperands({state, value});
  os.addTypes({state.getType(), result});
  if (store)
    os.addAttribute("slot_word", builder.getI32IntegerAttr(0));
  Operation *op = builder.create(os);
  // The newly inserted operation consumes the old state; the following
  // operation must consume the new one. The verified input has one chain.
  state.replaceAllUsesExcept(op->getResult(0), op);
  return op;
}

void spill(Value value) {
  SmallVector<Operation *> consumers;
  for (Operation &op : *value.getParentBlock())
    if (llvm::is_contained(op.getOperands(), value))
      consumers.push_back(&op);
  bool bf16 = isa<VirtualBF16Type>(value.getType());
  Operation *store = insertMemoryOp(value.getDefiningOp()->getNextNode(),
      bf16 ? "atlas.virtual_spill_bf16" : "atlas.virtual_spill_fp8",
      value, VirtualSpillType::get(value.getContext()), true);
  for (Operation *consumer : consumers) {
    Operation *reload = insertMemoryOp(consumer,
        bf16 ? "atlas.virtual_reload_bf16" : "atlas.virtual_reload_fp8",
        store->getResult(1), value.getType(), false);
    for (OpOperand &operand : consumer->getOpOperands())
      if (operand.get() == value)
        operand.set(reload->getResult(1));
  }
}

struct SpillAtlasVirtualPass
    : PassWrapper<SpillAtlasVirtualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpillAtlasVirtualPass)
  SpillAtlasVirtualPass() = default;
  SpillAtlasVirtualPass(const SpillAtlasVirtualPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "spill-atlas-virtual"; }
  StringRef getDescription() const final {
    return "Split straight-line FP8/BF16 register lifetimes through a checked VMEM spill frame";
  }
  Option<unsigned> maxSpills{*this, "max-spills", llvm::cl::init(256),
      llvm::cl::desc("Bound pressure-driven spill rewrites")};
  Option<unsigned> frameWords{*this, "vmem-words", llvm::cl::init(0),
      llvm::cl::desc("Spill-frame word budget; zero uses the selected reserved frame")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyAtlasVirtualModule(module)))
      return signalPassFailure();
    auto functions = module.getOps<func::FuncOp>();
    if (!llvm::hasSingleElement(functions)) {
      module.emitError("spilling requires one virtual function");
      return signalPassFailure();
    }
    auto function = *functions.begin();
    auto fixed = VirtualAllocationPlan().fixed();
    unsigned limit = frameWords ? unsigned(frameWords) : fixed.spillWindowWords;
    if (limit > fixed.spillWindowWords || limit % 256) {
      function.emitError("spill budget must be aligned and fit the reserved VMEM frame");
      return signalPassFailure();
    }
    if (llvm::hasSingleElement(function.getBody()) &&
        failed(placeSlots(function, limit)))
      return signalPassFailure();
    llvm::DenseSet<Value> saved;
    unsigned count = 0;
    for (;;) {
      VirtualAllocationPlan plan;
      LogicalResult fits = failure();
      {
        // Allocation failure is feedback, not an error if a spill recovers it.
        ScopedDiagnosticHandler feedback(&getContext(), [](Diagnostic &) { return success(); });
        fits = plan.allocate(function);
      }
      if (succeeded(fits)) {
        if (failed(plan.verify()) || failed(verifyAtlasSpillSlots(function)))
          return signalPassFailure();
        Builder b(&getContext());
        unsigned peak = 0, stores = 0, loads = 0;
        if (llvm::hasSingleElement(function.getBody())) {
          for (auto slot : slotsOf(function)) {
            peak = std::max(peak, slot.word + slot.words - fixed.spillWord);
            ++stores;
            loads += std::distance(slot.store->getResult(1).use_begin(),
                                   slot.store->getResult(1).use_end());
          }
        }
        module->setAttr("atlas.spill_report", b.getDictionaryAttr({
            b.getNamedAttr("stores", b.getI32IntegerAttr(stores)),
            b.getNamedAttr("reloads", b.getI32IntegerAttr(loads)),
            b.getNamedAttr("frame_words_used", b.getI32IntegerAttr(peak)),
            b.getNamedAttr("frame_words_available", b.getI32IntegerAttr(limit))}));
        return;
      }
      if (!llvm::hasSingleElement(function.getBody())) {
        function.emitError("pressure-driven spilling across CFG edges is not supported yet");
        return signalPassFailure();
      }
      if (!plan.registerPressure(RegisterKind::Scalar).used) {
        function.emitError("scalar register spilling is not supported");
        return signalPassFailure();
      }
      bool bf16 = !plan.registerPressure(RegisterKind::BF16).used;
      bool fp8 = !plan.registerPressure(RegisterKind::FP8).used;
      if ((!bf16 && !fp8) || count == maxSpills) {
        function.emitError(!bf16 && !fp8
            ? "allocation failure is not recoverable by tensor spilling"
            : "spill rewrite work limit reached");
        return signalPassFailure();
      }
      llvm::DenseMap<Operation *, unsigned> order;
      unsigned index = 0;
      for (Operation &op : function.front())
        order[&op] = index++;
      Value victim;
      unsigned longest = 0;
      for (Operation &op : function.front()) {
        if (isAtlasSpillLoad(&op) || isAtlasSpillStore(&op))
          continue;
        for (Value value : op.getResults()) {
          if (saved.contains(value) ||
              (!(bf16 && isa<VirtualBF16Type>(value.getType())) &&
               !(fp8 && isa<VirtualFP8Type>(value.getType()))))
            continue;
          unsigned last = order.lookup(&op);
          for (Operation *user : value.getUsers())
            last = std::max(last, order.lookup(user));
          unsigned span = last - order.lookup(&op);
          if (span > longest) {
            longest = span;
            victim = value;
          }
        }
      }
      if (!victim || longest < 2) {
        function.emitError("insufficient non-spillable working registers");
        return signalPassFailure();
      }
      saved.insert(victim);
      spill(victim);
      ++count;
      if (failed(placeSlots(function, limit)))
        return signalPassFailure();
    }
  }
};
} // namespace

LogicalResult mlir::atlas::verifyAtlasSpillSlots(func::FuncOp function) {
  bool present = false;
  function.walk([&](Operation *op) {
    present |= isAtlasSpillStore(op) || isAtlasSpillLoad(op);
  });
  if (!present)
    return success();
  if (!llvm::hasSingleElement(function.getBody()))
    return function.emitOpError("spill-slot lifetimes currently require a single block");
  auto slots = slotsOf(function);
  for (unsigned i = 0; i < slots.size(); ++i) {
    const Slot &slot = slots[i];
    if (failed(verifyAtlasSpillOperation(slot.store)))
      return failure();
    for (Operation *user : slot.store->getResult(1).getUsers())
      if (!isAtlasSpillLoad(user) || failed(verifyAtlasSpillOperation(user)))
        return slot.store->emitOpError("spill identity has an invalid consumer");
    for (unsigned j = 0; j < i; ++j)
      if (liveTogether(slot, slots[j]) && overlap(slot, slots[j]))
        return slot.store->emitOpError("simultaneously live spill slots overlap");
  }
  return success();
}

void mlir::atlas::registerSpillAtlasVirtualPass() {
  PassRegistration<SpillAtlasVirtualPass>();
}
