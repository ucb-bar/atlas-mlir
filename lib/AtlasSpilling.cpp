#include "Atlas/AtlasSearchPolicy.h"
#include "Atlas/AtlasSpilling.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTarget.h"
#include "Atlas/AtlasVirtualAllocation.h"
#include "Atlas/AtlasVirtualVerification.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <algorithm>
#include <deque>

using namespace mlir;
using namespace mlir::atlas;

bool mlir::atlas::isAtlasSpillStore(Operation *op) {
  return isa<VirtualSpillBF16Op, VirtualSpillFP8Op>(op);
}

bool mlir::atlas::isAtlasSpillLoad(Operation *op) {
  return isa<VirtualReloadBF16Op, VirtualReloadFP8Op>(op);
}

unsigned mlir::atlas::atlasSpillWords(Operation *store) {
  if (auto slot = dyn_cast<VirtualSlotOp>(store))
    return slot.getWords();
  return isa<VirtualSpillBF16Op>(store) ? kAtlasBF16TileWords : kAtlasRegisterWords;
}

StringRef mlir::atlas::atlasSlotKind(Type type) {
  if (isa<VirtualBF16Type>(type)) return "bf16";
  if (isa<VirtualFP8Type>(type)) return "fp8";
  if (type.isInteger(1)) return "i1";
  if (type.isInteger(32)) return "i32";
  return "";
}

bool mlir::atlas::isAtlasSlotAccess(Operation *op) {
  return isa<VirtualSaveOp, VirtualRestoreOp, VirtualSlotCopyOp>(op);
}

Value mlir::atlas::atlasAccessedSlot(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<VirtualSaveOp, VirtualRestoreOp>([](auto x) { return x.getSlot(); })
      .Case([](VirtualSlotCopyOp copy) { return copy.getSrc(); })
      .Default([](Operation *) { return Value(); });
}

Value mlir::atlas::atlasSlotValue(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case([](VirtualSaveOp save) { return save.getSrc(); })
      .Case([](VirtualRestoreOp restore) { return restore.getValue(); })
      .Default([](Operation *) { return Value(); });
}

Value mlir::atlas::atlasSpillTile(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<VirtualSpillBF16Op, VirtualSpillFP8Op>([](auto x) { return x.getSrc(); })
      .Case<VirtualReloadBF16Op, VirtualReloadFP8Op>([](auto x) { return x.getValue(); })
      .Default([](Operation *) { return Value(); });
}

Operation *mlir::atlas::createAtlasSpillStore(OpBuilder &builder, Location loc,
                                              Value state, Value tile,
                                              uint32_t slotWord) {
  Type saved = VirtualSpillType::get(builder.getContext());
  if (isa<VirtualBF16Type>(tile.getType()))
    return VirtualSpillBF16Op::create(builder, loc, state.getType(), saved, state,
                                      tile, slotWord);
  return VirtualSpillFP8Op::create(builder, loc, state.getType(), saved, state,
                                   tile, slotWord);
}

Operation *mlir::atlas::createAtlasReload(OpBuilder &builder, Location loc,
                                          Value state, Value saved, Type type) {
  if (isa<VirtualBF16Type>(type))
    return VirtualReloadBF16Op::create(builder, loc, state.getType(), type, state,
                                       saved);
  return VirtualReloadFP8Op::create(builder, loc, state.getType(), type, state,
                                    saved);
}

Value mlir::atlas::atlasSpillSaved(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<VirtualSpillBF16Op, VirtualSpillFP8Op, VirtualReloadBF16Op,
            VirtualReloadFP8Op>([](auto x) { return x.getSaved(); })
      .Default([](Operation *) { return Value(); });
}

bool mlir::atlas::isAtlasMemoryControl(Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  auto function = arg ? dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp()) : func::FuncOp();
  return function && arg.getOwner() == &function.front() && function->hasAttr("atlas.memory_controls");
}

LogicalResult mlir::atlas::verifyAtlasSlotOperation(Operation *op) {
  if (auto slot = dyn_cast<VirtualSlotOp>(op)) {
    StringRef kind = slot.getKind();
    int64_t words = slot.getWordsAttr().getInt();
    auto fixed = VirtualAllocationPlan().fixed();
    if ((kind == "bf16" ? words != kAtlasBF16TileWords : kind == "fp8" ? words != kAtlasRegisterWords :
         kind == "i1" || kind == "i32" ? words != 1 :
         kind != "raw" || words <= 0 || words % kAtlasRegisterWords) ||
        words > fixed.spillWindowWords ||
        (slot.getPurpose() != "spill" && slot.getPurpose() != "edge" &&
         slot.getPurpose() != "scratch"))
      return op->emitOpError("invalid VMEM slot kind, extent or purpose");
    int64_t word = slot.getSlotWordAttr().getInt();
    unsigned alignment = words == 1 ? 1 : kAtlasRegisterWords;
    if (word != -1 && (word < fixed.spillWord || word % alignment ||
        uint64_t(word) + words > uint64_t(fixed.spillWord) + fixed.spillWindowWords))
      return op->emitOpError("VMEM slot is unaligned or outside the temporary pool");
    if (slot.getPinned() && word == -1)
      return op->emitOpError("pinned VMEM slot needs an address");
    auto function = op->getParentOfType<func::FuncOp>();
    if (!function || op->getBlock() != &function.front())
      return op->emitOpError("VMEM slots must be declared in the entry block");
    return success();
  }
  auto slot = atlasAccessedSlot(op).getDefiningOp<VirtualSlotOp>();
  if (!slot || slot->getParentOp() != op->getParentOp())
    return op->emitOpError("VMEM access needs a slot in the same function");
  if (auto copy = dyn_cast<VirtualSlotCopyOp>(op)) {
    auto destination = copy.getDst().getDefiningOp<VirtualSlotOp>();
    if (!destination || destination == slot ||
        destination->getParentOp() != op->getParentOp() ||
        destination.getKind() != slot.getKind() || destination.getWords() != slot.getWords())
      return op->emitOpError("slot copy needs distinct slots with matching formats and extents");
  } else {
    if (atlasSlotKind(atlasSlotValue(op).getType()) != slot.getKind())
      return op->emitOpError("VMEM access type differs from its slot format");
  }
  return success();
}

LogicalResult mlir::atlas::verifyAtlasSpillOperation(Operation *op) {
  if (isAtlasSpillStore(op)) {
    auto slot = op->getAttrOfType<IntegerAttr>("slot_word");
    auto fixed = VirtualAllocationPlan().fixed();
    unsigned words = atlasSpillWords(op);
    if (!slot || slot.getInt() < fixed.spillWord ||
        uint64_t(slot.getInt()) + words > uint64_t(fixed.spillWord) + fixed.spillWindowWords ||
        slot.getInt() % kAtlasRegisterWords)
      return op->emitOpError("spill slot is unaligned or outside the reserved VMEM frame");
  } else {
    Operation *store = atlasSpillSaved(op).getDefiningOp();
    bool bf16 = isa<VirtualReloadBF16Op>(op);
    if (!store || !isAtlasSpillStore(store) ||
        bf16 != isa<VirtualSpillBF16Op>(store) ||
        store->getParentOp() != op->getParentOp())
      return op->emitOpError("reload needs a matching spill writer in the same function");
  }
  return success();
}

namespace {
struct Slot {
  Operation *store;
  unsigned begin, end, words;
  uint32_t word;
  unsigned alignment = kAtlasRegisterWords;
  bool pinned = false;
};

SmallVector<Slot> slotsOf(func::FuncOp function) {
  llvm::DenseMap<Operation *, unsigned> order;
  unsigned next = 0;
  for (Block &block : function.getBody())
    for (Operation &op : block)
      order[&op] = next++;
  SmallVector<Slot> slots;
  for (Block &block : function.getBody()) {
    for (Operation &op : block) {
      if (!isAtlasSpillStore(&op) && !isa<VirtualSlotOp>(op))
        continue;
      unsigned end = order.lookup(&op);
      Value handle = isa<VirtualSlotOp>(op) ? op.getResult(0) : op.getResult(1);
      for (Operation *user : handle.getUsers())
        end = std::max(end, order.lookup(user));
      auto word = op.getAttrOfType<IntegerAttr>("slot_word");
      // A textual interval is not a CFG lifetime. Keep distinct slots for the
      // whole invocation, including backedges and mutually exclusive paths.
      unsigned begin = order.lookup(&op);
      if (!llvm::hasSingleElement(function.getBody()) || isa<VirtualSlotOp>(op)) {
        begin = 0;
        end = next;
      }
      slots.push_back({&op, begin, end, atlasSpillWords(&op),
                       word ? uint32_t(word.getInt()) : 0});
      if (auto slot = dyn_cast<VirtualSlotOp>(op)) {
        slots.back().alignment = slot.getWords() == 1 ? 1 : kAtlasRegisterWords;
        slots.back().pinned = slot.getPinned();
      }
    }
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
  llvm::stable_sort(slots, [](const Slot &a, const Slot &b) { return a.pinned > b.pinned; });
  Builder builder(function.getContext());
  for (unsigned i = 0; i < slots.size(); ++i) {
    Slot &slot = slots[i];
    uint64_t word = slot.pinned ? slot.word : fixed.spillWord;
    while (word + slot.words <= uint64_t(fixed.spillWord) + limit) {
      slot.word = word;
      bool occupied = false;
      for (unsigned j = 0; j < i; ++j) {
        if (!liveTogether(slot, slots[j]) || !overlap(slot, slots[j]))
          continue;
        if (slot.pinned)
          return slot.store->emitOpError("pinned VMEM allocations overlap");
        word = (uint64_t(slots[j].word) + slots[j].words + slot.alignment - 1) /
               slot.alignment * slot.alignment;
        occupied = true;
        break;
      }
      if (!occupied)
        break;
    }
    if (word + slot.words > uint64_t(fixed.spillWord) + limit)
      return slot.store->emitOpError("VMEM spill frame exhausted (shared temporary pool)");
    slot.store->setAttr("slot_word", builder.getI32IntegerAttr(word));
  }
  return success();
}

Value stateBefore(Operation *where) {
  Value state;
  for (BlockArgument argument : where->getBlock()->getArguments())
    if (isa<VirtualStateType>(argument.getType()))
      state = argument;
  for (Operation &op : *where->getBlock()) {
    if (&op == where)
      break;
    for (Value value : op.getResults())
      if (isa<VirtualStateType>(value.getType()))
        state = value;
  }
  return state;
}

// Inserts the op `build` makes from the state before `before`. The new op
// consumes the old state; the following operation must consume the new one.
// The verified input has one chain.
Operation *insertOnChain(Operation *before,
                         function_ref<Operation *(OpBuilder &, Location, Value)> build) {
  OpBuilder builder(before);
  Value state = stateBefore(before);
  Operation *op = build(builder, before->getLoc(), state);
  state.replaceUsesWithIf(op->getResult(0), [&](OpOperand &use) {
    return use.getOwner() != op && use.getOwner()->getBlock() == before->getBlock();
  });
  return op;
}

void saveToSlot(Operation *before, Value slot, Value value) {
  insertOnChain(before, [&](OpBuilder &builder, Location loc, Value state) {
    return VirtualSaveOp::create(builder, loc, state.getType(), state, slot, value)
        .getOperation();
  });
}

Value restoreFromSlot(Operation *before, Value slot, Type type) {
  return insertOnChain(before, [&](OpBuilder &builder, Location loc, Value state) {
           return VirtualRestoreOp::create(builder, loc, state.getType(), type,
                                           state, slot)
               .getOperation();
         })
      ->getResult(1);
}

void copySlot(Operation *before, Value src, Value dst) {
  insertOnChain(before, [&](OpBuilder &builder, Location loc, Value state) {
    return VirtualSlotCopyOp::create(builder, loc, state.getType(), state, src, dst)
        .getOperation();
  });
}

Value newSlot(func::FuncOp function, Type type, StringRef purpose) {
  OpBuilder builder(&*std::next(function.front().begin()));
  StringRef kind = atlasSlotKind(type);
  unsigned words = kind == "bf16" ? kAtlasBF16TileWords : kind == "fp8" ? kAtlasRegisterWords : 1;
  return VirtualSlotOp::create(builder, function.getLoc(),
                               VirtualSpillType::get(function.getContext()), kind,
                               words, /*slot_word=*/-1, purpose);
}

void spillEntryArguments(func::FuncOp function) {
  function->setAttr("atlas.memory_controls", UnitAttr::get(function.getContext()));
  Operation *anchor = &*std::next(function.front().begin());
  for (BlockArgument argument : function.getArguments()) {
    SmallVector<Operation *> consumers;
    for (Block &block : function.getBody())
      for (Operation &op : block)
        if (llvm::is_contained(op.getOperands(), argument)) consumers.push_back(&op);
    if (consumers.empty()) continue;
    Value slot = newSlot(function, argument.getType(), "spill");
    Operation *load = insertOnChain(anchor, [&](OpBuilder &builder, Location loc, Value state) {
      return VirtualControlOp::create(builder, loc, state.getType(), argument.getType(),
                                      state, argument.getArgNumber())
          .getOperation();
    });
    saveToSlot(anchor, slot, load->getResult(1));
    for (Operation *consumer : consumers)
      consumer->replaceUsesOfWith(argument,
                                  restoreFromSlot(consumer, slot, argument.getType()));
  }
}

// Demote the selected block arguments together. Order each edge's memory
// copies so no unread source is overwritten; break cycles with one temp slot.
unsigned spillBlockArguments(func::FuncOp function, RegisterKind kind) {
  SmallVector<BlockArgument> arguments;
  llvm::DenseMap<Value, Value> slots;
  llvm::StringMap<Value> temporaries;
  for (Block &block : function.getBody()) {
    if (&block == &function.front()) continue;
    for (BlockArgument arg : block.getArguments())
      if (registerKindOf(arg) == kind)
        arguments.push_back(arg);
  }
  if (arguments.empty()) return 0;
  for (BlockArgument arg : arguments) {
    slots[arg] = newSlot(function, arg.getType(), "spill");
  }
  SmallVector<Operation *> terminators;
  for (Block &block : function.getBody()) terminators.push_back(block.getTerminator());
  for (Operation *terminator : terminators) {
    auto edge = [&](Block *destination, ValueRange incoming) -> std::pair<Block *, SmallVector<Value>> {
      bool needed = llvm::any_of(destination->getArguments(), [&](Value a) { return slots.count(a); });
      if (!needed) return {destination, SmallVector<Value>(incoming)};
      auto *block = new Block();
      function.getBody().push_back(block);
      block->addArgument(VirtualStateType::get(function.getContext()), terminator->getLoc());
      OpBuilder b(block, block->end());
      SmallVector<Value> values(incoming);
      values[0] = block->getArgument(0);
      auto branch = b.create<cf::BranchOp>(terminator->getLoc(), destination, values);
      struct Move { Value dst, src, value; Type type; };
      SmallVector<Move> moves;
      for (BlockArgument arg : destination->getArguments()) {
        if (!slots.count(arg)) continue;
        Value value = incoming[arg.getArgNumber()], from = slots.lookup(value);
        if (from != slots[arg]) moves.push_back({slots[arg], from, value, arg.getType()});
      }
      while (!moves.empty()) {
        auto ready = llvm::find_if(moves, [&](const Move &move) {
          return llvm::none_of(moves, [&](const Move &other) { return other.src == move.dst; });
        });
        if (ready == moves.end()) {
          Value source = moves.front().src;
          StringRef kind = atlasSlotKind(moves.front().type);
          Value &temporary = temporaries[kind];
          if (!temporary) temporary = newSlot(function, moves.front().type, "edge");
          copySlot(branch, source, temporary);
          for (Move &move : moves) if (move.src == source) move.src = temporary;
          continue;
        }
        if (ready->src) copySlot(branch, ready->src, ready->dst);
        else saveToSlot(branch, ready->dst, ready->value);
        moves.erase(ready);
      }
      for (BlockArgument arg : llvm::reverse(destination->getArguments()))
        if (slots.count(arg)) branch.getDestOperandsMutable().erase(arg.getArgNumber());
      return {block, {incoming.front()}};
    };
    OpBuilder builder(terminator);
    if (auto branch = dyn_cast<cf::BranchOp>(terminator)) {
      auto replacement = edge(branch.getDest(), branch.getDestOperands());
      builder.create<cf::BranchOp>(branch.getLoc(), replacement.first, replacement.second);
      branch.erase();
    } else if (auto branch = dyn_cast<cf::CondBranchOp>(terminator)) {
      auto first = edge(branch.getTrueDest(), branch.getTrueDestOperands());
      auto second = edge(branch.getFalseDest(), branch.getFalseDestOperands());
      builder.create<cf::CondBranchOp>(branch.getLoc(), branch.getCondition(),
          first.first, first.second, second.first, second.second);
      branch.erase();
    }
  }
  for (BlockArgument arg : arguments) {
    SmallVector<Operation *> consumers;
    for (Block &block : function.getBody())
      for (Operation &op : block)
        if (llvm::is_contained(op.getOperands(), arg)) consumers.push_back(&op);
    for (Operation *consumer : consumers)
      consumer->replaceUsesOfWith(arg, restoreFromSlot(consumer, slots[arg], arg.getType()));
  }
  for (BlockArgument arg : llvm::reverse(arguments)) arg.getOwner()->eraseArgument(arg.getArgNumber());
  return arguments.size();
}

void spill(Value value, Operation *staged) {
  SmallVector<Operation *> consumers;
  auto function = value.getDefiningOp()->getParentOfType<func::FuncOp>();
  for (Block &block : function.getBody())
    for (Operation &op : block)
      if (llvm::is_contained(op.getOperands(), value))
        consumers.push_back(&op);
  // Proven address/size constants need no memory round trip. Recreate their
  // wrapping-i32 value at each use, retaining exact static DMA proofs.
  if (auto constant = provenI32(value)) {
    for (Operation *consumer : consumers) {
      OpBuilder builder(consumer);
      auto rematerialized = arith::ConstantIntOp::create(
          builder, consumer->getLoc(), int64_t(*constant), 32);
      rematerialized->setAttr("atlas.spill_rematerialized", builder.getUnitAttr());
      consumer->replaceUsesOfWith(value, rematerialized);
    }
    return;
  }
  if (registerKindOf(value) == RegisterKind::Scalar) {
    Value slot = newSlot(function, value.getType(), "spill");
    saveToSlot(value.getDefiningOp()->getNextNode(), slot, value);
    for (Operation *consumer : consumers)
      consumer->replaceUsesOfWith(value, restoreFromSlot(consumer, slot, value.getType()));
    return;
  }
  // A loaded tile stays in its staging window until `staged`, where its
  // registers begin: save it after its first use there, which reads it
  // directly, or before a later launch that takes its window, rather than
  // waiting for its transfer at the launch.
  Operation *saveBefore = value.getDefiningOp()->getNextNode();
  if (staged && !staged->hasTrait<OpTrait::IsTerminator>()) {
    saveBefore = staged;
    if (!consumers.empty() && consumers.front() == staged) {
      saveBefore = staged->getNextNode();
      consumers.erase(consumers.begin());
    }
  }
  Operation *store = insertOnChain(saveBefore, [&](OpBuilder &builder, Location loc, Value state) {
    return createAtlasSpillStore(builder, loc, state, value, /*slotWord=*/0);
  });
  for (Operation *consumer : consumers) {
    Operation *reload = insertOnChain(consumer, [&](OpBuilder &builder, Location loc, Value state) {
      return createAtlasReload(builder, loc, state, store->getResult(1), value.getType());
    });
    for (OpOperand &operand : consumer->getOpOperands())
      if (operand.get() == value)
        operand.set(reload->getResult(1));
  }
}

// Source transfer records use program order. Lay out dominating blocks before
// their users without changing edges, block arguments or operation order.
void orderBlocks(func::FuncOp function) {
  SmallVector<Block *> blocks;
  for (Block *block : llvm::ReversePostOrderTraversal<Block *>(&function.front()))
    blocks.push_back(block);
  for (Block *block : blocks)
    block->moveBefore(&function.getBody(), function.getBody().end());
}

struct SpillAtlasVirtualPass
    : PassWrapper<SpillAtlasVirtualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SpillAtlasVirtualPass)
  SpillAtlasVirtualPass() = default;
  SpillAtlasVirtualPass(const SpillAtlasVirtualPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "spill-atlas-virtual"; }
  StringRef getDescription() const final {
    return "Spill MXU state, tensor and scalar values through shared VMEM temporaries";
  }
  Option<unsigned> maxSpills{*this, "max-spills", llvm::cl::init(search_policy::kSpillRewriteLimit),
      llvm::cl::desc("Bound pressure-driven spill rewrites")};
  Option<unsigned> frameWords{*this, "vmem-words", llvm::cl::init(0),
      llvm::cl::desc("Temporary VMEM word budget; zero uses the selected pool")};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyAtlasVirtualModule(module, /*allowMXUPressure=*/true,
                                      /*allowMXUCFG=*/true)))
      return signalPassFailure();
    auto functions = module.getOps<func::FuncOp>();
    if (!llvm::hasSingleElement(functions)) {
      module.emitError("spilling requires one virtual function");
      return signalPassFailure();
    }
    auto function = *functions.begin();
    auto fixed = VirtualAllocationPlan().fixed();
    unsigned limit = frameWords ? unsigned(frameWords) : fixed.spillWindowWords;
    if (limit > fixed.spillWindowWords || limit % kAtlasRegisterWords) {
      function.emitError("spill budget must be aligned and fit the reserved VMEM frame");
      return signalPassFailure();
    }
    unsigned mxuSpills = 0;
    if (failed(spillAtlasMXUState(function, maxSpills, mxuSpills)))
      return signalPassFailure();
    if (!slotsOf(function).empty())
      orderBlocks(function);
    if (failed(placeSlots(function, limit)) ||
        failed(verifyAtlasVirtualModule(module)))
      return signalPassFailure();
    llvm::DenseSet<Value> saved;
    // FP8 has no register-to-register edge move in this selected lowering.
    // Transport its block arguments through checked raw slots instead.
    unsigned count = mxuSpills + spillBlockArguments(function, RegisterKind::FP8);
    if (count > maxSpills || failed(placeSlots(function, limit))) {
      function.emitError("FP8 edge spill work limit or VMEM budget exceeded");
      return signalPassFailure();
    }
    if (count) orderBlocks(function);
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
        unsigned peak = 0, stores = 0, loads = 0, copies = 0, rematerialized = 0;
        unsigned mxuSaves = 0, mxuRestores = 0;
        for (auto slot : slotsOf(function)) {
          peak = std::max(peak, slot.word + slot.words - fixed.spillWord);
        }
        function.walk([&](Operation *op) {
          stores += isAtlasSpillStore(op) || isa<VirtualSaveOp>(op);
          loads += isAtlasSpillLoad(op) || isa<VirtualRestoreOp>(op);
          copies += isa<VirtualSlotCopyOp>(op);
          rematerialized += op->hasAttr("atlas.spill_rematerialized");
          if (auto action = op->getAttrOfType<StringAttr>("atlas.mxu_spill")) {
            mxuSaves += action.getValue() == "save";
            mxuRestores += action.getValue() == "restore";
          }
        });
        module->setAttr("atlas.spill_report", b.getDictionaryAttr({
            b.getNamedAttr("mxu_acc_saves", b.getI32IntegerAttr(mxuSaves)),
            b.getNamedAttr("mxu_restores", b.getI32IntegerAttr(mxuRestores)),
            b.getNamedAttr("stores", b.getI32IntegerAttr(stores)),
            b.getNamedAttr("reloads", b.getI32IntegerAttr(loads)),
            b.getNamedAttr("copies", b.getI32IntegerAttr(copies)),
            b.getNamedAttr("rematerialized", b.getI32IntegerAttr(rematerialized)),
            b.getNamedAttr("frame_words_used", b.getI32IntegerAttr(peak)),
            b.getNamedAttr("frame_words_available", b.getI32IntegerAttr(limit))}));
        return;
      }
      bool scalar = !plan.registerPressure(RegisterKind::Scalar).used;
      bool bf16 = !plan.registerPressure(RegisterKind::BF16).used;
      bool fp8 = !plan.registerPressure(RegisterKind::FP8).used;
      if ((!bf16 && !fp8 && !scalar) || count >= maxSpills) {
        function.emitError(!bf16 && !fp8 && !scalar
            ? "allocation failure is not recoverable by tensor spilling"
            : "spill rewrite work limit reached");
        return signalPassFailure();
      }
      if (scalar && !function->hasAttr("atlas.memory_controls") &&
          function.getNumArguments() > registerCapacities(function)[2]) {
        if (function.getNumArguments() > 256 || count + function.getNumArguments() > maxSpills) {
          function.emitError("entry control buffer or spill work limit exceeded");
          return signalPassFailure();
        }
        spillEntryArguments(function);
        count += function.getNumArguments();
        if (failed(placeSlots(function, limit))) return signalPassFailure();
        continue;
      }
      unsigned demoted = spillBlockArguments(function, scalar ? RegisterKind::Scalar : RegisterKind::BF16);
      if (demoted) {
        count += demoted;
        if (count > maxSpills || failed(placeSlots(function, limit))) {
          function.emitError("spill rewrite work limit or VMEM budget exceeded");
          return signalPassFailure();
        }
        orderBlocks(function);
        continue;
      }
      if (!count)
        orderBlocks(function);
      llvm::DenseMap<Operation *, unsigned> order;
      unsigned index = 0;
      for (Block &block : function.getBody())
        for (Operation &op : block)
          order[&op] = index++;
      Value victim;
      unsigned longest = 0;
      for (Block &block : function.getBody()) {
        for (Operation &op : block) {
          if (isAtlasSpillLoad(&op) || isAtlasSpillStore(&op) || isAtlasSlotAccess(&op))
            continue;
          for (Value value : op.getResults()) {
            std::optional<RegisterKind> kind = registerKindOf(value);
            bool overflowed = kind == RegisterKind::BF16 ? bf16 : kind == RegisterKind::FP8 ? fp8 : scalar;
            if (saved.contains(value) || !kind || !overflowed)
              continue;
            // A loaded tile holds registers from where it is staged.
            unsigned first = order.lookup(&op);
            if (Operation *staged = plan.stagingPoint(value))
              first = order.lookup(staged);
            unsigned last = first;
            for (Operation *user : value.getUsers())
              last = std::max(last, order.lookup(user));
            unsigned span = last - first;
            if (span > longest) {
              longest = span;
              victim = value;
            }
          }
        }
      }
      if (!victim || longest < 2) {
        function.emitError("insufficient non-spillable working registers");
        return signalPassFailure();
      }
      saved.insert(victim);
      spill(victim, plan.stagingPoint(victim));
      ++count;
      if (failed(placeSlots(function, limit)))
        return signalPassFailure();
    }
  }
};
} // namespace

LogicalResult mlir::atlas::placeAtlasSpillSlots(func::FuncOp function, unsigned words) {
  return placeSlots(function, words);
}

LogicalResult mlir::atlas::verifyAtlasSpillSlots(func::FuncOp function) {
  bool present = false;
  function.walk([&](Operation *op) {
    present |= isAtlasSpillStore(op) || isAtlasSpillLoad(op) || isa<VirtualSlotOp>(op);
  });
  if (!present)
    return success();
  DominanceInfo dominance(function);
  auto slots = slotsOf(function);
  for (unsigned i = 0; i < slots.size(); ++i) {
    const Slot &slot = slots[i];
    bool mutableSlot = isa<VirtualSlotOp>(slot.store);
    if (failed(mutableSlot ? verifyAtlasSlotOperation(slot.store) : verifyAtlasSpillOperation(slot.store)))
      return failure();
    Value handle = slot.store->getResult(mutableSlot ? 0 : 1);
    for (Operation *user : handle.getUsers()) {
      if (mutableSlot ? !isAtlasSlotAccess(user) || failed(verifyAtlasSlotOperation(user)) :
          !isAtlasSpillLoad(user) || failed(verifyAtlasSpillOperation(user)))
        return slot.store->emitOpError("spill identity has an invalid consumer");
      if (!dominance.properlyDominates(slot.store, user))
        return user->emitOpError("reload must be dominated by its spill writer");
    }
    for (unsigned j = 0; j < i; ++j)
      if (slot.word != uint32_t(-1) && slots[j].word != uint32_t(-1) &&
          liveTogether(slot, slots[j]) && overlap(slot, slots[j]))
        return slot.store->emitOpError("simultaneously live spill slots overlap");
  }
  using Initialized = llvm::DenseSet<Value>;
  llvm::DenseMap<Block *, Initialized> entries;
  llvm::DenseSet<Block *> reached{&function.front()};
  std::deque<Block *> work{&function.front()};
  auto transfer = [&](Block *block, Initialized &state, bool diagnose) -> LogicalResult {
    for (Operation &op : *block) {
      if (!isAtlasSlotAccess(&op)) continue;
      if (failed(verifyAtlasSlotOperation(&op))) return failure();
      if (isa<VirtualRestoreOp, VirtualSlotCopyOp>(op) &&
          !state.contains(atlasAccessedSlot(&op)) && diagnose)
        return op.emitOpError("VMEM slot is not initialized on every reaching path");
      if (auto save = dyn_cast<VirtualSaveOp>(op)) state.insert(save.getSlot());
      if (auto copy = dyn_cast<VirtualSlotCopyOp>(op)) state.insert(copy.getDst());
    }
    return success();
  };
  while (!work.empty()) {
    Block *block = work.front(); work.pop_front();
    Initialized state = entries[block];
    if (failed(transfer(block, state, false))) return failure();
    for (Block *successor : block->getSuccessors()) {
      if (reached.insert(successor).second) {
        entries[successor] = state;
        work.push_back(successor);
      } else {
        auto before = entries[successor];
        for (Value slot : before)
          if (!state.contains(slot)) entries[successor].erase(slot);
        if (entries[successor].size() != before.size()) work.push_back(successor);
      }
    }
  }
  for (Block &block : function.getBody()) {
    Initialized state = entries[&block];
    if (failed(transfer(&block, state, true))) return failure();
  }
  return success();
}

void mlir::atlas::registerSpillAtlasVirtualPass() {
  registerSpillAtlasMXUStatePass();
  PassRegistration<SpillAtlasVirtualPass>();
}

std::unique_ptr<Pass> mlir::atlas::createSpillAtlasVirtualPass() {
  return std::make_unique<SpillAtlasVirtualPass>();
}
