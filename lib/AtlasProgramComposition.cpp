#include "Atlas/AtlasProgramComposition.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasSpilling.h"
#include "Atlas/AtlasTypes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <limits>

using namespace mlir;

namespace {
struct ComposeAtlasProgramPass
    : PassWrapper<ComposeAtlasProgramPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComposeAtlasProgramPass)
  StringRef getArgument() const final { return "compose-atlas-program"; }
  StringRef getDescription() const final {
    return "Join explicitly ordered virtual functions into one program before allocation";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto entries = module->getAttrOfType<ArrayAttr>("atlas.program_entries");
    if (!entries || entries.empty()) {
      module.emitError("composition requires a nonempty atlas.program_entries array");
      return signalPassFailure();
    }
    SmallVector<func::FuncOp> functions;
    llvm::SmallPtrSet<Operation *, 8> originals;
    uint64_t outputBegin = std::numeric_limits<uint64_t>::max(), outputEnd = 0;
    uint64_t inputBegin = std::numeric_limits<uint64_t>::max();
    uint64_t codeBudget = std::numeric_limits<uint64_t>::max();
    SmallVector<std::pair<uint64_t, uint64_t>> outputRanges;
    NamedAttrList shared;
    auto boundary = [](StringRef name) {
      return name == "atlas.input_dram_base" || name == "atlas.output_dram_base" ||
             name == "atlas.output_dram_bytes" || name == "atlas.program_size_limit_bytes";
    };
    for (Attribute entry : entries) {
      auto name = dyn_cast<StringAttr>(entry);
      auto function = name ? module.lookupSymbol<func::FuncOp>(name.getValue()) : func::FuncOp();
      if (!function || function.isDeclaration() || function.getNumArguments() ||
          function.getFunctionType().getNumResults() != 1 ||
          !isa<atlas::VirtualStateType>(function.getFunctionType().getResult(0))) {
        module.emitError("each program entry must be a defined () -> !atlas.virtual_state function");
        return signalPassFailure();
      }
      unsigned starts = 0;
      bool invalid = false;
      function.walk([&](Operation *op) {
        StringRef kind = op->getName().getStringRef();
        if (kind == "atlas.virtual_start") {
          ++starts;
          invalid |= op->getBlock() != &function.getBody().front();
        }
        // Boundary indices depend on a function-local ABI. Composition requires
        // explicit DMA and cannot guess how indexed ports from different calls alias.
        invalid |= isa<func::CallOp>(op) || kind.starts_with("atlas.virtual_input_") ||
                   kind.starts_with("atlas.virtual_output_");
      });
      if (starts != 1 || invalid) {
        function.emitError("composition needs one entry start, explicit DMA and no calls");
        return signalPassFailure();
      }
      if (failed(atlas::verifyAtlasSpillSlots(function))) return signalPassFailure();
      auto base = function->getAttrOfType<IntegerAttr>("atlas.output_dram_base");
      auto bytes = function->getAttrOfType<IntegerAttr>("atlas.output_dram_bytes");
      auto input = function->getAttrOfType<IntegerAttr>("atlas.input_dram_base");
      auto isI64 = [](IntegerAttr attr) {
        return attr && attr.getType().isSignlessInteger(64);
      };
      if (!isI64(input) || !isI64(base) || !isI64(bytes) ||
          input.getInt() < 0 || uint64_t(input.getInt()) >= (uint64_t(1) << 32) ||
          base.getInt() < 0 || bytes.getInt() <= 0 ||
          uint64_t(base.getInt()) + uint64_t(bytes.getInt()) > (uint64_t(1) << 32)) {
        function.emitError("composition requires a bounded output DRAM range");
        return signalPassFailure();
      }
      outputBegin = std::min(outputBegin, uint64_t(base.getInt()));
      outputEnd = std::max(outputEnd, uint64_t(base.getInt()) + uint64_t(bytes.getInt()));
      outputRanges.emplace_back(base.getInt(), base.getInt() + bytes.getInt());
      inputBegin = std::min(inputBegin, uint64_t(input.getInt()));
      if (auto budget = function->getAttrOfType<IntegerAttr>("atlas.program_size_limit_bytes")) {
        if (!budget.getType().isSignlessInteger(64) || budget.getInt() <= 0) {
          function.emitError("composition requires a positive i64 program-size budget");
          return signalPassFailure();
        }
        codeBudget = std::min(codeBudget, uint64_t(budget.getInt()));
      }
      for (NamedAttribute attr : function->getAttrs()) {
        StringRef key = attr.getName().strref();
        if (!key.starts_with("atlas.") || boundary(key)) continue;
        if (functions.empty()) shared.set(attr.getName(), attr.getValue());
        else if (shared.get(attr.getName()) != attr.getValue()) {
          function.emitError("composition cannot discard differing Atlas function attributes");
          return signalPassFailure();
        }
      }
      for (NamedAttribute attr : shared)
        if (function->getAttr(attr.getName()) != attr.getValue()) {
          function.emitError("composition requires matching Atlas function attributes");
          return signalPassFailure();
        }
      functions.push_back(function);
      originals.insert(function);
    }
    // The current memory checker accepts one output interval. Do not enlarge
    // that permission to include holes between separately declared buffers.
    std::sort(outputRanges.begin(), outputRanges.end());
    uint64_t coveredEnd = outputRanges.front().second;
    for (auto [begin, end] : llvm::drop_begin(outputRanges)) {
      if (begin > coveredEnd) {
        module.emitError("composition requires contiguous output ranges; disjoint output permissions are unsupported");
        return signalPassFailure();
      }
      coveredEnd = std::max(coveredEnd, end);
    }
    for (auto function : module.getOps<func::FuncOp>())
      if (!originals.contains(function)) {
        function.emitError("composition must account for every function in the module");
        return signalPassFailure();
      }
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    auto state = atlas::VirtualStateType::get(module.getContext());
    auto joined = func::FuncOp::create(builder, module.getLoc(), "atlas_program",
                                     builder.getFunctionType({}, {state}));
    // Preserve output bounds used by the independent memory-effect checker.
    joined->setAttr("atlas.output_dram_base", builder.getI64IntegerAttr(outputBegin));
    joined->setAttr("atlas.output_dram_bytes", builder.getI64IntegerAttr(outputEnd - outputBegin));
    joined->setAttr("atlas.input_dram_base", builder.getI64IntegerAttr(inputBegin));
    for (NamedAttribute attr : shared) joined->setAttr(attr.getName(), attr.getValue());
    if (codeBudget != std::numeric_limits<uint64_t>::max())
      joined->setAttr("atlas.program_size_limit_bytes", builder.getI64IntegerAttr(codeBudget));
    Block *header = joined.addEntryBlock();
    SmallVector<Block *> starts;
    SmallVector<func::ReturnOp> returns;
    SmallVector<atlas::VirtualSlotOp> scratch;
    for (auto function : functions) {
      IRMapping mapping;
      function.getBody().cloneInto(&joined.getBody(), mapping);
      Block *entry = mapping.lookup(&function.getBody().front());
      Value incoming = entry->addArgument(state, function.getLoc());
      starts.push_back(entry);
      SmallVector<Operation *> remove;
      SmallVector<atlas::VirtualSlotOp> slots;
      for (Block &original : function.getBody()) {
        Block *block = mapping.lookup(&original);
        for (Operation &op : *block) {
          StringRef kind = op.getName().getStringRef();
          if (kind == "atlas.virtual_start") {
            op.getResult(0).replaceAllUsesWith(incoming);
            remove.push_back(&op);
          }
          if (auto slot = dyn_cast<atlas::VirtualSlotOp>(op)) slots.push_back(slot);
        }
      }
      for (Operation *op : remove) op->erase();
      // Scratch is private to each sequential fragment. Share matching slots
      // across fragments, keeping each fragment's simultaneous slots distinct.
      llvm::SmallPtrSet<Operation *, 8> used;
      for (auto slot : slots) {
        atlas::VirtualSlotOp available;
        bool share = slot.getPurpose() == "scratch" && !slot.getPinned() && slot.getSlotWord() == -1;
        if (share)
          for (auto candidate : scratch)
            if (!used.contains(candidate) && candidate.getKind() == slot.getKind() &&
                candidate.getWords() == slot.getWords()) { available = candidate; break; }
        if (available) {
          slot.getSlot().replaceAllUsesWith(available.getSlot());
          slot.erase();
        } else {
          slot->moveBefore(header, header->end());
          available = slot;
          if (share) scratch.push_back(slot);
        }
        used.insert(available);
      }
      // Each cloned function may have several return blocks. Record all of them
      // with the next entry rather than assuming the return is textually last.
      for (Block &original : function.getBody()) {
        Block *block = mapping.lookup(&original);
        if (auto ret = dyn_cast<func::ReturnOp>(block->getTerminator())) returns.push_back(ret);
      }
    }
    Block *finish = new Block();
    joined.getBody().push_back(finish);
    Value finalState = finish->addArgument(state, module.getLoc());
    builder.setInsertionPointToEnd(finish);
    func::ReturnOp::create(builder, module.getLoc(), ValueRange{finalState});
    unsigned returnIndex = 0;
    for (unsigned i = 0; i < functions.size(); ++i) {
      unsigned count = 0;
      functions[i].walk([&](func::ReturnOp) { ++count; });
      for (unsigned j = 0; j < count; ++j) {
        auto ret = returns[returnIndex++];
        builder.setInsertionPoint(ret);
        cf::BranchOp::create(builder, ret.getLoc(), i + 1 < starts.size() ? starts[i + 1] : finish,
                             ret.getOperands());
        ret.erase();
      }
    }
    builder.setInsertionPointToStart(header);
    OperationState start(module.getLoc(), "atlas.virtual_start");
    start.addTypes(state);
    Operation *begun = builder.create(start);
    builder.setInsertionPointToEnd(header);
    cf::BranchOp::create(builder, module.getLoc(), starts.front(), begun->getResults());
    for (Operation *function : originals) function->erase();
    module->removeAttr("atlas.program_entries");
  }
};
} // namespace

void mlir::atlas::registerComposeAtlasProgramPass() {
  PassRegistration<ComposeAtlasProgramPass>();
}
