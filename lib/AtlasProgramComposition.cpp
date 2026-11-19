#include "Atlas/AtlasProgramComposition.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasTarget.h"
#include "Atlas/AtlasTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <limits>

using namespace mlir;

namespace {
// Only pointer addition and CFG forwarding may carry a parameter to DMA.
// In particular, lengths, loop guards and arithmetic data remain literals.
bool usedOnlyAsDMAAddress(Value value) {
  SmallVector<Value> work{value};
  llvm::DenseSet<Value> visited;
  bool addressUse = false;
  while (!work.empty()) {
    Value current = work.pop_back_val();
    if (!visited.insert(current).second) continue;
    for (OpOperand &use : current.getUses()) {
      Operation *op = use.getOwner();
      unsigned position = use.getOperandNumber();
      bool load = isa<atlas::VirtualDMALoadFP8Op, atlas::VirtualDMALoadBF16Op>(op);
      if (load || isa<atlas::VirtualDMAStoreFP8Op, atlas::VirtualDMAStoreBF16Op>(op)) {
        if (position != (load ? 1u : 2u)) return false;
        addressUse = true;
      } else if (auto add = dyn_cast<arith::AddIOp>(op)) {
        work.push_back(add.getResult());
      } else if (auto branch = dyn_cast<cf::BranchOp>(op)) {
        work.push_back(branch.getDest()->getArgument(position));
      } else if (auto branch = dyn_cast<cf::CondBranchOp>(op)) {
        if (!position) return false;
        --position;
        unsigned trueCount = branch.getTrueDestOperands().size();
        work.push_back(position < trueCount ? branch.getTrueDest()->getArgument(position)
                                           : branch.getFalseDest()->getArgument(position-trueCount));
      } else return false;
    }
  }
  return addressUse;
}

// Abstract only plain address literals. Operation names, types, operand uses,
// branch targets, numeric attributes and all other constants remain identical.
SmallVector<arith::ConstantOp> addressParameters(func::FuncOp function) {
  SmallVector<arith::ConstantOp> constants;
  function.walk([&](arith::ConstantOp op) {
    auto integer = dyn_cast<IntegerAttr>(op.getValue());
    if (op.getType().isSignlessInteger(32) && integer &&
        op->getDiscardableAttrDictionary().empty() &&
        integer.getValue().getZExtValue() % 32 == 0 &&
        atlas::atlasDMAMemorySpan(integer.getValue().getZExtValue(), 32) &&
        usedOnlyAsDMAAddress(op.getResult()))
      constants.push_back(op);
  });
  return constants;
}

// A counted dispatcher visits the authored entries in order. Equivalent stage
// bodies share one CFG; differing addresses become entry block arguments. This
// shares code only: no aliasing, layout, or verification obligation changes.
bool shareProgramBodies(ArrayRef<func::FuncOp> functions, func::FuncOp joined) {
  SmallVector<OwningOpRef<func::FuncOp>> normalized;
  SmallVector<SmallVector<arith::ConstantOp>> parameters;
  SmallVector<unsigned> groupOf, representatives;
  for (auto function : functions) {
    parameters.push_back(addressParameters(function));
    OwningOpRef<func::FuncOp> copy(function.clone());
    for (auto constant : addressParameters(*copy))
      constant.setValueAttr(IntegerAttr::get(constant.getType(), 0));
    unsigned group = 0;
    for (; group < representatives.size(); ++group)
      if (OperationEquivalence::isRegionEquivalentTo(
              &copy->getBody(), &normalized[representatives[group]]->getBody(),
              OperationEquivalence::IgnoreLocations))
        break;
    if (group == representatives.size()) representatives.push_back(groupOf.size());
    groupOf.push_back(group);
    normalized.push_back(std::move(copy));
  }
  if (representatives.size() == functions.size()) return false;

  SmallVector<SmallVector<unsigned>> varying(representatives.size());
  for (unsigned group = 0; group < representatives.size(); ++group) {
    unsigned first = representatives[group];
    for (unsigned p = 0; p < parameters[first].size(); ++p)
      for (unsigned stage = 0; stage < functions.size(); ++stage)
        if (groupOf[stage] == group &&
            parameters[stage][p].getValue() != parameters[first][p].getValue()) {
          varying[group].push_back(p);
          break;
        }
  }

  OpBuilder builder(joined.getContext());
  Location loc = joined.getLoc();
  Type state = atlas::VirtualStateType::get(joined.getContext());
  Block *entry = joined.addEntryBlock();
  auto block = [&]() {
    auto *result = new Block();
    joined.getBody().push_back(result);
    return result;
  };
  Block *dispatch = block(), *choose = block(), *advance = block(), *finish = block();
  choose->addArgument(state, loc);
  Value token = dispatch->addArgument(state, loc);
  Value index = dispatch->addArgument(builder.getI32Type(), loc);
  Value completed = advance->addArgument(state, loc);
  Value finalState = finish->addArgument(state, loc);
  SmallVector<Block *> bodies;
  for (unsigned group = 0; group < representatives.size(); ++group) {
    unsigned first = representatives[group];
    auto function = functions[first];
    IRMapping mapping;
    function.getBody().cloneInto(&joined.getBody(), mapping);
    Block *body = mapping.lookup(&function.getBody().front());
    bodies.push_back(body);
    Value incoming = body->addArgument(state, loc);
    for (unsigned p : varying[group]) {
      Value argument = body->addArgument(builder.getI32Type(), loc);
      Value original = mapping.lookup(parameters[first][p].getResult());
      original.replaceAllUsesWith(argument);
      original.getDefiningOp()->erase();
    }
    for (Block &original : function.getBody()) {
      Block *cloned = mapping.lookup(&original);
      for (Operation &op : llvm::make_early_inc_range(*cloned)) {
        if (isa<atlas::VirtualStartOp>(op)) {
          op.getResult(0).replaceAllUsesWith(incoming);
          op.erase();
        } else if (isa<atlas::VirtualSlotOp>(op)) {
          op.moveBefore(entry, entry->end());
        } else if (auto ret = dyn_cast<func::ReturnOp>(op)) {
          builder.setInsertionPoint(ret);
          cf::BranchOp::create(builder, ret.getLoc(), advance, ret.getOperands());
          ret.erase();
        }
      }
    }
  }
  builder.setInsertionPointToStart(entry);
  Value begun = atlas::VirtualStartOp::create(builder, loc, state);
  builder.setInsertionPointToEnd(entry);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  cf::BranchOp::create(builder, loc, dispatch, ValueRange{begun, zero});

  builder.setInsertionPointToEnd(dispatch);
  Value count = arith::ConstantIntOp::create(builder, loc, functions.size(), 32);
  Value more = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult, index, count);
  cf::CondBranchOp::create(builder, loc, more, choose, ValueRange{token}, finish, ValueRange{token});
  for (unsigned stage = 0; stage < functions.size(); ++stage) {
    builder.setInsertionPointToEnd(choose);
    Block *setup = block();
    setup->addArgument(state, loc);
    Value selectedState = choose->getArgument(0);
    if (stage + 1 < functions.size()) {
      Block *next = block();
      next->addArgument(state, loc);
      Value ordinal = arith::ConstantIntOp::create(builder, loc, stage, 32);
      Value selected = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, index, ordinal);
      cf::CondBranchOp::create(builder, loc, selected, setup, ValueRange{selectedState}, next, ValueRange{selectedState});
      choose = next;
    } else {
      cf::BranchOp::create(builder, loc, setup, ValueRange{selectedState});
    }
    builder.setInsertionPointToEnd(setup);
    SmallVector<Value> arguments{setup->getArgument(0)};
    for (unsigned p : varying[groupOf[stage]])
      arguments.push_back(builder.clone(*parameters[stage][p])->getResult(0));
    cf::BranchOp::create(builder, loc, bodies[groupOf[stage]], arguments);
  }
  builder.setInsertionPointToEnd(advance);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
  Value next = arith::AddIOp::create(builder, loc, index, one);
  cf::BranchOp::create(builder, loc, dispatch, ValueRange{completed, next});
  builder.setInsertionPointToEnd(finish);
  func::ReturnOp::create(builder, loc, ValueRange{finalState});
  return true;
}

struct ComposeAtlasProgramPass
    : PassWrapper<ComposeAtlasProgramPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComposeAtlasProgramPass)
  ComposeAtlasProgramPass() = default;
  ComposeAtlasProgramPass(const ComposeAtlasProgramPass &other) : PassWrapper(other) {}
  Option<bool> shareBodies{*this, "share-bodies",
      llvm::cl::desc("Share equivalent stage CFGs with explicit address parameters"),
      llvm::cl::init(false)};
  StringRef getArgument() const final { return "compose-atlas-program"; }
  StringRef getDescription() const final {
    return "Join explicitly ordered virtual functions into one program before allocation";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto entries = module->getAttrOfType<ArrayAttr>("atlas.program_entries");
    if (!entries || entries.empty()) {
      module.emitError("composition requires a nonempty atlas.program_entries array");
      return signalPassFailure();
    }
    if (entries.size() > std::numeric_limits<uint32_t>::max()) {
      module.emitError("program entry count exceeds the i32 dispatch index");
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
      return name == "atlas.input_dram_base" || name == "atlas.input_dram_bytes" ||
             name == "atlas.output_dram_base" ||
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
      // A member's input frame may be another member's output, so this is not an
      // invocation-wide immutable interval; DMA operands keep their own proofs.
      if (auto raw = function->getAttr("atlas.input_dram_bytes")) {
        auto extent = dyn_cast<IntegerAttr>(raw);
        if (!isI64(extent) || extent.getInt() <= 0 ||
            uint64_t(extent.getInt()) > (uint64_t(1) << 32) - uint64_t(input.getInt())) {
          function.emitError("composition requires a bounded optional input DRAM extent");
          return signalPassFailure();
        }
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
    if (shareBodies && shareProgramBodies(functions, joined)) {
      for (Operation *function : originals) function->erase();
      module->removeAttr("atlas.program_entries");
      return;
    }
    Block *header = joined.addEntryBlock();
    SmallVector<Block *> starts;
    SmallVector<func::ReturnOp> returns;
    for (auto function : functions) {
      IRMapping mapping;
      function.getBody().cloneInto(&joined.getBody(), mapping);
      Block *entry = mapping.lookup(&function.getBody().front());
      Value incoming = entry->addArgument(state, function.getLoc());
      starts.push_back(entry);
      SmallVector<Operation *> remove;
      for (Block &original : function.getBody()) {
        Block *block = mapping.lookup(&original);
        for (Operation &op : llvm::make_early_inc_range(*block)) {
          StringRef kind = op.getName().getStringRef();
          if (kind == "atlas.virtual_start") {
            op.getResult(0).replaceAllUsesWith(incoming);
            remove.push_back(&op);
          }
          // Operand-free slot handles must dominate every cloned access, as
          // the slot verifier requires, so they move to the entry block.
          if (kind == "atlas.virtual_slot")
            op.moveBefore(header, header->end());
        }
      }
      for (Operation *op : remove) op->erase();
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
