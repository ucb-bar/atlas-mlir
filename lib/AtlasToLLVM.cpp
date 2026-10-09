#include "Atlas/AtlasToLLVM.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <string>
#include <vector>

using namespace mlir;

namespace {
struct StagedInstruction {
  std::string name;
  DictionaryAttr fields;
  uint32_t word;
  Location location;
};

void emitWordBlock(ModuleOp module, ArrayRef<uint32_t> words) {
  std::string assembly;
  llvm::raw_string_ostream asmStream(assembly);
  for (uint32_t word : words)
    asmStream << ".word 0x" << llvm::format_hex_no_prefix(word, 8) << '\n';
  asmStream.flush();

  for (Operation &op : llvm::make_early_inc_range(
           llvm::reverse(module.getBody()->getOperations())))
    op.erase();

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());
  Location loc = module.getLoc();
  auto voidType = LLVM::LLVMVoidType::get(module.getContext());
  auto function = LLVM::LLVMFuncOp::create(
      builder, loc, "atlas_program", LLVM::LLVMFunctionType::get(voidType, {}));
  Block *entry = function.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  LLVM::InlineAsmOp::create(
      builder, loc, TypeRange{}, ValueRange{}, assembly, "~{memory}",
      /*has_side_effects=*/true, /*is_align_stack=*/false,
      LLVM::TailCallKind::None, LLVM::AsmDialectAttr{}, ArrayAttr{});
  LLVM::ReturnOp::create(builder, loc, ValueRange{});
}

struct ConvertAtlasToLLVMPass
    : PassWrapper<ConvertAtlasToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAtlasToLLVMPass)

  StringRef getArgument() const final { return "convert-atlas-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower a flat selected-RTL Atlas instruction stream to one ordered LLVM inline-assembly block";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::SmallVector<uint32_t> words;
    if (failed(mlir::atlas::collectAtlasWords(module, words, true))) {
      signalPassFailure();
      return;
    }

    // Keep this direct entrypoint for existing users. The structured LLVM
    // handoff below offers an inspectable stage before the same final block.
    emitWordBlock(module, words);
  }
};

struct ConvertAtlasToLLVMCallsPass
    : PassWrapper<ConvertAtlasToLLVMCallsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAtlasToLLVMCallsPass)

  StringRef getArgument() const final { return "convert-atlas-to-llvm-calls"; }
  StringRef getDescription() const final {
    return "Preserve each checked Atlas instruction as a structured LLVM-dialect call";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::SmallVector<uint32_t> words;
    if (failed(mlir::atlas::collectAtlasWords(module, words, true))) {
      signalPassFailure();
      return;
    }
    std::vector<StagedInstruction> records;
    unsigned index = 0;
    for (Operation &op : module.getBody()->getOperations()) {
      if (isa<mlir::atlas::StartOp>(op))
        continue;
      records.push_back({op.getName().getStringRef().str(),
                         DictionaryAttr::get(module.getContext(), op.getAttrs()),
                         words[index++], op.getLoc()});
    }
    for (Operation &op : llvm::make_early_inc_range(
             llvm::reverse(module.getBody()->getOperations())))
      op.erase();

    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    auto voidType = LLVM::LLVMVoidType::get(module.getContext());
    std::set<std::string> declared;
    for (const StagedInstruction &record : records) {
      std::string symbol = "atlas_emit_" + record.name.substr(6);
      if (declared.insert(symbol).second)
        LLVM::LLVMFuncOp::create(
            builder, record.location, symbol,
            LLVM::LLVMFunctionType::get(voidType, {}));
    }
    auto function = LLVM::LLVMFuncOp::create(
        builder, module.getLoc(), "atlas_program",
        LLVM::LLVMFunctionType::get(voidType, {}));
    Block *entry = function.addEntryBlock(builder);
    builder.setInsertionPointToStart(entry);
    for (auto [wordIndex, record] : llvm::enumerate(records)) {
      std::string symbol = "atlas_emit_" + record.name.substr(6);
      auto declaration = module.lookupSymbol<LLVM::LLVMFuncOp>(symbol);
      auto call = LLVM::CallOp::create(builder, record.location,
                                      declaration, ValueRange{});
      call->setAttr("atlas.source_op", builder.getStringAttr(record.name));
      call->setAttr("atlas.fields", record.fields);
      call->setAttr("atlas.word_index", builder.getI32IntegerAttr(wordIndex));
      call->setAttr("atlas.word", builder.getI32IntegerAttr(record.word));
      call->setAttr("atlas.effect_scope", builder.getStringAttr(
          "conservative_physical_state_read_write"));
      call->setAttr("atlas.availability", builder.getStringAttr("unknown"));
    }
    LLVM::ReturnOp::create(builder, module.getLoc(), ValueRange{});
    module->setAttr("atlas.structured_handoff", builder.getUnitAttr());
  }
};

struct FinalizeAtlasLLVMCallsPass
    : PassWrapper<FinalizeAtlasLLVMCallsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeAtlasLLVMCallsPass)

  StringRef getArgument() const final { return "finalize-atlas-llvm-calls"; }
  StringRef getDescription() const final {
    return "Recheck structured Atlas LLVM calls and produce one ordered executable word block";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<LLVM::LLVMDialect, mlir::atlas::AtlasDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module->hasAttr("atlas.structured_handoff")) {
      module.emitError("expected a structured Atlas LLVM handoff");
      signalPassFailure();
      return;
    }
    auto function = module.lookupSymbol<LLVM::LLVMFuncOp>("atlas_program");
    if (!function || function.isExternal() ||
        !llvm::hasSingleElement(function.getBody())) {
      module.emitError("expected one atlas_program LLVM block");
      signalPassFailure();
      return;
    }
    for (Operation &operation : module.getBody()->getOperations()) {
      if (&operation == function.getOperation())
        continue;
      auto declaration = dyn_cast<LLVM::LLVMFuncOp>(operation);
      if (!declaration || !declaration.isExternal() ||
          !declaration.getSymName().starts_with("atlas_emit_")) {
        operation.emitError("unexpected operation in structured Atlas LLVM module");
        signalPassFailure();
        return;
      }
    }

    OpBuilder builder(module.getContext());
    ModuleOp reconstructed = ModuleOp::create(module.getLoc());
    for (StringRef name : {"atlas.generated_from_virtual", "atlas.virtual_dma_contract", "atlas.virtual_mxu_contract", "atlas.virtual_tile_contract"})
      if (Attribute value = module->getAttr(name))
        reconstructed->setAttr(name, value);
    builder.setInsertionPointToStart(reconstructed.getBody());
    OperationState startState(module.getLoc(), "atlas.start");
    auto stateType = mlir::atlas::StateType::get(module.getContext());
    startState.addTypes(stateType);
    Value previous = builder.create(startState)->getResult(0);

    std::vector<uint32_t> stagedWords;
    unsigned index = 0;
    Block &entry = function.getBody().front();
    for (Operation &operation : entry.getOperations()) {
      if (isa<LLVM::ReturnOp>(operation)) {
        if (&operation != &entry.back()) {
          operation.emitError("LLVM return must end the Atlas stream");
          signalPassFailure();
          return;
        }
        continue;
      }
      auto call = dyn_cast<LLVM::CallOp>(operation);
      auto opName = operation.getAttrOfType<StringAttr>("atlas.source_op");
      auto fields = operation.getAttrOfType<DictionaryAttr>("atlas.fields");
      auto wordIndex = operation.getAttrOfType<IntegerAttr>("atlas.word_index");
      auto encoded = operation.getAttrOfType<IntegerAttr>("atlas.word");
      auto callee = operation.getAttrOfType<FlatSymbolRefAttr>("callee");
      if (!call || !opName || !fields || !wordIndex || !encoded || !callee ||
          wordIndex.getInt() != index ||
          !opName.getValue().starts_with("atlas.") ||
          callee.getValue() !=
              ("atlas_emit_" + opName.getValue().drop_front(6)).str() ||
          call.getNumOperands() != 0 || call.getNumResults() != 0 ||
          !module.lookupSymbol<LLVM::LLVMFuncOp>(callee.getValue())) {
        operation.emitError("invalid or reordered structured Atlas LLVM call");
        signalPassFailure();
        return;
      }
      stagedWords.push_back(static_cast<uint32_t>(encoded.getValue().getZExtValue()));
      OperationState machineState(operation.getLoc(), opName.getValue());
      machineState.addOperands(previous);
      machineState.addTypes(stateType);
      machineState.addAttributes(fields.getValue());
      builder.setInsertionPointToEnd(reconstructed.getBody());
      previous = builder.create(machineState)->getResult(0);
      ++index;
    }
    llvm::SmallVector<uint32_t> checked;
    if (stagedWords.empty() ||
        failed(mlir::atlas::collectAtlasWords(reconstructed, checked, true)) ||
        !llvm::equal(stagedWords, checked)) {
      module.emitError("structured Atlas LLVM calls disagree with checked encodings");
      signalPassFailure();
      return;
    }
    reconstructed.erase();
    module->removeAttr("atlas.structured_handoff");
    emitWordBlock(module, checked);
  }
};
} // namespace

void mlir::atlas::registerConvertAtlasToLLVMPass() {
  PassRegistration<ConvertAtlasToLLVMPass>();
}

void mlir::atlas::registerConvertAtlasToLLVMCallsPass() {
  PassRegistration<ConvertAtlasToLLVMCallsPass>();
}

void mlir::atlas::registerFinalizeAtlasLLVMCallsPass() {
  PassRegistration<FinalizeAtlasLLVMCallsPass>();
}
