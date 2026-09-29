#include "Atlas/AtlasToLLVM.h"
#include "Atlas/AtlasEncoding.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
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

    // A single side-effecting assembly block keeps branch targets and the
    // selected RTL delay-slot instruction adjacent during LLVM code generation.
    std::string assembly;
    llvm::raw_string_ostream asmStream(assembly);
    for (uint32_t word : words) {
      asmStream << ".word 0x" << llvm::format_hex_no_prefix(word, 8) << '\n';
    }
    asmStream.flush();

    for (Operation &op : llvm::make_early_inc_range(
             llvm::reverse(module.getBody()->getOperations())))
      op.erase();

    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    Location loc = module.getLoc();
    auto voidType = LLVM::LLVMVoidType::get(module.getContext());
    auto function = LLVM::LLVMFuncOp::create(
        builder,
        loc, "atlas_program", LLVM::LLVMFunctionType::get(voidType, {}));
    Block *entry = function.addEntryBlock(builder);
    builder.setInsertionPointToStart(entry);
    LLVM::InlineAsmOp::create(
        builder, loc, TypeRange{}, ValueRange{}, assembly, "~{memory}",
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        LLVM::TailCallKind::None, LLVM::AsmDialectAttr{}, ArrayAttr{});
    LLVM::ReturnOp::create(builder, loc, ValueRange{});
  }
};
} // namespace

void mlir::atlas::registerConvertAtlasToLLVMPass() {
  PassRegistration<ConvertAtlasToLLVMPass>();
}
