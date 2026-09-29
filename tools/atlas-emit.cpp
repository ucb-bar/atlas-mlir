#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasEncoding.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::atlas;

int main(int argc, char **argv) {
  if (argc != 2) {
    llvm::errs() << "usage: atlas-emit <flat-mlir-module>\n";
    return 2;
  }
  DialectRegistry registry;
  registry.insert<AtlasDialect>();
  MLIRContext context(registry);
  auto module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module || failed(verify(*module))) return 1;

  llvm::SmallVector<uint32_t> words;
  if (failed(collectAtlasWords(*module, words, false))) return 1;
  for (uint32_t word : words)
    llvm::outs() << llvm::format_hex_no_prefix(word, 8) << '\n';
  return 0;
}
