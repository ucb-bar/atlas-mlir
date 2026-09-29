#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::atlas::AtlasDialect, mlir::LLVM::LLVMDialect>();
  mlir::atlas::registerConvertAtlasToLLVMPass();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Atlas dialect verifier\n", registry));
}
