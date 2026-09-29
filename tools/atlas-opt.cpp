#include "Atlas/AtlasDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::atlas::AtlasDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Atlas dialect verifier\n", registry));
}
