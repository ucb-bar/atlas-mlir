#include "Atlas/AtlasStreamVerification.h"
#include "Atlas/AtlasEncoding.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct VerifyAtlasMachineStreamPass
    : PassWrapper<VerifyAtlasMachineStreamPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyAtlasMachineStreamPass)

  StringRef getArgument() const final { return "verify-atlas-machine-stream"; }
  StringRef getDescription() const final {
    return "Check the exact Atlas state chain, encodings, delay slots, and in-block LLVM targets before scheduling or lowering";
  }

  void runOnOperation() override {
    llvm::SmallVector<uint32_t> words;
    if (failed(mlir::atlas::verifyAtlasArtifact(getOperation(),
                                                 /*llvmBlock=*/true, words)))
      signalPassFailure();
  }
};
} // namespace

void mlir::atlas::registerVerifyAtlasMachineStreamPass() {
  PassRegistration<VerifyAtlasMachineStreamPass>();
}
