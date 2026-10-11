#include "VerificationTestSupport.h"

using namespace atlas_test;

int main(int argc, char **argv) {
  static const std::pair<llvm::StringRef, Suite> suites[] = {
      {"buffer-contract", runBufferContract}, {"cfg-contract", runCFGContract},
      {"dma-allocation", runDMAAllocation}, {"dma-helper", runDMAHelper},
      {"mxu-allocation", runMXUAllocation}, {"mxu-contract", runMXUContract},
      {"source-memory-contract", runSourceMemoryContract}, {"tile-contract", runTileContract},
      {"timing-provider", runTimingProvider}};
  const auto *selected = llvm::find_if(suites, [&](const auto &suite) { return argc == 2 && suite.first == argv[1]; });
  if (selected == std::end(suites)) {
    llvm::errs() << "usage: " << argv[0] << " <suite>; suites:";
    for (const auto &suite : suites)
      llvm::errs() << ' ' << suite.first;
    llvm::errs() << '\n';
    return 2;
  }
  DialectRegistry registry;
  registry.insert<atlas::AtlasDialect, arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect>();
  atlas::registerLowerAtlasVirtualToMachinePass();
  atlas::registerInsertAtlasDelaysPass();
  MLIRContext context(registry);
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &diagnostic) {
    llvm::raw_string_ostream stream(diagnostics);
    stream << diagnostic << '\n';
    for (const Diagnostic &note : diagnostic.getNotes())
      stream << note << '\n';
    return success();
  });
  selected->second(context);
  llvm::outs() << checks << ' ' << selected->first << " checks, " << failures << " failures\n";
  return failures != 0;
}
