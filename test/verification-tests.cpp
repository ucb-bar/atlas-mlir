#include "VerificationTestSupport.h"

int main(int argc, char **argv) {
  using namespace atlas_test;
  static const std::pair<llvm::StringRef, Suite> suites[] = {
      {"buffer-contract", runBufferContract},
      {"cfg-contract", runCFGContract},
      {"dma-allocation", runDMAAllocation},
      {"dma-contract", runDMAContract},
      {"dma-helper", runDMAHelper},
      {"mxu-allocation", runMXUAllocation},
      {"mxu-contract", runMXUContract},
      {"source-memory-contract", runSourceMemoryContract},
      {"tile-contract", runTileContract},
      {"timing-provider", runTimingProvider},
      {"verification-context", runVerificationContext},
  };
  return runSuite(argc, argv, suites);
}
