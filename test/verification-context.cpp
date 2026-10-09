#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasGeneratedSchedule.h"
#include "Atlas/AtlasStream.h"
#include "VerificationTestSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace atlas_test;

namespace {
struct Counts {
  unsigned decodes, encodes;
};

template <typename Fn>
Counts counted(Fn &&fn) {
  unsigned decodes = atlasStreamDecodeCount(), encodes = atlasWordEncodeCount();
  fn();
  return {atlasStreamDecodeCount() - decodes, atlasWordEncodeCount() - encodes};
}
} // namespace

// A real virtual program lowered in-process: each verification boundary decodes and encodes the artifact once.
void atlas_test::runVerificationContext(MLIRContext &context) {
  auto module = parseSourceFile<ModuleOp>(ATLAS_GENERATED_FIXTURE, ParserConfig(&context));
  check(bool(module), "virtual fixture parses");
  if (!module)
    return;
  auto pm = PassManager::on<ModuleOp>(&context);
  bool lowered = succeeded(parsePassPipeline("lower-atlas-virtual-to-machine,insert-atlas-delays", pm)) && succeeded(pm.run(*module));
  check(lowered, "virtual fixture lowers and is timed");
  if (!lowered)
    return;
  auto kind = classifyAtlasGeneratedArtifact(*module);
  check(succeeded(kind) && *kind == AtlasArtifactKind::Generated, "lowering output classifies as generated");
  llvm::SmallVector<uint32_t> expected;
  check(succeeded(encodeAtlasWords(*module, expected)) && !expected.empty(), "generated artifact encodes");
  bool ok = false;
  Counts counts = counted([&] { ok = succeeded(verifyAtlasGeneratedSchedule(*module)); });
  check(ok, "generated schedule accepts the lowered artifact");
  check(counts.decodes == 1 && counts.encodes == 1, "generated schedule decodes and encodes once");
  llvm::SmallVector<uint32_t> words;
  counts = counted([&] { ok = succeeded(verifyAtlasArtifact(*module, /*llvmBlock=*/true, words)); });
  check(ok && words == expected, "boundary returns the single encoding");
  check(counts.decodes == 1 && counts.encodes == 1, "boundary decodes and encodes once");
}
