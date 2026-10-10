#include "Atlas/AtlasGeneratedArtifact.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::atlas;

namespace {
constexpr llvm::StringRef kTags[] = {
    kAtlasTagDMATransfer, kAtlasTagMXUCommand, kAtlasTagTileCommand,
    kAtlasTagCFGBlock, kAtlasTagCFGEdge, kAtlasTagCFGBranch,
    kAtlasTagCFGSource, kAtlasTagCFGOperation, kAtlasTagCFGHelper,
    kAtlasTagScalarResult, kAtlasTagTensorResult, kAtlasTagScalarArgument};
constexpr llvm::StringRef kContracts[] = {kAtlasDMAContract, kAtlasMXUContract,
                                          kAtlasTileContract, kAtlasCFGContract,
                                          kAtlasSourceMemoryContract, kAtlasBufferContract};
constexpr llvm::StringRef kPreserved[] = {
    kAtlasGeneratedMarker, kAtlasDMAContract, kAtlasMXUContract,
    kAtlasTileContract, kAtlasCFGContract, kAtlasSourceMemoryContract, kAtlasBufferContract,
    kAtlasTimingState, kAtlasTimingProvider};
} // namespace

ArrayRef<StringRef> mlir::atlas::atlasPreservedModuleAttrs() { return kPreserved; }

FailureOr<AtlasArtifactKind> mlir::atlas::classifyAtlasGeneratedArtifact(ModuleOp module) {
  Attribute marker = module->getAttr(kAtlasGeneratedMarker);
  if (!marker) {
    bool metadata = llvm::any_of(kContracts, [&](StringRef name) { return module->hasAttr(name); }) ||
                    llvm::any_of(module.getBody()->getOperations(), [](Operation &op) {
                      return llvm::any_of(kTags, [&](StringRef tag) { return op.hasAttr(tag); });
                    });
    if (!metadata)
      return AtlasArtifactKind::HandWritten;
    return module.emitOpError("generated resource metadata requires an Atlas virtual-to-machine artifact marked ")
        << kAtlasGeneratedMarker << " = \"" << kAtlasGeneratedVersion << '"';
  }
  auto version = dyn_cast<StringAttr>(marker);
  if (!version || version.getValue() != kAtlasGeneratedVersion)
    return module.emitOpError("unsupported Atlas virtual-to-machine artifact marker; only ")
        << kAtlasGeneratedMarker << " = \"" << kAtlasGeneratedVersion << "\" is supported";
  for (StringRef name : kContracts)
    if (!module->hasAttr(name))
      return module.emitOpError() << kAtlasGeneratedVersion << " artifact requires " << name;
  if (!module->hasAttr(kAtlasTimingState))
    return module.emitOpError() << kAtlasGeneratedVersion << " artifact requires an explicit " << kAtlasTimingState;
  return AtlasArtifactKind::Generated;
}

LogicalResult mlir::atlas::requireAtlasGeneratedArtifact(ModuleOp module) {
  auto kind = classifyAtlasGeneratedArtifact(module);
  if (failed(kind))
    return failure();
  if (*kind != AtlasArtifactKind::Generated)
    return module.emitOpError("expected an Atlas virtual-to-machine artifact");
  return success();
}
