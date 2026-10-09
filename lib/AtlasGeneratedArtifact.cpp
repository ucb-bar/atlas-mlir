#include "Atlas/AtlasGeneratedArtifact.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::atlas;

namespace {
// Per-instruction correspondence tags that only generated artifacts carry.
constexpr llvm::StringRef kTags[] = {
    "atlas.virtual_dma_transfer", "atlas.virtual_mxu_command",
    "atlas.virtual_tile_command", "atlas.virtual_cfg_block",
    "atlas.virtual_cfg_edge", "atlas.virtual_cfg_branch",
    "atlas.virtual_cfg_source", "atlas.virtual_cfg_operation",
    "atlas.virtual_cfg_helper", "atlas.virtual_scalar_result",
    "atlas.virtual_tensor_result", "atlas.virtual_scalar_argument"};
constexpr llvm::StringRef kContracts[] = {kAtlasDMAContract, kAtlasMXUContract,
                                          kAtlasTileContract, kAtlasCFGContract,
                                          kAtlasSourceMemoryContract};
constexpr llvm::StringRef kPreserved[] = {
    kAtlasGeneratedMarker, kAtlasDMAContract, kAtlasMXUContract,
    kAtlasTileContract, kAtlasCFGContract, kAtlasSourceMemoryContract,
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
    module.emitOpError("generated resource metadata requires an Atlas virtual-to-machine artifact marked ")
        << kAtlasGeneratedMarker << " = \"" << kAtlasGeneratedVersion << '"';
    return failure();
  }
  auto version = dyn_cast<StringAttr>(marker);
  if (!version || version.getValue() != kAtlasGeneratedVersion) {
    module.emitOpError("unsupported Atlas virtual-to-machine artifact marker; only ")
        << kAtlasGeneratedMarker << " = \"" << kAtlasGeneratedVersion << "\" is supported";
    return failure();
  }
  for (StringRef name : kContracts)
    if (!module->hasAttr(name)) {
      module.emitOpError() << kAtlasGeneratedVersion << " artifact requires " << name;
      return failure();
    }
  if (!module->hasAttr(kAtlasTimingState)) {
    module.emitOpError() << kAtlasGeneratedVersion << " artifact requires an explicit " << kAtlasTimingState;
    return failure();
  }
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
