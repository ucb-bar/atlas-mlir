#ifndef ATLAS_GENERATED_ARTIFACT_H
#define ATLAS_GENERATED_ARTIFACT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::atlas {

// Module attributes stamped by virtual-to-machine lowering. The marker names
// the only supported artifact version.
constexpr llvm::StringLiteral kAtlasGeneratedMarker = "atlas.generated_from_virtual";
constexpr llvm::StringLiteral kAtlasGeneratedVersion = "resource-contract-v4";
constexpr llvm::StringLiteral kAtlasDMAContract = "atlas.virtual_dma_contract";
constexpr llvm::StringLiteral kAtlasMXUContract = "atlas.virtual_mxu_contract";
constexpr llvm::StringLiteral kAtlasTileContract = "atlas.virtual_tile_contract";
constexpr llvm::StringLiteral kAtlasCFGContract = "atlas.virtual_cfg_contract";
constexpr llvm::StringLiteral kAtlasSourceMemoryContract = "atlas.virtual_source_memory_contract";
constexpr llvm::StringLiteral kAtlasTimingState = "atlas.timing_state";
constexpr llvm::StringLiteral kAtlasTimingProvider = "atlas.timing_provider";

// Module attributes that structured LLVM finalization carries into the stream
// it reconstructs.
llvm::ArrayRef<llvm::StringRef> atlasPreservedModuleAttrs();

enum class AtlasArtifactKind { HandWritten, Generated };

// HandWritten: no marker, contract attribute, or tag anywhere. Generated: the
// supported marker with all five contract attributes and a timing state.
// Any other combination fails with one diagnostic: legacy or unknown marker
// versions (including the bare unit marker), metadata without the marker, or
// the marker with a missing contract or timing state. Contract contents and
// timing-state values are validated by their own checkers.
FailureOr<AtlasArtifactKind> classifyAtlasGeneratedArtifact(ModuleOp module);
// Fails unless the module classifies as a generated artifact.
LogicalResult requireAtlasGeneratedArtifact(ModuleOp module);

} // namespace mlir::atlas

#endif
