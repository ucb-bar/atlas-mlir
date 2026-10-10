#ifndef ATLAS_GENERATED_ARTIFACT_H
#define ATLAS_GENERATED_ARTIFACT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::atlas {

// Module attributes stamped by virtual-to-machine lowering.
constexpr llvm::StringLiteral kAtlasGeneratedMarker = "atlas.generated_from_virtual";
constexpr llvm::StringLiteral kAtlasGeneratedVersion = "resource-contract-v5";
constexpr llvm::StringLiteral kAtlasDMAContract = "atlas.virtual_dma_contract";
constexpr llvm::StringLiteral kAtlasMXUContract = "atlas.virtual_mxu_contract";
constexpr llvm::StringLiteral kAtlasTileContract = "atlas.virtual_tile_contract";
constexpr llvm::StringLiteral kAtlasCFGContract = "atlas.virtual_cfg_contract";
constexpr llvm::StringLiteral kAtlasSourceMemoryContract = "atlas.virtual_source_memory_contract";
constexpr llvm::StringLiteral kAtlasBufferContract = "atlas.virtual_buffer_contract";
constexpr llvm::StringLiteral kAtlasTimingState = "atlas.timing_state";
constexpr llvm::StringLiteral kAtlasTimingProvider = "atlas.timing_provider";
// Per-instruction correspondence tags that only generated artifacts carry.
constexpr llvm::StringLiteral kAtlasTagDMATransfer = "atlas.virtual_dma_transfer";
constexpr llvm::StringLiteral kAtlasTagMXUCommand = "atlas.virtual_mxu_command";
constexpr llvm::StringLiteral kAtlasTagTileCommand = "atlas.virtual_tile_command";
constexpr llvm::StringLiteral kAtlasTagCFGBlock = "atlas.virtual_cfg_block";
constexpr llvm::StringLiteral kAtlasTagCFGEdge = "atlas.virtual_cfg_edge";
constexpr llvm::StringLiteral kAtlasTagCFGBranch = "atlas.virtual_cfg_branch";
constexpr llvm::StringLiteral kAtlasTagCFGSource = "atlas.virtual_cfg_source";
constexpr llvm::StringLiteral kAtlasTagCFGOperation = "atlas.virtual_cfg_operation";
constexpr llvm::StringLiteral kAtlasTagCFGHelper = "atlas.virtual_cfg_helper";
constexpr llvm::StringLiteral kAtlasTagScalarResult = "atlas.virtual_scalar_result";
constexpr llvm::StringLiteral kAtlasTagTensorResult = "atlas.virtual_tensor_result";
constexpr llvm::StringLiteral kAtlasTagScalarArgument = "atlas.virtual_scalar_argument";

// Module attributes that LLVM finalization carries into its rebuilt stream.
llvm::ArrayRef<llvm::StringRef> atlasPreservedModuleAttrs();

enum class AtlasArtifactKind { HandWritten, Generated };

// HandWritten: no marker, contract attribute or tag. Generated: the supported
// marker, all six contracts and a timing state. Anything else fails with one
// diagnostic; contract contents are left to their own checkers.
FailureOr<AtlasArtifactKind> classifyAtlasGeneratedArtifact(ModuleOp module);
// Fails unless the module classifies as a generated artifact.
LogicalResult requireAtlasGeneratedArtifact(ModuleOp module);

} // namespace mlir::atlas

#endif
