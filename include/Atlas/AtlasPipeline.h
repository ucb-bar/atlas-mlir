#ifndef ATLAS_PIPELINE_H
#define ATLAS_PIPELINE_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::atlas {

struct AtlasCompileOptions : PassPipelineOptions<AtlasCompileOptions> {
  Option<unsigned> programSizeLimit{
      *this, "program-size-limit-bytes", llvm::cl::init(0),
      llvm::cl::desc("Optional smaller instruction-memory budget in bytes")};
  Option<std::string> allocationReport{
      *this, "allocation-report",
      llvm::cl::desc("Write the allocation plan as JSON to this file, also "
                     "when it does not fit")};
};

// --atlas-compile: one virtual function to timed machine IR, with every
// verifier. MXU and register pressure are recovered by spilling, boundary
// tiles become schedulable DMA, and the stream scheduler reorders each block
// and inserts its delays. Every tool that compiles virtual IR runs this, or
// its stages in order: --atlas-compile-verify, -schedule, -spill, -lower and
// -time.
void buildAtlasCompilePipeline(OpPassManager &pm,
                               const AtlasCompileOptions &options);

// --atlas-compile-unscheduled: the native binding's qualification lowering.
// The same spilling, lowering and verifiers, but no boundary DMA, virtual or
// stream scheduling: delays are inserted in source order.
void buildAtlasCompileUnscheduledPipeline(OpPassManager &pm,
                                          const AtlasCompileOptions &options);

void registerAtlasCompilePipelines();

} // namespace mlir::atlas

#endif
