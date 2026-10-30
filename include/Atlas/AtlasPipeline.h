#ifndef ATLAS_PIPELINE_H
#define ATLAS_PIPELINE_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::atlas {

struct AtlasCompileOptions : PassPipelineOptions<AtlasCompileOptions> {
  Option<bool> waitReusedDMASlots{*this, "wait-reused-dma-slots",
      llvm::cl::desc("Wait before reusing DMA command slots"), llvm::cl::init(false)};
  Option<std::string> mregRanks{*this, "mreg-ranks", llvm::cl::desc("Free MREG ranks"), llvm::cl::init("")};
  Option<std::string> orderFile{*this, "order-file", llvm::cl::desc("Virtual order JSON"), llvm::cl::init("")};
  Option<std::string> graphOutput{*this, "graph-output", llvm::cl::desc("Virtual graph JSON"), llvm::cl::init("")};
  Option<bool> spill{*this, "spill", llvm::cl::desc("Recover register pressure"), llvm::cl::init(true)};
  enum class Boundary { All, None };
  Option<Boundary> boundary{
      *this, "boundary", llvm::cl::desc("Expose boundary DMA"),
      llvm::cl::values(clEnumValN(Boundary::All, "boundary", "Convert boundary tiles"),
                       clEnumValN(Boundary::None, "none", "Keep boundary tiles")),
      llvm::cl::init(Boundary::All)};
  Option<int> randomSeed{*this, "random-seed", llvm::cl::desc("Virtual schedule seed"), llvm::cl::init(-1)};
  Option<std::string> waitPlacement{*this, "wait-placement", llvm::cl::desc("Wait policy"), llvm::cl::init("default")};
  Option<std::string> provider{*this, "provider", llvm::cl::desc("Timing provider"), llvm::cl::init("")};
  Option<std::string> allocationReport{
      *this, "allocation-report",
      llvm::cl::desc("Write the allocation plan as JSON to this file, also "
                     "when it does not fit")};
  Option<bool> guardedPack{
      *this, "guarded-pack", llvm::cl::desc("Convert proven-finite PACK through MXU"),
      llvm::cl::init(false)};
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
