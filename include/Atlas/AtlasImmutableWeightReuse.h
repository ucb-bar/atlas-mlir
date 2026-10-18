#ifndef ATLAS_IMMUTABLE_WEIGHT_REUSE_H
#define ATLAS_IMMUTABLE_WEIGHT_REUSE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::atlas {
// Requires verified virtual IR. Reuse repeated, provably unwritten FP8 weight
// inputs in a single block, only when the resulting source order allocates.
// Boundary slots and explicit constant-address DMA loads remain separate.
// No candidate, uncertain aliasing, or allocation failure leaves IR unchanged.
void reuseAtlasImmutableWeights(func::FuncOp function);
void registerReuseAtlasImmutableWeightsPass();
} // namespace mlir::atlas

#endif
