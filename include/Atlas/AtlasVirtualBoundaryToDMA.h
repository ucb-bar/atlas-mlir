#ifndef ATLAS_VIRTUAL_BOUNDARY_TO_DMA_H
#define ATLAS_VIRTUAL_BOUNDARY_TO_DMA_H

#include "mlir/Pass/Pass.h"

namespace mlir::atlas {
// --lower-atlas-virtual-boundary-to-dma: boundary input and output tiles
// become explicit virtual DMA at the DRAM addresses the boundary ABI gives
// them, so the virtual scheduler can queue and overlap their transfers.
void registerLowerAtlasVirtualBoundaryToDMAPass();
std::unique_ptr<Pass> createLowerAtlasVirtualBoundaryToDMAPass();
} // namespace mlir::atlas

#endif
