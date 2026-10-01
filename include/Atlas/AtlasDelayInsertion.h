#ifndef ATLAS_DELAY_INSERTION_H
#define ATLAS_DELAY_INSERTION_H

namespace mlir::atlas {

// Register the pass that recomputes every atlas.delay from the timing model.
void registerInsertAtlasDelaysPass();

} // namespace mlir::atlas

#endif
