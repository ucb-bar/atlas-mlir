#ifndef ATLAS_PROGRAM_COMPOSITION_H
#define ATLAS_PROGRAM_COMPOSITION_H

namespace mlir::atlas {
// Compose explicitly ordered virtual functions before scheduling and allocation.
void registerComposeAtlasProgramPass();
} // namespace mlir::atlas

#endif
