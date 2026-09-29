add_mlir_dialect_library(MLIRAtlas
  AtlasDialect.cpp
  AtlasTypes.cpp
  AtlasOps.cpp
  DEPENDS MLIRAtlasOpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRSideEffectInterfaces)
