add_mlir_dialect_library(MLIRAtlas
  AtlasDialect.cpp
  AtlasTypes.cpp
  AtlasOps.cpp
  AtlasEncoding.cpp
  AtlasToLLVM.cpp
  AtlasStreamVerification.cpp
  AtlasTiming.cpp
  AtlasStream.cpp
  AtlasDelayInsertion.cpp
  AtlasScheduling.cpp
  AtlasVirtualVerification.cpp
  AtlasVirtualToMachine.cpp
  AtlasVirtualAllocation.cpp
  AtlasRegisterAllocationVerification.cpp
  AtlasGeneratedSchedule.cpp
  AtlasVirtualScheduling.cpp
  DEPENDS MLIRAtlasOpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRAnalysis MLIRSideEffectInterfaces MLIRArithDialect
                   MLIRControlFlowDialect MLIRFuncDialect MLIRLLVMDialect
                   MLIRPass)
