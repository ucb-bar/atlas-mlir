add_mlir_dialect_library(MLIRAtlas
  AtlasDialect.cpp
  AtlasTypes.cpp
  AtlasOps.cpp
  AtlasEncoding.cpp
  AtlasToLLVM.cpp
  AtlasStreamVerification.cpp
  AtlasTiming.cpp
  AtlasTimingProvider.cpp
  AtlasStream.cpp
  AtlasLoop.cpp
  AtlasPackLoop.cpp
  AtlasDelayInsertion.cpp
  AtlasDMAWaitInsertion.cpp
  AtlasScheduling.cpp
  AtlasVirtualVerification.cpp
  AtlasVirtualToMachine.cpp
  AtlasVirtualAllocation.cpp
  AtlasSpilling.cpp
  AtlasVirtualBoundaryToDMA.cpp
  AtlasVirtualScheduling.cpp
  AtlasRegisterAllocationVerification.cpp
  AtlasDMAAllocationVerification.cpp
  AtlasMXUAllocationVerification.cpp
  AtlasDMAContractVerification.cpp
  AtlasMXUContractVerification.cpp
  AtlasTileContractVerification.cpp
  AtlasCFGContractVerification.cpp
  AtlasSourceMemoryEffectContract.cpp
  AtlasBufferContractVerification.cpp
  AtlasGeneratedArtifact.cpp
  AtlasGeneratedSchedule.cpp
  AtlasVerificationContext.cpp
  DEPENDS MLIRAtlasOpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRAnalysis MLIRSideEffectInterfaces MLIRArithDialect
                   MLIRControlFlowDialect MLIRFuncDialect MLIRLLVMDialect
                   MLIRPass)
