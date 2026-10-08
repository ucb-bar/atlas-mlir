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
  AtlasDMAAllocationVerification.cpp
  AtlasMXUAllocationVerification.cpp
  AtlasDMAMemoryVerification.cpp
  AtlasDMAContractVerification.cpp
  AtlasMXUContractVerification.cpp
  AtlasGeneratedSchedule.cpp
  DEPENDS MLIRAtlasOpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRSideEffectInterfaces MLIRArithDialect
                   MLIRControlFlowDialect MLIRFuncDialect MLIRLLVMDialect
                   MLIRPass)
