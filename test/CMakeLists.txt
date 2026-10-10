add_llvm_executable(atlas-register-allocation-test
  register-allocation-verification.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-register-allocation-test)
target_link_libraries(atlas-register-allocation-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR MLIRArithDialect MLIRControlFlowDialect
  MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-register-allocation-test)
add_test(NAME atlas-register-allocation
  COMMAND atlas-register-allocation-test)

# One executable runs each verification suite named by its argument (see VerificationTestSupport.h).
add_llvm_executable(atlas-verification-test
  verification-tests.cpp
  buffer-contract-verification.cpp
  cfg-contract-verification.cpp
  dma-allocation-verification.cpp
  dma-contract-verification.cpp
  dma-helper-verification.cpp
  mxu-allocation-verification.cpp
  mxu-contract-verification.cpp
  source-memory-effect-contract.cpp
  tile-contract-verification.cpp
  timing-provider-verification.cpp
  verification-context.cpp
  PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-verification-test)
target_compile_definitions(atlas-verification-test PRIVATE
  ATLAS_GENERATED_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/examples/virtual_bf16_shared_relu.mlir")
target_link_libraries(atlas-verification-test PRIVATE
  MLIRAtlas MLIRParser MLIRPass MLIRIR MLIRArithDialect MLIRControlFlowDialect
  MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-verification-test)
foreach(suite buffer-contract cfg-contract dma-allocation dma-contract dma-helper mxu-allocation
    mxu-contract source-memory-contract tile-contract timing-provider
    verification-context)
  add_test(NAME atlas-${suite} COMMAND atlas-verification-test ${suite})
endforeach()
