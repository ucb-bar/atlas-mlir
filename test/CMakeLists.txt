add_llvm_executable(atlas-register-allocation-test
  register-allocation-verification.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-register-allocation-test)
target_link_libraries(atlas-register-allocation-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR MLIRArithDialect MLIRControlFlowDialect
  MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-register-allocation-test)
add_test(NAME atlas-register-allocation
  COMMAND atlas-register-allocation-test)

add_llvm_executable(atlas-dma-timing-core-test
  dma-timing-core.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-dma-timing-core-test)
target_link_libraries(atlas-dma-timing-core-test PRIVATE MLIRAtlas)
mlir_check_all_link_libraries(atlas-dma-timing-core-test)
add_test(NAME atlas-dma-timing-core
  COMMAND atlas-dma-timing-core-test)

add_llvm_executable(atlas-dma-stream-policy-test
  dma-stream-policy.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-dma-stream-policy-test)
target_link_libraries(atlas-dma-stream-policy-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR)
mlir_check_all_link_libraries(atlas-dma-stream-policy-test)
add_test(NAME atlas-dma-stream-policy
  COMMAND atlas-dma-stream-policy-test)
