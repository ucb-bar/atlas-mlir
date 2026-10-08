add_llvm_executable(atlas-register-allocation-test
  register-allocation-verification.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-register-allocation-test)
target_link_libraries(atlas-register-allocation-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR MLIRArithDialect MLIRControlFlowDialect
  MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-register-allocation-test)
add_test(NAME atlas-register-allocation
  COMMAND atlas-register-allocation-test)

add_llvm_executable(atlas-dma-allocation-test
  dma-allocation-verification.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-dma-allocation-test)
target_link_libraries(atlas-dma-allocation-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR MLIRArithDialect MLIRControlFlowDialect
  MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-dma-allocation-test)
add_test(NAME atlas-dma-allocation
  COMMAND atlas-dma-allocation-test)

add_llvm_executable(atlas-mxu-allocation-test
  mxu-allocation-verification.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-mxu-allocation-test)
target_link_libraries(atlas-mxu-allocation-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR MLIRArithDialect MLIRControlFlowDialect
  MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-mxu-allocation-test)
add_test(NAME atlas-mxu-allocation
  COMMAND atlas-mxu-allocation-test)
