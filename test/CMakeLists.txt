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

add_llvm_executable(atlas-dma-helper-test
  dma-helper-verification.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-dma-helper-test)
target_link_libraries(atlas-dma-helper-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR MLIRArithDialect MLIRControlFlowDialect
  MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-dma-helper-test)
add_test(NAME atlas-dma-helper
  COMMAND atlas-dma-helper-test)

add_llvm_executable(atlas-dma-contract-test
  dma-contract-verification.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-dma-contract-test)
target_link_libraries(atlas-dma-contract-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR MLIRArithDialect MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-dma-contract-test)
add_test(NAME atlas-dma-contract
  COMMAND atlas-dma-contract-test)

add_llvm_executable(atlas-mxu-contract-test
  mxu-contract-verification.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-mxu-contract-test)
target_link_libraries(atlas-mxu-contract-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR MLIRArithDialect MLIRControlFlowDialect MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-mxu-contract-test)
add_test(NAME atlas-mxu-contract COMMAND atlas-mxu-contract-test)

add_llvm_executable(atlas-tile-contract-test
  tile-contract-verification.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-tile-contract-test)
target_link_libraries(atlas-tile-contract-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR MLIRArithDialect MLIRControlFlowDialect MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-tile-contract-test)
add_test(NAME atlas-tile-contract COMMAND atlas-tile-contract-test)

add_llvm_executable(atlas-timing-provider-test
  timing-provider-verification.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-timing-provider-test)
target_link_libraries(atlas-timing-provider-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR)
mlir_check_all_link_libraries(atlas-timing-provider-test)
add_test(NAME atlas-timing-provider COMMAND atlas-timing-provider-test)
