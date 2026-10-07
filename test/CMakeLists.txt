# Two test executables share this directory.
add_llvm_executable(atlas-register-allocation-test PARTIAL_SOURCES_INTENDED
  register-allocation-verification.cpp)
llvm_update_compile_flags(atlas-register-allocation-test)
target_link_libraries(atlas-register-allocation-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR MLIRArithDialect MLIRControlFlowDialect
  MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-register-allocation-test)
add_test(NAME atlas-register-allocation
  COMMAND atlas-register-allocation-test)

add_llvm_executable(atlas-timing-test PARTIAL_SOURCES_INTENDED
  timing-model-test.cpp)
llvm_update_compile_flags(atlas-timing-test)
target_link_libraries(atlas-timing-test PRIVATE MLIRAtlas)
mlir_check_all_link_libraries(atlas-timing-test)
add_test(NAME atlas-timing-model COMMAND atlas-timing-test)
