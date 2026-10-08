add_llvm_executable(atlas-register-allocation-test
  register-allocation-verification.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-register-allocation-test)
target_link_libraries(atlas-register-allocation-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR MLIRArithDialect MLIRControlFlowDialect
  MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-register-allocation-test)
add_test(NAME atlas-register-allocation
  COMMAND atlas-register-allocation-test)

add_llvm_executable(atlas-rtl-evidence-test rtl-evidence.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-rtl-evidence-test)
target_link_libraries(atlas-rtl-evidence-test PRIVATE MLIRAtlas)
mlir_check_all_link_libraries(atlas-rtl-evidence-test)
# This executable takes explicitly selected local evidence; no private artifact
# path or unqualified report is silently installed as a default CTest input.
