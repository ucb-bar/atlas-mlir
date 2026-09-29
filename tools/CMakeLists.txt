add_llvm_executable(atlas-opt atlas-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-opt)
target_link_libraries(atlas-opt PRIVATE MLIROptLib MLIRAtlas)
mlir_check_all_link_libraries(atlas-opt)

add_llvm_executable(atlas-emit atlas-emit.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-emit)
target_link_libraries(atlas-emit PRIVATE MLIRParser MLIRAtlas)
mlir_check_all_link_libraries(atlas-emit)
