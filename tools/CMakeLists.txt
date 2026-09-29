add_llvm_executable(atlas-opt atlas-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-opt)
target_link_libraries(atlas-opt PRIVATE MLIROptLib MLIRAtlas)
mlir_check_all_link_libraries(atlas-opt)

add_llvm_executable(atlas-emit atlas-emit.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-emit)
target_link_libraries(atlas-emit PRIVATE MLIRParser MLIRAtlas)
mlir_check_all_link_libraries(atlas-emit)

# Host-side packaging of a checked reset-entry .text section. This tool is
# independent of ACT and is deliberately narrower than an ELF loader/ABI.
configure_file(atlas_boot_pack.py ${CMAKE_BINARY_DIR}/bin/atlas-boot-pack COPYONLY)
install(PROGRAMS atlas_boot_pack.py DESTINATION bin RENAME atlas-boot-pack)
