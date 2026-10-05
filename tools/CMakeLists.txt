add_llvm_executable(atlas-opt atlas-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-opt)
target_link_libraries(atlas-opt PRIVATE MLIROptLib MLIRAtlas)
mlir_check_all_link_libraries(atlas-opt)

add_llvm_executable(atlas-emit atlas-emit.cpp PARTIAL_SOURCES_INTENDED)
set(ATLAS_INVENTORY ${PROJECT_SOURCE_DIR}/docs/selected-variant-inventory.json)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ATLAS_INVENTORY})
file(READ ${ATLAS_INVENTORY} ATLAS_INVENTORY_JSON)
string(JSON ATLAS_SELECTED_RTL_REVISION GET "${ATLAS_INVENTORY_JSON}"
       selected_sources rtl_revision)
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/Atlas)
configure_file(AtlasSelectedTarget.h.in
               ${PROJECT_BINARY_DIR}/include/Atlas/AtlasSelectedTarget.h @ONLY)
llvm_update_compile_flags(atlas-emit)
target_link_libraries(atlas-emit PRIVATE MLIRParser MLIRAtlas)
mlir_check_all_link_libraries(atlas-emit)

# Host-side packaging of a checked reset-entry .text section. This tool is
# independent of ACT and is deliberately narrower than an ELF loader/ABI.
configure_file(atlas_boot_pack.py ${CMAKE_BINARY_DIR}/bin/atlas-boot-pack COPYONLY)
install(PROGRAMS atlas_boot_pack.py DESTINATION bin RENAME atlas-boot-pack)
