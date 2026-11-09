add_library(atlas-host STATIC atlas_host.c)
target_include_directories(atlas-host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(atlas-host PROPERTIES C_STANDARD 11)
install(TARGETS atlas-host ARCHIVE DESTINATION lib)
install(FILES atlas_host.h DESTINATION include)

add_library(atlas-model-tile STATIC atlas_model_tile.c)
target_include_directories(atlas-model-tile PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(atlas-model-tile PUBLIC atlas-host)
set_target_properties(atlas-model-tile PROPERTIES C_STANDARD 11)
install(TARGETS atlas-model-tile ARCHIVE DESTINATION lib)
install(FILES atlas_model_tile.h DESTINATION include)

add_library(atlas-ordered-k STATIC atlas_ordered_k.c)
target_include_directories(atlas-ordered-k PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(atlas-ordered-k PUBLIC atlas-host)
set_target_properties(atlas-ordered-k PROPERTIES C_STANDARD 11)
install(TARGETS atlas-ordered-k ARCHIVE DESTINATION lib)
install(FILES atlas_ordered_k.h DESTINATION include)

add_library(atlas-ordered-mn STATIC atlas_ordered_mn.c)
target_include_directories(atlas-ordered-mn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(atlas-ordered-mn PUBLIC atlas-ordered-k)
set_target_properties(atlas-ordered-mn PROPERTIES C_STANDARD 11)
install(TARGETS atlas-ordered-mn ARCHIVE DESTINATION lib)
install(FILES atlas_ordered_mn.h DESTINATION include)

add_library(atlas-source-mn STATIC atlas_source_mn.c)
target_include_directories(atlas-source-mn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(atlas-source-mn PUBLIC atlas-ordered-mn)
set_target_properties(atlas-source-mn PROPERTIES C_STANDARD 11)
install(TARGETS atlas-source-mn ARCHIVE DESTINATION lib)
install(FILES atlas_source_mn.h DESTINATION include)

# The generic precision-boundary algorithms have one authority in Merlin.
# Select its source or installed runtime-header directory explicitly.
find_path(MERLIN_RUNTIME_INCLUDE_DIR NAMES finite_float_boundary.h
  HINTS ENV MERLIN_RUNTIME_INCLUDE_DIR
  DOC "Selected Merlin runtime headers containing finite_float_boundary.h"
  REQUIRED)
add_library(atlas-graph-memref STATIC atlas_graph_memref.c)
target_include_directories(atlas-graph-memref PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR} ${MERLIN_RUNTIME_INCLUDE_DIR})
target_link_libraries(atlas-graph-memref PUBLIC atlas-model-tile)
set_target_properties(atlas-graph-memref PROPERTIES C_STANDARD 11)
install(TARGETS atlas-graph-memref ARCHIVE DESTINATION lib)
install(FILES atlas_graph_memref.h DESTINATION include)

add_library(atlas-bf16-call STATIC atlas_bf16_call.c)
target_include_directories(atlas-bf16-call PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR} ${MERLIN_RUNTIME_INCLUDE_DIR})
target_link_libraries(atlas-bf16-call PUBLIC atlas-host)
set_target_properties(atlas-bf16-call PROPERTIES C_STANDARD 11)
install(TARGETS atlas-bf16-call ARCHIVE DESTINATION lib)
install(FILES atlas_bf16_call.h DESTINATION include)
