cmake_minimum_required(VERSION 3.22)
project(atlas_mlir LANGUAGES CXX C)

# An editable pip install ships only the Python packages; the tools come from
# the separate CMake build in the root README.
if(SKBUILD_STATE STREQUAL "editable")
  return()
endif()

option(ATLAS_ENABLE_TESTS "Build the host runtime libraries and the tests" ON)

find_package(MLIR REQUIRED CONFIG)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR} ${LLVM_CMAKE_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)
separate_arguments(ATLAS_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${ATLAS_LLVM_DEFINITIONS})
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(SKBUILD)
  # The wheel installs only this component: atlas-opt and atlas-emit with every
  # non-system shared library they load, plus the scripts and files atlasc
  # runs, in the source layout.
  set(wheel COMPONENT atlas-wheel)
  set_target_properties(atlas-opt atlas-emit PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
  install(TARGETS atlas-opt atlas-emit RUNTIME DESTINATION atlas_mlir/bin ${wheel})
  # A wheel cannot hold symlinks, so each library is copied once under the
  # name the loader asks for. The build-tree binaries still carry the
  # toolchain's RPATH, so this finds the C++ runtime they were linked against
  # rather than an older system one.
  install(CODE "set(atlas_llvm_library_dir \"${LLVM_LIBRARY_DIR}\")
                set(atlas_strip \"${CMAKE_STRIP}\")
                set(atlas_smoke_source \"${PROJECT_SOURCE_DIR}/test/examples/virtual_fp8_two_layer_mlp.mlir\")" ${wheel})
  install(CODE [[
    # A tool can print --help yet crash when an incompatible compiler/LLVM ABI
    # first parses MLIR. Refuse to publish such a wheel.
    execute_process(COMMAND "$<TARGET_FILE:atlas-opt>" --atlas-compile
      "${atlas_smoke_source}" -o "${CMAKE_INSTALL_PREFIX}/atlas-smoke.mlir"
      RESULT_VARIABLE smoke_status ERROR_VARIABLE smoke_error)
    if(NOT smoke_status EQUAL 0)
      message(FATAL_ERROR "Atlas compiler smoke test failed (${smoke_status}). Use the C/C++ compiler belonging to the LLVM installation (activate its conda environment). ${smoke_error}")
    endif()
    execute_process(COMMAND "$<TARGET_FILE:atlas-emit>" "${CMAKE_INSTALL_PREFIX}/atlas-smoke.mlir"
      RESULT_VARIABLE smoke_status OUTPUT_QUIET ERROR_VARIABLE smoke_error)
    file(REMOVE "${CMAKE_INSTALL_PREFIX}/atlas-smoke.mlir")
    if(NOT smoke_status EQUAL 0)
      message(FATAL_ERROR "Atlas emitter smoke test failed: ${smoke_error}")
    endif()
    file(GET_RUNTIME_DEPENDENCIES
      EXECUTABLES "$<TARGET_FILE:atlas-opt>" "$<TARGET_FILE:atlas-emit>"
      DIRECTORIES "${atlas_llvm_library_dir}"
      PRE_EXCLUDE_REGEXES "^ld-linux" "^lib(c|m|dl|rt|pthread)\\.so"
      POST_EXCLUDE_REGEXES "^/lib" "^/usr/lib"
      RESOLVED_DEPENDENCIES_VAR libraries
      UNRESOLVED_DEPENDENCIES_VAR missing)
    if(missing)
      message(FATAL_ERROR "atlas-opt needs libraries that were not found: ${missing}")
    endif()
    file(MAKE_DIRECTORY "${CMAKE_INSTALL_PREFIX}/atlas_mlir/lib")
    foreach(library IN LISTS libraries)
      get_filename_component(name "${library}" NAME)
      file(REAL_PATH "${library}" source)
      set(destination "${CMAKE_INSTALL_PREFIX}/atlas_mlir/lib/${name}")
      message(STATUS "Bundling: ${destination}")
      file(COPY_FILE "${source}" "${destination}")
      if(CMAKE_INSTALL_DO_STRIP AND atlas_strip)
        execute_process(COMMAND "${atlas_strip}" --strip-unneeded "${destination}"
                        COMMAND_ERROR_IS_FATAL ANY)
      endif()
    endforeach()
  ]] ${wheel})
  set(share atlas_mlir/share)
  install(PROGRAMS tools/compile_atlas_linalg_tile.py tools/atlas_boot_pack.py
    tools/atlas_captured_host_data.py tools/atlas_host_embed.py
    DESTINATION ${share}/tools ${wheel})
  install(FILES docs/selected-variant-inventory.json DESTINATION ${share}/docs ${wheel})
  install(FILES runtime/atlas_host.c runtime/atlas_host.h runtime/atlas_run.c
    runtime/selected-host-map.json DESTINATION ${share}/runtime ${wheel})
  install(FILES examples/handoff/captured_attention_mlp/start.S
    examples/handoff/captured_attention_mlp/link.ld
    DESTINATION ${share}/examples/handoff/captured_attention_mlp ${wheel})
  install(FILES test/examples/captured_mlp32_linalg.mlir
    test/examples/captured_mlp32_arguments.json
    test/examples/captured_mlp32_weights.safetensors
    DESTINATION ${share}/test/examples ${wheel})
endif()

if(NOT ATLAS_ENABLE_TESTS)
  return()
endif()
add_subdirectory(runtime)

enable_testing()
add_subdirectory(test)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
# One test per Python test module, run from test/ as discovery would, with the
# native binding package importable. Modules of plain test functions use
# pytest; the rest are unittest suites.
file(GLOB atlas_python_tests CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/test/test_*.py)
foreach(test_file IN LISTS atlas_python_tests)
  get_filename_component(module ${test_file} NAME_WE)
  file(STRINGS ${test_file} unittest_suite REGEX "unittest\\.TestCase")
  if(unittest_suite)
    set(runner unittest -v ${module})
  else()
    set(runner pytest -v -p no:cacheprovider ${module}.py)
  endif()
  add_test(NAME atlas-python-${module}
    COMMAND ${Python3_EXECUTABLE} -m ${runner}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
  set_tests_properties(atlas-python-${module} PROPERTIES
    ENVIRONMENT ATLAS_OOT_BIN_DIR=$<TARGET_FILE_DIR:atlas-opt>
    ENVIRONMENT_MODIFICATION PYTHONPATH=path_list_prepend:${PROJECT_SOURCE_DIR}/python)
endforeach()
