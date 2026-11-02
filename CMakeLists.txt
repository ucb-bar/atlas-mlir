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
add_definitions(${LLVM_DEFINITIONS})
file(READ "${PROJECT_SOURCE_DIR}/runtime/selected-host-map.json" ATLAS_HOST_MAP)
string(JSON ATLAS_IMEM_BYTES GET "${ATLAS_HOST_MAP}" imem_capacity_bytes)
configure_file(include/Atlas/AtlasTargetConfig.h.in
  include/Atlas/AtlasTargetConfig.h @ONLY)
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
                set(atlas_strip \"${CMAKE_STRIP}\")" ${wheel})
  install(CODE [[
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
endif()

if(NOT ATLAS_ENABLE_TESTS)
  return()
endif()
add_subdirectory(runtime)

enable_testing()
add_subdirectory(test)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
# These portable suites require the model even when no RTL tests are enabled.
# Validate it with the test interpreter, independently of ambient PYTHONPATH.
set(ATLAS_NPU_MODEL_ROOT "" CACHE PATH
  "npu-model checkout containing npu_model, required for CTest")
if("${ATLAS_NPU_MODEL_ROOT}" STREQUAL "" OR
    NOT EXISTS "${ATLAS_NPU_MODEL_ROOT}/npu_model/__init__.py")
  message(FATAL_ERROR
    "CTest requires -DATLAS_NPU_MODEL_ROOT=/path/to/npu-model "
    "(a checkout containing npu_model).")
endif()
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -I -c
    "import sys; sys.path.insert(0, sys.argv[1]); from npu_model.configs import isa_definition; from npu_model.configs.numerics import RtlNumerics; from npu_model.util.converter import input_to_program"
    "${ATLAS_NPU_MODEL_ROOT}"
  RESULT_VARIABLE atlas_model_import_result
  OUTPUT_VARIABLE atlas_model_import_output
  ERROR_VARIABLE atlas_model_import_error
  TIMEOUT 60)
if(NOT "${atlas_model_import_result}" STREQUAL "0")
  message(FATAL_ERROR
    "Cannot import the required npu_model assembler and RTL numerics from "
    "ATLAS_NPU_MODEL_ROOT=${ATLAS_NPU_MODEL_ROOT} with ${Python3_EXECUTABLE}. "
    "Install the model dependencies in that interpreter.\n"
    "${atlas_model_import_output}${atlas_model_import_error}")
endif()
set(atlas_model_python_path
  "PYTHONPATH=path_list_prepend:${ATLAS_NPU_MODEL_ROOT}")
# One test per Python test module, run from test/ as discovery would, with the
# native binding package importable. Modules with test functions or pytest
# fixtures use pytest; the rest are unittest suites.
file(GLOB atlas_python_tests CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/test/test_*.py)
set(atlas_python_paths
  PYTHONPATH=path_list_prepend:${PROJECT_SOURCE_DIR}/python
  PYTHONPATH=path_list_prepend:${PROJECT_SOURCE_DIR}
  PYTHONPATH=path_list_prepend:${PROJECT_SOURCE_DIR}/tools
  "${atlas_model_python_path}")
foreach(test_file IN LISTS atlas_python_tests)
  get_filename_component(module ${test_file} NAME_WE)
  file(STRINGS ${test_file} unittest_suite REGEX "unittest\\.TestCase")
  file(STRINGS ${test_file} pytest_suite REGEX "^def test_")
  file(STRINGS ${test_file} pytest_fixtures REGEX "^@pytest\\.fixture")
  if(unittest_suite AND NOT pytest_suite AND NOT pytest_fixtures)
    set(runner unittest -v ${module})
  else()
    set(runner pytest -v -p no:cacheprovider ${module}.py)
  endif()
  add_test(NAME atlas-python-${module}
    COMMAND ${Python3_EXECUTABLE} -m ${runner}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
  set_tests_properties(atlas-python-${module} PROPERTIES
    ENVIRONMENT ATLAS_OOT_BIN_DIR=$<TARGET_FILE_DIR:atlas-opt>
    ENVIRONMENT_MODIFICATION
      "${atlas_python_paths}")
endforeach()
