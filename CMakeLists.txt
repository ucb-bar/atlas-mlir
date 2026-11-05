cmake_minimum_required(VERSION 3.22)
project(atlas_mlir LANGUAGES CXX C)

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
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
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
# native binding package importable. Modules of plain test functions use
# pytest; the rest are unittest suites.
file(GLOB atlas_python_tests CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/test/test_*.py)
set(atlas_python_paths
  PYTHONPATH=path_list_prepend:${PROJECT_SOURCE_DIR}/python
  PYTHONPATH=path_list_prepend:${PROJECT_SOURCE_DIR}/tools
  "${atlas_model_python_path}")
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
    ENVIRONMENT_MODIFICATION
      "${atlas_python_paths}")
endforeach()
