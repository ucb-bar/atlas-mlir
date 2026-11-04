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
# Explicit source-join gate. Required capture inputs and dependencies must be
# supplied by the caller; the runner turns any required skip into a failure.
add_custom_target(atlas-faithfulness
  COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tools/check_faithfulness.py
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  USES_TERMINAL)
# One test per Python test module, run from test/ as discovery would, with the
# native binding package importable. Modules of plain test functions use
# pytest; the rest are unittest suites.
# Includes test_model_coverage.py, which checks plans against the pinned kernel fixtures.
file(GLOB atlas_python_tests CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/test/test_*.py)
set(atlas_python_paths
  PYTHONPATH=path_list_prepend:${PROJECT_SOURCE_DIR}/python
  PYTHONPATH=path_list_prepend:${PROJECT_SOURCE_DIR}/tools)
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
# Kernel generators live in the atlas-kernels submodule; run each of its test
# modules from its own tests directory, one test per module as above.
file(GLOB atlas_kernel_tests
  ${PROJECT_SOURCE_DIR}/third_party/atlas-kernels/kernels/*/tests/test_*.py
  ${PROJECT_SOURCE_DIR}/third_party/atlas-kernels/lib/tests/test_*.py)
foreach(test_file IN LISTS atlas_kernel_tests)
  get_filename_component(module ${test_file} NAME_WE)
  get_filename_component(suite_dir ${test_file} DIRECTORY)
  get_filename_component(suite ${suite_dir} DIRECTORY)
  get_filename_component(suite ${suite} NAME)
  add_test(NAME atlas-kernels-${suite}-${module}
    COMMAND ${Python3_EXECUTABLE} -m unittest -v ${module}
    WORKING_DIRECTORY ${suite_dir})
  # lib/atlas_backend.py encodes kernels with the atlas-opt and atlas-emit built here.
  set_tests_properties(atlas-kernels-${suite}-${module} PROPERTIES
    ENVIRONMENT ATLAS_OOT_BIN_DIR=$<TARGET_FILE_DIR:atlas-opt>)
endforeach()
