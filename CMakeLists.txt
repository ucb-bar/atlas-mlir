cmake_minimum_required(VERSION 3.20)
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

enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME atlas-dialect
  COMMAND ${CMAKE_COMMAND} -E env
    ATLAS_OOT_BIN_DIR=$<TARGET_FILE_DIR:atlas-opt>
    ${Python3_EXECUTABLE} -m unittest discover -s ${PROJECT_SOURCE_DIR}/test -v)
