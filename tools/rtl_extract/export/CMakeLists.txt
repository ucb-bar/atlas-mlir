cmake_minimum_required(VERSION 3.20)
project(atlas_hw_ir_export LANGUAGES C CXX)
find_package(CIRCT REQUIRED CONFIG)
add_executable(hw_ir_export hw_ir_export.cpp)
target_include_directories(hw_ir_export SYSTEM PRIVATE
  ${CIRCT_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_compile_features(hw_ir_export PRIVATE cxx_std_17)
target_compile_options(hw_ir_export PRIVATE -fno-rtti)
target_link_libraries(hw_ir_export PRIVATE
  CIRCTHW CIRCTComb CIRCTSeq CIRCTSV CIRCTVerif CIRCTLTL CIRCTEmit CIRCTOM
  CIRCTSim MLIRParser MLIRIR LLVMSupport)
