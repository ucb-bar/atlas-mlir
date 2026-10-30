add_llvm_executable(atlas-register-allocation-test
  register-allocation-verification.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-register-allocation-test)
target_link_libraries(atlas-register-allocation-test PRIVATE
  MLIRAtlas MLIRParser MLIRIR MLIRArithDialect MLIRControlFlowDialect
  MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-register-allocation-test)
add_test(NAME atlas-register-allocation
  COMMAND atlas-register-allocation-test)

add_executable(atlas-host-launch-test host-launch-test.c)
target_link_libraries(atlas-host-launch-test PRIVATE atlas-host)
set_target_properties(atlas-host-launch-test PROPERTIES C_STANDARD 11)
add_test(NAME atlas-host-launch COMMAND atlas-host-launch-test)

add_executable(atlas-model-tile-test model-tile-test.c)
target_link_libraries(atlas-model-tile-test PRIVATE atlas-model-tile)
set_target_properties(atlas-model-tile-test PROPERTIES C_STANDARD 11)
add_test(NAME atlas-model-tile COMMAND atlas-model-tile-test)

add_executable(atlas-ordered-k-host-test ordered-k-host-test.c)
target_link_libraries(atlas-ordered-k-host-test PRIVATE atlas-ordered-k)
set_target_properties(atlas-ordered-k-host-test PROPERTIES C_STANDARD 11)
add_test(NAME atlas-ordered-k-host COMMAND atlas-ordered-k-host-test)

add_executable(atlas-ordered-mn-host-test ordered-mn-host-test.c)
target_link_libraries(atlas-ordered-mn-host-test PRIVATE atlas-source-mn)
set_target_properties(atlas-ordered-mn-host-test PROPERTIES C_STANDARD 11)
add_test(NAME atlas-ordered-mn-host COMMAND atlas-ordered-mn-host-test)

add_executable(atlas-ordered-mn-tail-host-test ordered-mn-host-test.c)
target_compile_definitions(atlas-ordered-mn-tail-host-test PRIVATE ATLAS_TEST_MN_TAILS=1)
target_link_libraries(atlas-ordered-mn-tail-host-test PRIVATE atlas-source-mn)
set_target_properties(atlas-ordered-mn-tail-host-test PROPERTIES C_STANDARD 11)
add_test(NAME atlas-ordered-mn-tail-host COMMAND atlas-ordered-mn-tail-host-test)

add_executable(atlas-ordered-mn-crossed-host-test ordered-mn-host-test.c)
target_compile_definitions(atlas-ordered-mn-crossed-host-test PRIVATE ATLAS_TEST_MN_CROSSED=1)
target_link_libraries(atlas-ordered-mn-crossed-host-test PRIVATE atlas-source-mn)
set_target_properties(atlas-ordered-mn-crossed-host-test PROPERTIES C_STANDARD 11)
add_test(NAME atlas-ordered-mn-crossed-host COMMAND atlas-ordered-mn-crossed-host-test)

foreach(thin_k IN ITEMS 1 65)
  add_executable(atlas-ordered-mn-thin-k${thin_k}-host-test ordered-mn-host-test.c)
  target_compile_definitions(atlas-ordered-mn-thin-k${thin_k}-host-test
    PRIVATE ATLAS_TEST_MN_THIN_K${thin_k}=1)
  target_link_libraries(atlas-ordered-mn-thin-k${thin_k}-host-test PRIVATE atlas-source-mn)
  set_target_properties(atlas-ordered-mn-thin-k${thin_k}-host-test PROPERTIES C_STANDARD 11)
  add_test(NAME atlas-ordered-mn-thin-k${thin_k}-host
    COMMAND atlas-ordered-mn-thin-k${thin_k}-host-test)
endforeach()

add_executable(atlas-ordered-mn-single-host-test ordered-mn-host-test.c)
target_compile_definitions(atlas-ordered-mn-single-host-test PRIVATE ATLAS_TEST_MN_SINGLE=1)
target_link_libraries(atlas-ordered-mn-single-host-test PRIVATE atlas-source-mn)
set_target_properties(atlas-ordered-mn-single-host-test PROPERTIES C_STANDARD 11)
add_test(NAME atlas-ordered-mn-single-host COMMAND atlas-ordered-mn-single-host-test)

add_executable(atlas-ordered-mn-k1-host-test ordered-mn-host-test.c)
target_compile_definitions(atlas-ordered-mn-k1-host-test PRIVATE ATLAS_TEST_MN_K1=1)
target_link_libraries(atlas-ordered-mn-k1-host-test PRIVATE atlas-source-mn)
set_target_properties(atlas-ordered-mn-k1-host-test PROPERTIES C_STANDARD 11)
add_test(NAME atlas-ordered-mn-k1-host COMMAND atlas-ordered-mn-k1-host-test)

foreach(k_tail IN ITEMS 1 31 65)
  add_executable(atlas-ordered-mn-ktail${k_tail}-host-test ordered-mn-host-test.c)
  target_compile_definitions(atlas-ordered-mn-ktail${k_tail}-host-test
    PRIVATE ATLAS_TEST_MN_KTAIL${k_tail}=1)
  target_link_libraries(atlas-ordered-mn-ktail${k_tail}-host-test PRIVATE atlas-source-mn)
  set_target_properties(atlas-ordered-mn-ktail${k_tail}-host-test PROPERTIES C_STANDARD 11)
  add_test(NAME atlas-ordered-mn-ktail${k_tail}-host
    COMMAND atlas-ordered-mn-ktail${k_tail}-host-test)
endforeach()

# One executable runs each verification suite named by its argument (see VerificationTestSupport.h).
add_llvm_executable(atlas-verification-test
  verification-tests.cpp
  buffer-contract-verification.cpp
  cfg-contract-verification.cpp
  dma-allocation-verification.cpp
  dma-helper-verification.cpp
  mxu-allocation-verification.cpp
  mxu-contract-verification.cpp
  pack-loop-verification.cpp
  source-memory-effect-contract.cpp
  tile-contract-verification.cpp
  timing-provider-verification.cpp
  model-tables-verification.cpp
  PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(atlas-verification-test)
target_compile_definitions(atlas-verification-test PRIVATE
  ATLAS_GENERATED_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/examples/virtual_bf16_shared_relu.mlir")
target_link_libraries(atlas-verification-test PRIVATE
  MLIRAtlas MLIRParser MLIRPass MLIRIR MLIRArithDialect MLIRControlFlowDialect
  MLIRFuncDialect)
mlir_check_all_link_libraries(atlas-verification-test)
foreach(suite buffer-contract cfg-contract dma-allocation dma-helper mxu-allocation mxu-contract
    pack-loop source-memory-contract tile-contract timing-provider model-tables)
  add_test(NAME atlas-${suite} COMMAND atlas-verification-test ${suite})
endforeach()
