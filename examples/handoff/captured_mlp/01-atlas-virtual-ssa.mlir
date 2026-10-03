module {
  func.func @atlas_compiled() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415984640 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %t0 = "atlas.virtual_input_fp8"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2, %t1 = "atlas.virtual_input_fp8"(%io1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %t2 = "atlas.virtual_mxu_matmul"(%t0, %t1) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io3, %t4 = "atlas.virtual_input_bf16"(%io2) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %t3 = "atlas.virtual_vpu_binary"(%t2, %t4) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %t5 = "atlas.virtual_vpu_unary"(%t3) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %t6 = "atlas.virtual_pack_fp8"(%t5) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io4, %t7 = "atlas.virtual_input_fp8"(%io3) {index = 3 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %t8 = "atlas.virtual_mxu_matmul"(%t6, %t7) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io5, %t10 = "atlas.virtual_input_bf16"(%io4) {index = 4 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %t9 = "atlas.virtual_vpu_binary"(%t8, %t10) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %io6 = "atlas.virtual_output_bf16"(%io5, %t9) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io6 : !atlas.virtual_state
  }
}
