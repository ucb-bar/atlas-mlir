module {
  func.func @native_tile() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415925248 : i64, atlas.output_dram_base = 2415990784 : i64} {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %v1 = "atlas.virtual_input_fp8"(%s0) {index = 5 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s2, %v3 = "atlas.virtual_input_fp8"(%s1) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s3, %weight4 = "atlas.virtual_mxu_load_weight"(%s2, %v3) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %s4, %acc4 = "atlas.virtual_mxu_reset"(%s3, %v1, %weight4) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s5, %v4 = "atlas.virtual_mxu_readout_bf16"(%s4, %acc4) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %v5 = "atlas.virtual_pack_fp8"(%v4) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %s6, %v7 = "atlas.virtual_input_fp8"(%s5) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s7, %weight8 = "atlas.virtual_mxu_load_weight"(%s6, %v7) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %s8, %acc8 = "atlas.virtual_mxu_reset"(%s7, %v1, %weight8) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s9, %v8 = "atlas.virtual_mxu_readout_bf16"(%s8, %acc8) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %v9 = "atlas.virtual_pack_fp8"(%v8) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %s10, %weight10 = "atlas.virtual_mxu_load_weight"(%s9, %v9) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %s11, %acc10 = "atlas.virtual_mxu_reset"(%s10, %v5, %weight10) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s12, %v10 = "atlas.virtual_mxu_readout_bf16"(%s11, %acc10) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s13, %v12 = "atlas.virtual_input_bf16"(%s12) {index = 6 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %v13 = "atlas.virtual_vpu_binary"(%v10, %v12) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %v14 = "atlas.virtual_vpu_reduce"(%v13) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %v15 = "atlas.virtual_vpu_binary"(%v13, %v14) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %v16 = "atlas.virtual_vpu_unary"(%v15) {kind = "exp"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %v17 = "atlas.virtual_vpu_reduce"(%v16) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %v18 = "atlas.virtual_vpu_unary"(%v17) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %v19 = "atlas.virtual_vpu_binary"(%v16, %v18) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %v20 = "atlas.virtual_pack_fp8"(%v19) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %s14, %v22 = "atlas.virtual_input_fp8"(%s13) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s15, %weight23 = "atlas.virtual_mxu_load_weight"(%s14, %v22) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %s16, %acc23 = "atlas.virtual_mxu_reset"(%s15, %v1, %weight23) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s17, %v23 = "atlas.virtual_mxu_readout_bf16"(%s16, %acc23) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %v24 = "atlas.virtual_pack_fp8"(%v23) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %v25 = "atlas.virtual_xlu_transpose"(%v24) : (!atlas.virtual_fp8) -> !atlas.virtual_fp8
    %s18, %weight26 = "atlas.virtual_mxu_load_weight"(%s17, %v25) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %s19, %acc26 = "atlas.virtual_mxu_reset"(%s18, %v20, %weight26) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s20, %v26 = "atlas.virtual_mxu_readout_bf16"(%s19, %acc26) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %v27 = "atlas.virtual_pack_fp8"(%v26) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %s21, %v29 = "atlas.virtual_input_fp8"(%s20) {index = 3 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s22, %weight30 = "atlas.virtual_mxu_load_weight"(%s21, %v29) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %s23, %acc30 = "atlas.virtual_mxu_reset"(%s22, %v27, %weight30) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s24, %v30 = "atlas.virtual_mxu_readout_bf16"(%s23, %acc30) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %v31 = "atlas.virtual_vpu_unary"(%v30) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %s25 = "atlas.virtual_output_bf16"(%s24, %v31) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %v33 = "atlas.virtual_pack_fp8"(%v31) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %s26, %v35 = "atlas.virtual_input_fp8"(%s25) {index = 4 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s27, %weight36 = "atlas.virtual_mxu_load_weight"(%s26, %v35) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %s28, %acc36 = "atlas.virtual_mxu_reset"(%s27, %v33, %weight36) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s29, %v36 = "atlas.virtual_mxu_readout_bf16"(%s28, %acc36) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s30 = "atlas.virtual_output_bf16"(%s29, %v36) {index = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s30 : !atlas.virtual_state
  }
}
