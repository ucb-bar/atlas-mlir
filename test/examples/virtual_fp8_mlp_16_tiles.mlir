module {
  func.func @multi_tile_mlp() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415984640 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %w0_0 = "atlas.virtual_input_fp8"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2, %w1_0 = "atlas.virtual_input_fp8"(%io1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3, %x0 = "atlas.virtual_input_fp8"(%io2) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_0 = "atlas.virtual_mxu_matmul"(%x0, %w0_0) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_0 = "atlas.virtual_vpu_unary"(%h0_0) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_0 = "atlas.virtual_pack_fp8"(%h1_0) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y0 = "atlas.virtual_mxu_matmul"(%h2_0, %w1_0) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io4 = "atlas.virtual_output_bf16"(%io3, %y0) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io5, %w0_1 = "atlas.virtual_input_fp8"(%io4) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io6, %w1_1 = "atlas.virtual_input_fp8"(%io5) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io7, %x1 = "atlas.virtual_input_fp8"(%io6) {index = 3 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_1 = "atlas.virtual_mxu_matmul"(%x1, %w0_1) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_1 = "atlas.virtual_vpu_unary"(%h0_1) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_1 = "atlas.virtual_pack_fp8"(%h1_1) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y1 = "atlas.virtual_mxu_matmul"(%h2_1, %w1_1) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io8 = "atlas.virtual_output_bf16"(%io7, %y1) {index = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io9, %w0_2 = "atlas.virtual_input_fp8"(%io8) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io10, %w1_2 = "atlas.virtual_input_fp8"(%io9) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io11, %x2 = "atlas.virtual_input_fp8"(%io10) {index = 4 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_2 = "atlas.virtual_mxu_matmul"(%x2, %w0_2) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_2 = "atlas.virtual_vpu_unary"(%h0_2) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_2 = "atlas.virtual_pack_fp8"(%h1_2) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y2 = "atlas.virtual_mxu_matmul"(%h2_2, %w1_2) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io12 = "atlas.virtual_output_bf16"(%io11, %y2) {index = 2 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io13, %w0_3 = "atlas.virtual_input_fp8"(%io12) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io14, %w1_3 = "atlas.virtual_input_fp8"(%io13) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io15, %x3 = "atlas.virtual_input_fp8"(%io14) {index = 5 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_3 = "atlas.virtual_mxu_matmul"(%x3, %w0_3) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_3 = "atlas.virtual_vpu_unary"(%h0_3) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_3 = "atlas.virtual_pack_fp8"(%h1_3) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y3 = "atlas.virtual_mxu_matmul"(%h2_3, %w1_3) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io16 = "atlas.virtual_output_bf16"(%io15, %y3) {index = 3 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io17, %w0_4 = "atlas.virtual_input_fp8"(%io16) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io18, %w1_4 = "atlas.virtual_input_fp8"(%io17) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io19, %x4 = "atlas.virtual_input_fp8"(%io18) {index = 6 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_4 = "atlas.virtual_mxu_matmul"(%x4, %w0_4) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_4 = "atlas.virtual_vpu_unary"(%h0_4) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_4 = "atlas.virtual_pack_fp8"(%h1_4) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y4 = "atlas.virtual_mxu_matmul"(%h2_4, %w1_4) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io20 = "atlas.virtual_output_bf16"(%io19, %y4) {index = 4 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io21, %w0_5 = "atlas.virtual_input_fp8"(%io20) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io22, %w1_5 = "atlas.virtual_input_fp8"(%io21) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io23, %x5 = "atlas.virtual_input_fp8"(%io22) {index = 7 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_5 = "atlas.virtual_mxu_matmul"(%x5, %w0_5) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_5 = "atlas.virtual_vpu_unary"(%h0_5) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_5 = "atlas.virtual_pack_fp8"(%h1_5) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y5 = "atlas.virtual_mxu_matmul"(%h2_5, %w1_5) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io24 = "atlas.virtual_output_bf16"(%io23, %y5) {index = 5 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io25, %w0_6 = "atlas.virtual_input_fp8"(%io24) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io26, %w1_6 = "atlas.virtual_input_fp8"(%io25) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io27, %x6 = "atlas.virtual_input_fp8"(%io26) {index = 8 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_6 = "atlas.virtual_mxu_matmul"(%x6, %w0_6) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_6 = "atlas.virtual_vpu_unary"(%h0_6) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_6 = "atlas.virtual_pack_fp8"(%h1_6) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y6 = "atlas.virtual_mxu_matmul"(%h2_6, %w1_6) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io28 = "atlas.virtual_output_bf16"(%io27, %y6) {index = 6 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io29, %w0_7 = "atlas.virtual_input_fp8"(%io28) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io30, %w1_7 = "atlas.virtual_input_fp8"(%io29) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io31, %x7 = "atlas.virtual_input_fp8"(%io30) {index = 9 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_7 = "atlas.virtual_mxu_matmul"(%x7, %w0_7) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_7 = "atlas.virtual_vpu_unary"(%h0_7) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_7 = "atlas.virtual_pack_fp8"(%h1_7) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y7 = "atlas.virtual_mxu_matmul"(%h2_7, %w1_7) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io32 = "atlas.virtual_output_bf16"(%io31, %y7) {index = 7 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io33, %w0_8 = "atlas.virtual_input_fp8"(%io32) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io34, %w1_8 = "atlas.virtual_input_fp8"(%io33) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io35, %x8 = "atlas.virtual_input_fp8"(%io34) {index = 10 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_8 = "atlas.virtual_mxu_matmul"(%x8, %w0_8) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_8 = "atlas.virtual_vpu_unary"(%h0_8) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_8 = "atlas.virtual_pack_fp8"(%h1_8) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y8 = "atlas.virtual_mxu_matmul"(%h2_8, %w1_8) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io36 = "atlas.virtual_output_bf16"(%io35, %y8) {index = 8 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io37, %w0_9 = "atlas.virtual_input_fp8"(%io36) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io38, %w1_9 = "atlas.virtual_input_fp8"(%io37) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io39, %x9 = "atlas.virtual_input_fp8"(%io38) {index = 11 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_9 = "atlas.virtual_mxu_matmul"(%x9, %w0_9) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_9 = "atlas.virtual_vpu_unary"(%h0_9) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_9 = "atlas.virtual_pack_fp8"(%h1_9) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y9 = "atlas.virtual_mxu_matmul"(%h2_9, %w1_9) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io40 = "atlas.virtual_output_bf16"(%io39, %y9) {index = 9 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io41, %w0_10 = "atlas.virtual_input_fp8"(%io40) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io42, %w1_10 = "atlas.virtual_input_fp8"(%io41) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io43, %x10 = "atlas.virtual_input_fp8"(%io42) {index = 12 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_10 = "atlas.virtual_mxu_matmul"(%x10, %w0_10) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_10 = "atlas.virtual_vpu_unary"(%h0_10) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_10 = "atlas.virtual_pack_fp8"(%h1_10) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y10 = "atlas.virtual_mxu_matmul"(%h2_10, %w1_10) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io44 = "atlas.virtual_output_bf16"(%io43, %y10) {index = 10 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io45, %w0_11 = "atlas.virtual_input_fp8"(%io44) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io46, %w1_11 = "atlas.virtual_input_fp8"(%io45) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io47, %x11 = "atlas.virtual_input_fp8"(%io46) {index = 13 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_11 = "atlas.virtual_mxu_matmul"(%x11, %w0_11) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_11 = "atlas.virtual_vpu_unary"(%h0_11) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_11 = "atlas.virtual_pack_fp8"(%h1_11) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y11 = "atlas.virtual_mxu_matmul"(%h2_11, %w1_11) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io48 = "atlas.virtual_output_bf16"(%io47, %y11) {index = 11 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io49, %w0_12 = "atlas.virtual_input_fp8"(%io48) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io50, %w1_12 = "atlas.virtual_input_fp8"(%io49) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io51, %x12 = "atlas.virtual_input_fp8"(%io50) {index = 14 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_12 = "atlas.virtual_mxu_matmul"(%x12, %w0_12) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_12 = "atlas.virtual_vpu_unary"(%h0_12) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_12 = "atlas.virtual_pack_fp8"(%h1_12) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y12 = "atlas.virtual_mxu_matmul"(%h2_12, %w1_12) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io52 = "atlas.virtual_output_bf16"(%io51, %y12) {index = 12 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io53, %w0_13 = "atlas.virtual_input_fp8"(%io52) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io54, %w1_13 = "atlas.virtual_input_fp8"(%io53) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io55, %x13 = "atlas.virtual_input_fp8"(%io54) {index = 15 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_13 = "atlas.virtual_mxu_matmul"(%x13, %w0_13) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_13 = "atlas.virtual_vpu_unary"(%h0_13) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_13 = "atlas.virtual_pack_fp8"(%h1_13) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y13 = "atlas.virtual_mxu_matmul"(%h2_13, %w1_13) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io56 = "atlas.virtual_output_bf16"(%io55, %y13) {index = 13 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io57, %w0_14 = "atlas.virtual_input_fp8"(%io56) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io58, %w1_14 = "atlas.virtual_input_fp8"(%io57) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io59, %x14 = "atlas.virtual_input_fp8"(%io58) {index = 16 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_14 = "atlas.virtual_mxu_matmul"(%x14, %w0_14) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_14 = "atlas.virtual_vpu_unary"(%h0_14) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_14 = "atlas.virtual_pack_fp8"(%h1_14) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y14 = "atlas.virtual_mxu_matmul"(%h2_14, %w1_14) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io60 = "atlas.virtual_output_bf16"(%io59, %y14) {index = 14 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io61, %w0_15 = "atlas.virtual_input_fp8"(%io60) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io62, %w1_15 = "atlas.virtual_input_fp8"(%io61) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io63, %x15 = "atlas.virtual_input_fp8"(%io62) {index = 17 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0_15 = "atlas.virtual_mxu_matmul"(%x15, %w0_15) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1_15 = "atlas.virtual_vpu_unary"(%h0_15) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2_15 = "atlas.virtual_pack_fp8"(%h1_15) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y15 = "atlas.virtual_mxu_matmul"(%h2_15, %w1_15) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io64 = "atlas.virtual_output_bf16"(%io63, %y15) {index = 15 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io64 : !atlas.virtual_state
  }
}
