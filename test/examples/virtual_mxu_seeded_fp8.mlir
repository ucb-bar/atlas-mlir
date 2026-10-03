// BF16 seed on MXU0, FP8 readout, then FP8 seed on MXU1.
module {
  func.func @seeded_fp8() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415935488 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %scale = "atlas.virtual_scale_constant"() {code = 129 : i32} : () -> !atlas.virtual_scale
    %io1, %seed = "atlas.virtual_input_bf16"(%io0) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io2, %x = "atlas.virtual_input_fp8"(%io1) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3, %w = "atlas.virtual_input_fp8"(%io2) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s0, %w0 = "atlas.virtual_mxu_load_weight"(%io3, %w) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %s1, %a0 = "atlas.virtual_mxu_load_acc_bf16"(%s0, %seed) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s2, %a1 = "atlas.virtual_mxu_accumulate"(%s1, %x, %w0, %a0) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s3, %packed = "atlas.virtual_mxu_readout_fp8"(%s2, %a1, %scale) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>, !atlas.virtual_scale) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s4, %w1 = "atlas.virtual_mxu_load_weight"(%s3, %w) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %s5, %a2 = "atlas.virtual_mxu_load_acc_fp8"(%s4, %packed) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %s6, %a3 = "atlas.virtual_mxu_accumulate"(%s5, %x, %w1, %a2) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %s7, %y = "atlas.virtual_mxu_readout_bf16"(%s6, %a3) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %out = "atlas.virtual_output_bf16"(%s7, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %out : !atlas.virtual_state
  }
}
