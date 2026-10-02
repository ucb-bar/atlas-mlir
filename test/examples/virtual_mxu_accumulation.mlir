// Explicit MXU handles describe resident weights and destructive accumulator
// updates. Both units may have an active chain; readout consumes each chain.
// Explicit lowering assigns weight/accumulator slot zero on each unit.
module {
  func.func @two_unit_accumulation() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %x = "atlas.virtual_input_fp8"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2, %w = "atlas.virtual_input_fp8"(%io1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3, %weight0 = "atlas.virtual_mxu_load_weight"(%io2, %w) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io4, %weight1 = "atlas.virtual_mxu_load_weight"(%io3, %w) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io5, %acc0 = "atlas.virtual_mxu_reset"(%io4, %x, %weight0) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6, %acc1 = "atlas.virtual_mxu_reset"(%io5, %x, %weight1) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io7, %next0 = "atlas.virtual_mxu_accumulate"(%io6, %x, %weight0, %acc0) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io8, %next1 = "atlas.virtual_mxu_accumulate"(%io7, %x, %weight1, %acc1) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9, %y0 = "atlas.virtual_mxu_readout_bf16"(%io8, %next0) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10, %y1 = "atlas.virtual_mxu_readout_bf16"(%io9, %next1) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io11 = "atlas.virtual_output_bf16"(%io10, %y0) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %io12 = "atlas.virtual_output_bf16"(%io11, %y1) {index = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io12 : !atlas.virtual_state
  }
}
