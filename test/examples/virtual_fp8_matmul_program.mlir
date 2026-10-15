// One complete selected MXU tile from virtual SSA to physical Atlas words.
// Input slots are 2 KiB apart; each FP8 tile occupies the first 1 KiB.
// Weight bytes encode W[output_column, reduction_index].
module {
  func.func @one_fp8_matmul() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %x = "atlas.virtual_input_fp8"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2, %w = "atlas.virtual_input_fp8"(%io1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %y = "atlas.virtual_mxu_matmul"(%x, %w) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io3 = "atlas.virtual_output_bf16"(%io2, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io3 : !atlas.virtual_state
  }
}
