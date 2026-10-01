// Change %choose from 0 to 1 to execute the other physical CFG arm.
module {
  func.func @choose_tile() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %t0 = "atlas.virtual_input_bf16"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %choose = arith.constant 0 : i1
    cf.cond_br %choose, ^left(%io1, %t0 : !atlas.virtual_state, !atlas.virtual_bf16), ^right(%io1, %t0 : !atlas.virtual_state, !atlas.virtual_bf16)
  ^left(%left_io: !atlas.virtual_state, %left_tile: !atlas.virtual_bf16):
    %left_result = "atlas.virtual_vpu_unary"(%left_tile) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    cf.br ^join(%left_io, %left_result : !atlas.virtual_state, !atlas.virtual_bf16)
  ^right(%right_io: !atlas.virtual_state, %right_tile: !atlas.virtual_bf16):
    %right_result = "atlas.virtual_vpu_unary"(%right_tile) {kind = "mov"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    cf.br ^join(%right_io, %right_result : !atlas.virtual_state, !atlas.virtual_bf16)
  ^join(%joined_io: !atlas.virtual_state, %joined_tile: !atlas.virtual_bf16):
    %io2 = "atlas.virtual_output_bf16"(%joined_io, %joined_tile) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io2 : !atlas.virtual_state
  }
}
