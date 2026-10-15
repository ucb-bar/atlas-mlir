// A complete bounded virtual kernel: two ReLU steps, one input tile, one output.
// DRAM bases are ABI addresses; runtime tensor values are never compiler inputs.
module {
  func.func @two_relu_steps() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %t0 = "atlas.virtual_input_bf16"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    cf.br ^loop(%io1, %t0, %c0 : !atlas.virtual_state, !atlas.virtual_bf16, i32)
  ^loop(%loop_io: !atlas.virtual_state, %loop_tile: !atlas.virtual_bf16, %i: i32):
    %more = arith.cmpi slt, %i, %c2 : i32
    cf.cond_br %more, ^body(%loop_io, %loop_tile, %i : !atlas.virtual_state, !atlas.virtual_bf16, i32), ^exit(%loop_io, %loop_tile : !atlas.virtual_state, !atlas.virtual_bf16)
  ^body(%body_io: !atlas.virtual_state, %body_tile: !atlas.virtual_bf16, %j: i32):
    %next_tile = "atlas.virtual_vpu_unary"(%body_tile) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %next_i = arith.addi %j, %c1 : i32
    cf.br ^loop(%body_io, %next_tile, %next_i : !atlas.virtual_state, !atlas.virtual_bf16, i32)
  ^exit(%exit_io: !atlas.virtual_state, %exit_tile: !atlas.virtual_bf16):
    %io2 = "atlas.virtual_output_bf16"(%exit_io, %exit_tile) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io2 : !atlas.virtual_state
  }
}
