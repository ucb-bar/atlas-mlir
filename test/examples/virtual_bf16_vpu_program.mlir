// Preserve a shared input across MOV, ADD, and ReLU before publishing four tiles.
module {
  func.func @vpu_shared_input() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415935488 : i64} {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %a = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s2, %b = "atlas.virtual_input_bf16"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %moved = "atlas.virtual_vpu_unary"(%a) {kind = "mov"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum = "atlas.virtual_vpu_binary"(%moved, %b) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %positive = "atlas.virtual_vpu_unary"(%sum) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %s3 = "atlas.virtual_output_bf16"(%s2, %a) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %s4 = "atlas.virtual_output_bf16"(%s3, %moved) {index = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %s5 = "atlas.virtual_output_bf16"(%s4, %sum) {index = 2 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %s6 = "atlas.virtual_output_bf16"(%s5, %positive) {index = 3 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s6 : !atlas.virtual_state
  }
}
