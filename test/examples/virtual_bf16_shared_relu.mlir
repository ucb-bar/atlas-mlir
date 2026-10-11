module {
  func.func @shared_relu() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64} {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %x = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %y = "atlas.virtual_vpu_unary"(%x) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    // Keep x live across ReLU so clobbering its storage changes output 0.
    %s2 = "atlas.virtual_output_bf16"(%s1, %x) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %s3 = "atlas.virtual_output_bf16"(%s2, %y) {index = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s3 : !atlas.virtual_state
  }
}
