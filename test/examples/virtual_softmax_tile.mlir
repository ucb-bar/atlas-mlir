// One attention softmax tile over BF16 scores S, row by row:
// P = exp2((S - rowmax(S)) * c) / rowsum(...), with c = 0x3E83 (about
// log2(e) / sqrt(32)), as in handoff_attention_tile.mlir.
module {
  func.func @softmax_tile() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %s = "atlas.virtual_input_bf16"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max = "atlas.virtual_vpu_reduce"(%s) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted = "atlas.virtual_vpu_binary"(%s, %max) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled = "atlas.virtual_vpu_binary"(%shifted, %scale) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e = "atlas.virtual_vpu_unary"(%scaled) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum = "atlas.virtual_vpu_reduce"(%e) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse = "atlas.virtual_vpu_unary"(%sum) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p = "atlas.virtual_vpu_binary"(%e, %inverse) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %io2 = "atlas.virtual_output_bf16"(%io1, %p) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io2 : !atlas.virtual_state
  }
}
