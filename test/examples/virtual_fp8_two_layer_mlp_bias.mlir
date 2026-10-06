// Fixed 32x32 quantized MLP tile with column-dependent BF16 biases.
// B0 and B1 are already broadcast to 32x32 BF16 boundary tiles. The virtual
// VPU adds happen after each MXU readout, matching Linear's bias placement.
// This is a prepared target-numeric graph, not a PyTorch capture.
module {
  func.func @two_layer_mlp_bias() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415931392 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %x = "atlas.virtual_input_fp8"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2, %w0 = "atlas.virtual_input_fp8"(%io1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3, %w1 = "atlas.virtual_input_fp8"(%io2) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4, %b0 = "atlas.virtual_input_bf16"(%io3) {index = 3 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io5, %b1 = "atlas.virtual_input_bf16"(%io4) {index = 4 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %h0 = "atlas.virtual_mxu_matmul"(%x, %w0) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h0b = "atlas.virtual_vpu_binary"(%h0, %b0) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h1 = "atlas.virtual_vpu_unary"(%h0b) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2 = "atlas.virtual_pack_fp8"(%h1) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %h3 = "atlas.virtual_mxu_matmul"(%h2, %w1) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %y = "atlas.virtual_vpu_binary"(%h3, %b1) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %io6 = "atlas.virtual_output_bf16"(%io5, %y) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io6 : !atlas.virtual_state
  }
}
