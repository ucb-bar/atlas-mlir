// Combined attention + MLP tile, standing in for the combined program until it
// lands: virtual_attention_tile.mlir -> unit-scale BF16-to-FP8 pack ->
// the two-layer MLP of virtual_fp8_two_layer_mlp.mlir. FP8 Q, K^T, V, W0, W1
// at 0x90000000 + 0x400 * i; the BF16 output at 0x90004000.
module {
  func.func @attention_mlp_tile() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415935488 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %q_addr = arith.constant -1879048192 : i32
    %k_addr = arith.constant -1879047168 : i32
    %v_addr = arith.constant -1879046144 : i32
    %w0_addr = arith.constant -1879045120 : i32
    %w1_addr = arith.constant -1879044096 : i32
    %out_addr = arith.constant -1879031808 : i32
    %fp8_size = arith.constant 1024 : i32
    %bf16_size = arith.constant 2048 : i32
    %io1, %q_event = "atlas.virtual_dma_load_fp8"(%io0, %q_addr, %fp8_size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %io2, %q = "atlas.virtual_dma_await_fp8"(%io1, %q_event) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3, %k_event = "atlas.virtual_dma_load_fp8"(%io2, %k_addr, %fp8_size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %io4, %k = "atlas.virtual_dma_await_fp8"(%io3, %k_event) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io5, %v_event = "atlas.virtual_dma_load_fp8"(%io4, %v_addr, %fp8_size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %io6, %v = "atlas.virtual_dma_await_fp8"(%io5, %v_event) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io7, %k_weight = "atlas.virtual_mxu_load_weight"(%io6, %k) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io8, %scores_acc = "atlas.virtual_mxu_reset"(%io7, %q, %k_weight) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io9, %s = "atlas.virtual_mxu_readout_bf16"(%io8, %scores_acc) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max = "atlas.virtual_vpu_reduce"(%s) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted = "atlas.virtual_vpu_binary"(%s, %max) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled = "atlas.virtual_vpu_binary"(%shifted, %scale) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e = "atlas.virtual_vpu_unary"(%scaled) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum = "atlas.virtual_vpu_reduce"(%e) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse = "atlas.virtual_vpu_unary"(%sum) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p = "atlas.virtual_vpu_binary"(%e, %inverse) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8 = "atlas.virtual_pack_fp8"(%p) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io10, %v_weight = "atlas.virtual_mxu_load_weight"(%io9, %v) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io11, %attn_acc = "atlas.virtual_mxu_reset"(%io10, %p8, %v_weight) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io12, %attn = "atlas.virtual_mxu_readout_bf16"(%io11, %attn_acc) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %x = "atlas.virtual_pack_fp8"(%attn) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io13, %w0_event = "atlas.virtual_dma_load_fp8"(%io12, %w0_addr, %fp8_size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %io14, %w0 = "atlas.virtual_dma_await_fp8"(%io13, %w0_event) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io15, %w1_event = "atlas.virtual_dma_load_fp8"(%io14, %w1_addr, %fp8_size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %io16, %w1 = "atlas.virtual_dma_await_fp8"(%io15, %w1_event) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %h0 = "atlas.virtual_mxu_matmul"(%x, %w0) {unit = 0 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %h1 = "atlas.virtual_vpu_unary"(%h0) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h2 = "atlas.virtual_pack_fp8"(%h1) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %y = "atlas.virtual_mxu_matmul"(%h2, %w1) {unit = 1 : i32} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %io17, %out_event = "atlas.virtual_dma_store_bf16"(%io16, %y, %out_addr, %bf16_size) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %io18 = "atlas.virtual_dma_wait"(%io17, %out_event) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    return %io18 : !atlas.virtual_state
  }
}
