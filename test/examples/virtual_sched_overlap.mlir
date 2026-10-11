// Two independent MXU chains written in naive order: every DMA load is awaited
// at once, both units compute, then both results are stored. The program is
// DMA-bound; --schedule-atlas-virtual hides compute under the transfers by
// moving each unit's work into issue-to-await intervals, and launches the
// next transfer before awaiting the current one, up to the transfers the
// verifier lets be pending at once.
module {
  func.func @two_unit_overlap() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415935488 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %x_addr = arith.constant -1879048192 : i32
    %w0_addr = arith.constant -1879047168 : i32
    %w1_addr = arith.constant -1879046144 : i32
    %b_addr = arith.constant -1879044096 : i32
    %y0_addr = arith.constant -1879040000 : i32
    %y1_addr = arith.constant -1879037952 : i32
    %fp8_size = arith.constant 1024 : i32
    %bf16_size = arith.constant 2048 : i32
    %io1, %x_event = "atlas.virtual_dma_load_fp8"(%io0, %x_addr, %fp8_size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %io2, %x = "atlas.virtual_dma_await_fp8"(%io1, %x_event) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3, %w0_event = "atlas.virtual_dma_load_fp8"(%io2, %w0_addr, %fp8_size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %io4, %w0 = "atlas.virtual_dma_await_fp8"(%io3, %w0_event) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io5, %w1_event = "atlas.virtual_dma_load_fp8"(%io4, %w1_addr, %fp8_size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %io6, %w1 = "atlas.virtual_dma_await_fp8"(%io5, %w1_event) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io7, %b_event = "atlas.virtual_dma_load_bf16"(%io6, %b_addr, %bf16_size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
    %io8, %b = "atlas.virtual_dma_await_bf16"(%io7, %b_event) : (!atlas.virtual_state, !atlas.virtual_dma_load_bf16) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io9, %weight0 = "atlas.virtual_mxu_load_weight"(%io8, %w0) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io10, %acc0 = "atlas.virtual_mxu_reset"(%io9, %x, %weight0) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io11, %h0 = "atlas.virtual_mxu_readout_bf16"(%io10, %acc0) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %h0_bias = "atlas.virtual_vpu_binary"(%h0, %b) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h0_relu = "atlas.virtual_vpu_unary"(%h0_bias) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %io12, %weight1 = "atlas.virtual_mxu_load_weight"(%io11, %w1) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io13, %acc1 = "atlas.virtual_mxu_reset"(%io12, %x, %weight1) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io14, %h1 = "atlas.virtual_mxu_readout_bf16"(%io13, %acc1) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %h1_bias = "atlas.virtual_vpu_binary"(%h1, %b) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %h1_relu = "atlas.virtual_vpu_unary"(%h1_bias) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %io15, %y0_event = "atlas.virtual_dma_store_bf16"(%io14, %h0_relu, %y0_addr, %bf16_size) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %io16 = "atlas.virtual_dma_wait"(%io15, %y0_event) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    %io17, %y1_event = "atlas.virtual_dma_store_bf16"(%io16, %h1_relu, %y1_addr, %bf16_size) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %io18 = "atlas.virtual_dma_wait"(%io17, %y1_event) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    return %io18 : !atlas.virtual_state
  }
}
