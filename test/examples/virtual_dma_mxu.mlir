// Explicit FP8 DMA loads feed a resident MXU chain and explicit FP8 DMA output.
module {
  func.func @dma_mxu() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %x_addr = arith.constant -1879048192 : i32
    %w_addr = arith.constant -1879047168 : i32
    %out_addr = arith.constant -1879044096 : i32
    %size = arith.constant 1024 : i32
    %scale = "atlas.virtual_scale_constant"() {code = 129 : i32} : () -> !atlas.virtual_scale
    %io1, %x_event = "atlas.virtual_dma_load_fp8"(%io0, %x_addr, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %io2, %x = "atlas.virtual_dma_await_fp8"(%io1, %x_event) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3, %w_event = "atlas.virtual_dma_load_fp8"(%io2, %w_addr, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_fp8)
    %io4, %w = "atlas.virtual_dma_await_fp8"(%io3, %w_event) : (!atlas.virtual_state, !atlas.virtual_dma_load_fp8) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s0, %weight = "atlas.virtual_mxu_load_weight"(%io4, %w) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %s1, %a0 = "atlas.virtual_mxu_reset"(%s0, %x, %weight) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s2, %a1 = "atlas.virtual_mxu_accumulate"(%s1, %x, %weight, %a0) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s3, %packed = "atlas.virtual_mxu_readout_fp8"(%s2, %a1, %scale) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>, !atlas.virtual_scale) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s4, %store_event = "atlas.virtual_dma_store_fp8"(%s3, %packed, %out_addr, %size) : (!atlas.virtual_state, !atlas.virtual_fp8, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %done = "atlas.virtual_dma_wait"(%s4, %store_event) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    return %done : !atlas.virtual_state
  }
}
