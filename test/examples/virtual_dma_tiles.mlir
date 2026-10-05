// DMA events own private logical tile storage until their matching completion.
module {
  func.func @dma_bf16_copy() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415923200 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %base = arith.constant -2147483648 : i32
    %offset = arith.constant 2048 : i32
    %source = arith.addi %base, %offset : i32
    %destination = arith.addi %source, %offset : i32
    %size = arith.constant 2048 : i32
    %io1, %load = "atlas.virtual_dma_load_bf16"(%io0, %source, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_load_bf16)
    %io2, %tile = "atlas.virtual_dma_await_bf16"(%io1, %load) : (!atlas.virtual_state, !atlas.virtual_dma_load_bf16) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %result = "atlas.virtual_vpu_unary"(%tile) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %io3, %store = "atlas.virtual_dma_store_bf16"(%io2, %result, %destination, %size) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_dma_store)
    %io4 = "atlas.virtual_dma_wait"(%io3, %store) : (!atlas.virtual_state, !atlas.virtual_dma_store) -> !atlas.virtual_state
    return %io4 : !atlas.virtual_state
  }
}
