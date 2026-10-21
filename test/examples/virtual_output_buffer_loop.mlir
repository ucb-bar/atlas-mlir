module {
  func.func @copy_tiles() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2416050176 : i64, atlas.output_dram_bytes = 6144 : i64} {
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %zero = arith.constant 0 : i32
    %input = arith.constant -1879048192 : i32
    %output = arith.constant -1878917120 : i32
    cf.br ^loop(%s, %zero, %input, %output : !atlas.virtual_state, i32, i32, i32)
  ^loop(%state: !atlas.virtual_state, %i: i32, %src: i32, %dst: i32):
    %count = arith.constant 3 : i32
    %more = arith.cmpi ult, %i, %count : i32
    cf.cond_br %more, ^body(%state : !atlas.virtual_state), ^done(%state : !atlas.virtual_state)
  ^body(%before: !atlas.virtual_state):
    %bytes = arith.constant 2048 : i32
    %s1, %tile = "atlas.virtual_dma_load_bf16"(%before, %src, %bytes) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s2 = "atlas.virtual_dma_store_bf16"(%s1, %tile, %dst, %bytes) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %one = arith.constant 1 : i32
    %next = arith.addi %i, %one : i32
    %sn = arith.addi %src, %bytes : i32
    %dn = arith.addi %dst, %bytes : i32
    cf.br ^loop(%s2, %next, %sn, %dn : !atlas.virtual_state, i32, i32, i32)
  ^done(%result: !atlas.virtual_state):
    return %result : !atlas.virtual_state
  }
}
