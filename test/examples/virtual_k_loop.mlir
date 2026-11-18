// Three 32-wide MXU0 reduction tiles; A and W advance by one FP8 tile.
// The loop carries completed BF16 data, with no MXU handle across an edge.
module {
 func.func @stream() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2416050176 : i64} {
  %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %zero = arith.constant 0 : i32
  %a = arith.constant -1879048192 : i32
  %w = arith.constant -1878982656 : i32
  %initial = "atlas.virtual_vpu_splat"() {bits = 0 : i32} : () -> !atlas.virtual_bf16
  cf.br ^loop(%s, %zero, %a, %w, %initial : !atlas.virtual_state, i32, i32, i32, !atlas.virtual_bf16)
 ^loop(%ls: !atlas.virtual_state, %k: i32, %ap: i32, %wp: i32, %acc: !atlas.virtual_bf16):
  %count = arith.constant 3 : i32
  %more = arith.cmpi ult, %k, %count : i32
  cf.cond_br %more, ^body(%ls : !atlas.virtual_state), ^exit(%ls, %acc : !atlas.virtual_state, !atlas.virtual_bf16)
 ^body(%bs: !atlas.virtual_state):
  %bytes = arith.constant 1024 : i32
  %s1, %x = "atlas.virtual_dma_load_fp8"(%bs, %ap, %bytes) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
  %s2, %y = "atlas.virtual_dma_load_fp8"(%s1, %wp, %bytes) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
  %s3, %weight = "atlas.virtual_mxu_load_weight"(%s2, %y) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
  %s4, %seed = "atlas.virtual_mxu_load_acc_bf16"(%s3, %acc) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
  %s5, %next = "atlas.virtual_mxu_accumulate"(%s4, %x, %weight, %seed) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
  %s6, %tile = "atlas.virtual_mxu_readout_bf16"(%s5, %next) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  %one = arith.constant 1 : i32
  %kn = arith.addi %k, %one : i32
  %an = arith.addi %ap, %bytes : i32
  %wn = arith.addi %wp, %bytes : i32
  cf.br ^loop(%s6, %kn, %an, %wn, %tile : !atlas.virtual_state, i32, i32, i32, !atlas.virtual_bf16)
 ^exit(%es: !atlas.virtual_state, %result: !atlas.virtual_bf16):
  %done = "atlas.virtual_output_bf16"(%es, %result) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  return %done : !atlas.virtual_state
 }
}
