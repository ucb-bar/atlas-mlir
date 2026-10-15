// The loop backedge swaps two BF16 block arguments, forcing a parallel-copy cycle.
module {
  func.func @swap_once() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415927296 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %io1, %a0 = "atlas.virtual_input_bf16"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io2, %b0 = "atlas.virtual_input_bf16"(%io1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    cf.br ^loop(%io2, %a0, %b0, %c0 : !atlas.virtual_state, !atlas.virtual_bf16, !atlas.virtual_bf16, i32)
  ^loop(%loop_io: !atlas.virtual_state, %a: !atlas.virtual_bf16, %b: !atlas.virtual_bf16, %i: i32):
    %more = arith.cmpi slt, %i, %c1 : i32
    %next_i = arith.addi %i, %c1 : i32
    cf.cond_br %more, ^loop(%loop_io, %b, %a, %next_i : !atlas.virtual_state, !atlas.virtual_bf16, !atlas.virtual_bf16, i32), ^exit(%loop_io, %a : !atlas.virtual_state, !atlas.virtual_bf16)
  ^exit(%exit_io: !atlas.virtual_state, %selected: !atlas.virtual_bf16):
    %io3 = "atlas.virtual_output_bf16"(%exit_io, %selected) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %io3 : !atlas.virtual_state
  }
}
