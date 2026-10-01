// Pre-allocation example: %t values are logical BF16 tiles, never physical
// tensor-register numbers. %io values order external reads and writes;
// %s names are reserved for the post-allocation physical state chain.
module {
  %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %io1, %t0 = "atlas.virtual_input_bf16"(%io0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  %t1 = "atlas.virtual_vpu_unary"(%t0) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
  %t2 = "atlas.virtual_vpu_binary"(%t0, %t1) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
  %io2 = "atlas.virtual_output_bf16"(%io1, %t1) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
  %io3 = "atlas.virtual_output_bf16"(%io2, %t2) {index = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
}
