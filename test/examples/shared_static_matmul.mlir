module {
  %s0 = "atlas.start"() : () -> !atlas.state
  // 0x3838 in BF16 storage produces two E4M3 1.0 bytes per element.
  %s1 = "atlas.vli"(%s0) {mode = "all", dst = 0 : i32, immediate = 14392 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.vli"(%s1) {mode = "all", dst = 4 : i32, immediate = 14392 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.mxu_push"(%s2) {kind = "weight_fp8", unit = 0 : i32, src = 4 : i32, slot = 0 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.mxu_matmul"(%s3) {unit = 0 : i32, src = 0 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.mxu_pop"(%s4) {format = "bf16", unit = 0 : i32, dst = 2 : i32, slot = 0 : i32, scale_reg = 0 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.alu_imm"(%s5) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s7 = "atlas.csr"(%s6) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s8 = "atlas.trap"(%s7) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
