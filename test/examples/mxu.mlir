module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.vload"(%s0) {dst = 0 : i32, base = 1 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.mxu_push"(%s1) {kind = "weight_fp8", unit = 0 : i32, src = 0 : i32, slot = 0 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.mxu_matmul"(%s2) {unit = 0 : i32, src = 1 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.mxu_pop"(%s3) {format = "bf16", unit = 0 : i32, dst = 2 : i32, slot = 0 : i32, scale_reg = 0 : i32} : (!atlas.state) -> !atlas.state
}
