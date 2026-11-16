module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.vli"(%s0) {mode = "all", dst = 0 : i32, immediate = 49024 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.vpu_unary"(%s1) {kind = "relu", dst = 2 : i32, src = 0 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.vli"(%s2) {mode = "all", dst = 4 : i32, immediate = 16256 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.vpu_binary"(%s3) {kind = "add", dst = 6 : i32, lhs = 2 : i32, rhs = 4 : i32} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.alu_imm"(%s4) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.csr"(%s5) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s7 = "atlas.trap"(%s6) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
