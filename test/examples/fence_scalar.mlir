// Canonical selected FENCE lies between pre/post scalar CSR markers.
// This bounded stream distinguishes a one-step FENCE from a frontend wait.
module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 7 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.csr"(%s1) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.fence"(%s2) : (!atlas.state) -> !atlas.state
  %s4 = "atlas.alu_imm"(%s3) {kind = "addi", dst = 2 : i32, src = 1 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.csr"(%s4) {kind = "rrw", dst = 0 : i32, source = 2 : i32, address = 3089 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.trap"(%s5) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
