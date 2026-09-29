// A taken branch must execute the next instruction and skip the following one.
// x10=0xAA only under the selected one-slot behavior; dbg0 receives x10-0xA9.
module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 10 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.branch"(%s1) {kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 6 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.alu_imm"(%s2) {kind = "addi", dst = 10 : i32, src = 0 : i32, immediate = 170 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.alu_imm"(%s3) {kind = "addi", dst = 10 : i32, src = 0 : i32, immediate = 187 : i32} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.alu_imm"(%s4) {kind = "addi", dst = 1 : i32, src = 10 : i32, immediate = -169 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.csr"(%s5) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s7 = "atlas.trap"(%s6) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
