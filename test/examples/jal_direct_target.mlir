// Selected RTL: JAL at word 1 adds encoded byte displacement 8 >> 1,
// reaches word 5, and writes link word index 2 to x10. Word 2 is the
// single visible delay slot. Words 3 and 4 must not execute.
module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 12 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.jump"(%s1) {kind = "jal", dst = 10 : i32, base = 0 : i32, offset = 8 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.alu_imm"(%s2) {kind = "addi", dst = 12 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.alu_imm"(%s3) {kind = "addi", dst = 12 : i32, src = 0 : i32, immediate = 2 : i32} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.alu_imm"(%s4) {kind = "addi", dst = 12 : i32, src = 0 : i32, immediate = 4 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.alu_reg"(%s5) {kind = "add", dst = 11 : i32, lhs = 10 : i32, rhs = 12 : i32} : (!atlas.state) -> !atlas.state
  %s7 = "atlas.csr"(%s6) {kind = "rrw", dst = 0 : i32, source = 11 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s8 = "atlas.trap"(%s7) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
