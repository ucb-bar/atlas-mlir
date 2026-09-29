// Taken BEQ: dbg0 write at the first post-branch position must execute;
// dbg1 write at the second must be skipped on selected one-slot AtlasCore.
// Backward BLT repeats twice and falls through on its third evaluation.
// Encoded branch offsets are signed bytes; selected PC is a word index and
// applies offset >> 1. This source is diagnostic, not a general loop lowering.
module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 10 : i32, src = 0 : i32, immediate = 17 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.alu_imm"(%s1) {kind = "addi", dst = 11 : i32, src = 0 : i32, immediate = 34 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.branch"(%s2) {kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = 6 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.csr"(%s3) {kind = "rrw", dst = 0 : i32, source = 10 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.csr"(%s4) {kind = "rrw", dst = 0 : i32, source = 11 : i32, address = 3089 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.alu_imm"(%s5) {kind = "addi", dst = 12 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s7 = "atlas.alu_imm"(%s6) {kind = "addi", dst = 13 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s8 = "atlas.alu_imm"(%s7) {kind = "addi", dst = 14 : i32, src = 0 : i32, immediate = 3 : i32} : (!atlas.state) -> !atlas.state
  %s9 = "atlas.alu_imm"(%s8) {kind = "addi", dst = 15 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s10 = "atlas.alu_imm"(%s9) {kind = "addi", dst = 12 : i32, src = 12 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s11 = "atlas.branch"(%s10) {kind = "blt", lhs = 12 : i32, rhs = 14 : i32, offset_bytes = -2 : i32} : (!atlas.state) -> !atlas.state
  %s12 = "atlas.alu_imm"(%s11) {kind = "addi", dst = 13 : i32, src = 13 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s13 = "atlas.alu_imm"(%s12) {kind = "addi", dst = 15 : i32, src = 15 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s14 = "atlas.trap"(%s13) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
