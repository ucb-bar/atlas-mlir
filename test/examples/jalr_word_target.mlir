// JALR target = x1 + (-1) = odd word index 5, without RISC-V byte-PC masking.
// Selected link x10 = jump word index + 1 = 2; one delay slot sets x12 = 1.
// Thus x11 and dbg0 become 3. Wrong target alignment or delay changes that.
module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 6 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.jump"(%s1) {kind = "jalr", dst = 10 : i32, base = 1 : i32, offset = -1 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.alu_imm"(%s2) {kind = "addi", dst = 12 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.alu_imm"(%s3) {kind = "addi", dst = 12 : i32, src = 0 : i32, immediate = 2 : i32} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.alu_imm"(%s4) {kind = "addi", dst = 12 : i32, src = 0 : i32, immediate = 3 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.alu_reg"(%s5) {kind = "add", dst = 11 : i32, lhs = 10 : i32, rhs = 12 : i32} : (!atlas.state) -> !atlas.state
  %s7 = "atlas.csr"(%s6) {kind = "rrw", dst = 0 : i32, source = 11 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s8 = "atlas.trap"(%s7) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
