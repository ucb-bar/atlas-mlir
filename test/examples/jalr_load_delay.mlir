// Selected-core diagnostic only. LW returns VMEM[0] = word target 11.
// DELAY 8 is a conservative schedule after LW, not a scalar completion wait.
// LLVM conversion must reject the memory-derived JALR target.
module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 2 : i32, src = 0 : i32, immediate = 11 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.scalar_store"(%s1) {kind = "sw", src = 2 : i32, base = 0 : i32, offset = 0 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.delay"(%s2) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.alu_imm"(%s3) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 8 : i32} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.scalar_load"(%s4) {kind = "lw", dst = 1 : i32, base = 0 : i32, offset = 0 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.delay"(%s5) {cycles = 8 : i32} : (!atlas.state) -> !atlas.state
  %s7 = "atlas.jump"(%s6) {kind = "jalr", dst = 10 : i32, base = 1 : i32, offset = 0 : i32} : (!atlas.state) -> !atlas.state
  %s8 = "atlas.alu_imm"(%s7) {kind = "addi", dst = 12 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s9 = "atlas.alu_imm"(%s8) {kind = "addi", dst = 12 : i32, src = 0 : i32, immediate = 2 : i32} : (!atlas.state) -> !atlas.state
  %s10 = "atlas.csr"(%s9) {kind = "rrw", dst = 0 : i32, source = 12 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s11 = "atlas.trap"(%s10) {kind = "ecall"} : (!atlas.state) -> !atlas.state
  %s12 = "atlas.alu_imm"(%s11) {kind = "addi", dst = 12 : i32, src = 0 : i32, immediate = 3 : i32} : (!atlas.state) -> !atlas.state
  %s13 = "atlas.csr"(%s12) {kind = "rrw", dst = 0 : i32, source = 12 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s14 = "atlas.trap"(%s13) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
