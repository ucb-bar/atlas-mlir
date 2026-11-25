// Counted prefetch loop with alternating DMA channels.
module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 5 : i32, src = 0 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.upper"(%s1) {kind = "lui", dst = 8 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.upper"(%s2) {kind = "lui", dst = 15 : i32, immediate = 490 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.alu_imm"(%s3) {kind = "addi", dst = 19 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.alu_reg"(%s4) {kind = "add", dst = 20 : i32, lhs = 15 : i32, rhs = 0 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.alu_imm"(%s5) {kind = "addi", dst = 21 : i32, src = 0 : i32, immediate = 5 : i32} : (!atlas.state) -> !atlas.state
  %s7 = "atlas.alu_reg"(%s6) {kind = "add", dst = 20 : i32, lhs = 20 : i32, rhs = 8 : i32} : (!atlas.state) -> !atlas.state
  %s8 = "atlas.dma"(%s7) {direction = "load", channel = 0 : i32, reg = 19 : i32, dram = 20 : i32, size = 5 : i32} : (!atlas.state) -> !atlas.state
  %s9 = "atlas.alu_reg"(%s8) {kind = "sub", dst = 20 : i32, lhs = 20 : i32, rhs = 8 : i32} : (!atlas.state) -> !atlas.state
  %s10 = "atlas.dma_wait"(%s9) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s11 = "atlas.alu_imm"(%s10) {kind = "addi", dst = 24 : i32, src = 19 : i32, immediate = 256 : i32} : (!atlas.state) -> !atlas.state
  %s12 = "atlas.alu_imm"(%s11) {kind = "addi", dst = 20 : i32, src = 20 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s13 = "atlas.alu_imm"(%s12) {kind = "addi", dst = 21 : i32, src = 21 : i32, immediate = -1 : i32} : (!atlas.state) -> !atlas.state
  %s14 = "atlas.branch"(%s13) {kind = "beq", lhs = 21 : i32, rhs = 0 : i32, offset_bytes = 30 : i32} : (!atlas.state) -> !atlas.state
  %s15 = "atlas.alu_imm"(%s14) {kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s16 = "atlas.alu_reg"(%s15) {kind = "add", dst = 20 : i32, lhs = 20 : i32, rhs = 8 : i32} : (!atlas.state) -> !atlas.state
  %s17 = "atlas.dma"(%s16) {direction = "load", channel = 1 : i32, reg = 24 : i32, dram = 20 : i32, size = 5 : i32} : (!atlas.state) -> !atlas.state
  %s18 = "atlas.alu_reg"(%s17) {kind = "sub", dst = 20 : i32, lhs = 20 : i32, rhs = 8 : i32} : (!atlas.state) -> !atlas.state
  %s19 = "atlas.vload"(%s18) {dst = 1 : i32, base = 19 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s20 = "atlas.delay"(%s19) {cycles = 34 : i32} : (!atlas.state) -> !atlas.state
  %s21 = "atlas.xlu_transpose"(%s20) {dst = 1 : i32, src = 1 : i32} : (!atlas.state) -> !atlas.state
  %s22 = "atlas.delay"(%s21) {cycles = 66 : i32} : (!atlas.state) -> !atlas.state
  %s23 = "atlas.vstore"(%s22) {src = 1 : i32, base = 19 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s24 = "atlas.delay"(%s23) {cycles = 34 : i32} : (!atlas.state) -> !atlas.state
  %s25 = "atlas.dma_wait"(%s24) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s26 = "atlas.alu_reg"(%s25) {kind = "add", dst = 19 : i32, lhs = 24 : i32, rhs = 0 : i32} : (!atlas.state) -> !atlas.state
  %s27 = "atlas.jump"(%s26) {kind = "jal", dst = 0 : i32, base = 0 : i32, offset = -32 : i32} : (!atlas.state) -> !atlas.state
  %s28 = "atlas.alu_imm"(%s27) {kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s29 = "atlas.vload"(%s28) {dst = 1 : i32, base = 19 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s30 = "atlas.delay"(%s29) {cycles = 34 : i32} : (!atlas.state) -> !atlas.state
  %s31 = "atlas.xlu_transpose"(%s30) {dst = 1 : i32, src = 1 : i32} : (!atlas.state) -> !atlas.state
  %s32 = "atlas.delay"(%s31) {cycles = 66 : i32} : (!atlas.state) -> !atlas.state
  %s33 = "atlas.vstore"(%s32) {src = 1 : i32, base = 19 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s34 = "atlas.delay"(%s33) {cycles = 34 : i32} : (!atlas.state) -> !atlas.state
  %s35 = "atlas.alu_imm"(%s34) {kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s36 = "atlas.trap"(%s35) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
