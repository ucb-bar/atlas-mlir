// Hand-authored physical stream corresponding to the selected public DMA loopback.
// Scalar addresses are byte addresses; the DMA tensor address is a VMEM word.
module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 28 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.alu_imm"(%s1) {kind = "addi", dst = 5 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.dma_config"(%s2) {channel = 0 : i32, base_reg = 5 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.alu_imm"(%s3) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.upper"(%s4) {kind = "lui", dst = 1 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.alu_imm"(%s5) {kind = "addi", dst = 2 : i32, src = 0 : i32, immediate = 128 : i32} : (!atlas.state) -> !atlas.state
  %s7 = "atlas.dma"(%s6) {direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s8 = "atlas.dma_wait"(%s7) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s9 = "atlas.upper"(%s8) {kind = "lui", dst = 3 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s10 = "atlas.alu_imm"(%s9) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s11 = "atlas.dma"(%s10) {direction = "store", channel = 1 : i32, reg = 6 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s12 = "atlas.dma_wait"(%s11) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s13 = "atlas.alu_imm"(%s12) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s14 = "atlas.csr"(%s13) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s15 = "atlas.trap"(%s14) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
