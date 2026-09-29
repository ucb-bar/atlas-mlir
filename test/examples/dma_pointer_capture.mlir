// Probe whether a queued DMA load uses the scalar address captured at launch.
// x1 is overwritten immediately after issue and before DMA.WAIT.
// DRAM A=0x90000000, B=0x90001000, output=0x90000400; VMEM address is a word.
module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 28 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.alu_imm"(%s1) {kind = "addi", dst = 5 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.dma_config"(%s2) {channel = 0 : i32, base_reg = 5 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.alu_imm"(%s3) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.upper"(%s4) {kind = "lui", dst = 1 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.alu_imm"(%s5) {kind = "addi", dst = 2 : i32, src = 0 : i32, immediate = 128 : i32} : (!atlas.state) -> !atlas.state
  %s7 = "atlas.dma"(%s6) {direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s8 = "atlas.upper"(%s7) {kind = "lui", dst = 1 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s9 = "atlas.dma_wait"(%s8) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s10 = "atlas.upper"(%s9) {kind = "lui", dst = 3 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s11 = "atlas.alu_imm"(%s10) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s12 = "atlas.dma"(%s11) {direction = "store", channel = 1 : i32, reg = 6 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s13 = "atlas.dma_wait"(%s12) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s14 = "atlas.alu_imm"(%s13) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s15 = "atlas.csr"(%s14) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s16 = "atlas.trap"(%s15) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
