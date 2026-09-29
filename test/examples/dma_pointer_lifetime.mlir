// Two independent DMA source regions and two guarded outputs.
// x1 becomes B immediately after the first launch; waits precede VMEM reuse.
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
  %s10 = "atlas.alu_imm"(%s9) {kind = "addi", dst = 10 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s11 = "atlas.upper"(%s10) {kind = "lui", dst = 3 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s12 = "atlas.alu_imm"(%s11) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s13 = "atlas.dma"(%s12) {direction = "store", channel = 1 : i32, reg = 6 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s14 = "atlas.dma_wait"(%s13) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s15 = "atlas.alu_imm"(%s14) {kind = "addi", dst = 11 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s16 = "atlas.dma"(%s15) {direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s17 = "atlas.dma_wait"(%s16) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s18 = "atlas.upper"(%s17) {kind = "lui", dst = 4 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s19 = "atlas.alu_imm"(%s18) {kind = "addi", dst = 4 : i32, src = 4 : i32, immediate = 1536 : i32} : (!atlas.state) -> !atlas.state
  %s20 = "atlas.dma"(%s19) {direction = "store", channel = 1 : i32, reg = 6 : i32, dram = 4 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s21 = "atlas.dma_wait"(%s20) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s22 = "atlas.alu_imm"(%s21) {kind = "addi", dst = 12 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s23 = "atlas.alu_imm"(%s22) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s24 = "atlas.csr"(%s23) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s25 = "atlas.trap"(%s24) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
