// Diagnostic VRECIP.BF16 stream for the selected AtlasCore. Delays follow the
// selected public vpu_unary_simple.S example and are not general timing bounds.
// DRAM: BF16 input halves at 0x90000000/0x90000400, output halves at
// 0x90000800/0x90000C00. Each half is 32 rows of 16 BF16 elements.
module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 28 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.alu_imm"(%s1) {kind = "addi", dst = 5 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.dma_config"(%s2) {channel = 0 : i32, base_reg = 5 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.alu_imm"(%s3) {kind = "addi", dst = 2 : i32, src = 0 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.alu_imm"(%s4) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.upper"(%s5) {kind = "lui", dst = 1 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s7 = "atlas.dma"(%s6) {direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s8 = "atlas.dma_wait"(%s7) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s9 = "atlas.vload"(%s8) {dst = 0 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s10 = "atlas.delay"(%s9) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s11 = "atlas.alu_imm"(%s10) {kind = "addi", dst = 7 : i32, src = 0 : i32, immediate = 256 : i32} : (!atlas.state) -> !atlas.state
  %s12 = "atlas.upper"(%s11) {kind = "lui", dst = 1 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s13 = "atlas.alu_imm"(%s12) {kind = "addi", dst = 1 : i32, src = 1 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s14 = "atlas.dma"(%s13) {direction = "load", channel = 0 : i32, reg = 7 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s15 = "atlas.dma_wait"(%s14) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s16 = "atlas.vload"(%s15) {dst = 1 : i32, base = 7 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s17 = "atlas.delay"(%s16) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s18 = "atlas.vpu_unary"(%s17) {kind = "recip", dst = 4 : i32, src = 0 : i32} : (!atlas.state) -> !atlas.state
  %s19 = "atlas.delay"(%s18) {cycles = 64 : i32} : (!atlas.state) -> !atlas.state
  %s20 = "atlas.alu_imm"(%s19) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 512 : i32} : (!atlas.state) -> !atlas.state
  %s21 = "atlas.vstore"(%s20) {src = 4 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s22 = "atlas.delay"(%s21) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s23 = "atlas.upper"(%s22) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s24 = "atlas.alu_imm"(%s23) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = -2048 : i32} : (!atlas.state) -> !atlas.state
  %s25 = "atlas.dma"(%s24) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s26 = "atlas.dma_wait"(%s25) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s27 = "atlas.alu_imm"(%s26) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 768 : i32} : (!atlas.state) -> !atlas.state
  %s28 = "atlas.vstore"(%s27) {src = 5 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s29 = "atlas.delay"(%s28) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s30 = "atlas.upper"(%s29) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s31 = "atlas.alu_imm"(%s30) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = -1024 : i32} : (!atlas.state) -> !atlas.state
  %s32 = "atlas.dma"(%s31) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s33 = "atlas.dma_wait"(%s32) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s34 = "atlas.alu_imm"(%s33) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s35 = "atlas.csr"(%s34) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s36 = "atlas.trap"(%s35) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
