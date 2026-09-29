// Two BF16 input pairs, one BF16 VSUB output pair. Diagnostic waits follow
// selected public vpu_binary.S; they are not general availability bounds.
// DRAM A: +0/+0x400, B: +0x800/+0xc00, result: +0x1000/+0x1400.
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
  %s11 = "atlas.alu_imm"(%s10) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 256 : i32} : (!atlas.state) -> !atlas.state
  %s12 = "atlas.upper"(%s11) {kind = "lui", dst = 1 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s13 = "atlas.alu_imm"(%s12) {kind = "addi", dst = 1 : i32, src = 1 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s14 = "atlas.dma"(%s13) {direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s15 = "atlas.dma_wait"(%s14) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s16 = "atlas.vload"(%s15) {dst = 1 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s17 = "atlas.delay"(%s16) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s18 = "atlas.alu_imm"(%s17) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 512 : i32} : (!atlas.state) -> !atlas.state
  %s19 = "atlas.upper"(%s18) {kind = "lui", dst = 1 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s20 = "atlas.alu_imm"(%s19) {kind = "addi", dst = 1 : i32, src = 1 : i32, immediate = -2048 : i32} : (!atlas.state) -> !atlas.state
  %s21 = "atlas.dma"(%s20) {direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s22 = "atlas.dma_wait"(%s21) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s23 = "atlas.vload"(%s22) {dst = 2 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s24 = "atlas.delay"(%s23) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s25 = "atlas.alu_imm"(%s24) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 768 : i32} : (!atlas.state) -> !atlas.state
  %s26 = "atlas.upper"(%s25) {kind = "lui", dst = 1 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s27 = "atlas.alu_imm"(%s26) {kind = "addi", dst = 1 : i32, src = 1 : i32, immediate = -1024 : i32} : (!atlas.state) -> !atlas.state
  %s28 = "atlas.dma"(%s27) {direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s29 = "atlas.dma_wait"(%s28) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s30 = "atlas.vload"(%s29) {dst = 3 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s31 = "atlas.delay"(%s30) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s32 = "atlas.vpu_binary"(%s31) {kind = "sub", dst = 4 : i32, lhs = 0 : i32, rhs = 2 : i32} : (!atlas.state) -> !atlas.state
  %s33 = "atlas.delay"(%s32) {cycles = 64 : i32} : (!atlas.state) -> !atlas.state
  %s34 = "atlas.alu_imm"(%s33) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s35 = "atlas.vstore"(%s34) {src = 4 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s36 = "atlas.delay"(%s35) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s37 = "atlas.upper"(%s36) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s38 = "atlas.dma"(%s37) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s39 = "atlas.dma_wait"(%s38) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s40 = "atlas.alu_imm"(%s39) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1280 : i32} : (!atlas.state) -> !atlas.state
  %s41 = "atlas.vstore"(%s40) {src = 5 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s42 = "atlas.delay"(%s41) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s43 = "atlas.upper"(%s42) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s44 = "atlas.alu_imm"(%s43) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s45 = "atlas.dma"(%s44) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s46 = "atlas.dma_wait"(%s45) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s47 = "atlas.alu_imm"(%s46) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s48 = "atlas.csr"(%s47) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s49 = "atlas.trap"(%s48) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
