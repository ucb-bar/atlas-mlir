// Two K tiles for MXU0. Weights are already in N-by-K orientation, so no XLU
// transform is required. Delays follow the selected public K-chain diagnostic;
// they are not qualified availability bounds for arbitrary schedules.
// Inputs A0,A1,W0,W1: DRAM +0,+0x400,+0x800,+0xC00, respectively.
// Output BF16 columns 0-15 and 16-31: DRAM +0x1000,+0x1400.
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
  %s12 = "atlas.upper"(%s11) {kind = "lui", dst = 1 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s13 = "atlas.alu_imm"(%s12) {kind = "addi", dst = 1 : i32, src = 1 : i32, immediate = -2048 : i32} : (!atlas.state) -> !atlas.state
  %s14 = "atlas.dma"(%s13) {direction = "load", channel = 0 : i32, reg = 7 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s15 = "atlas.dma_wait"(%s14) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s16 = "atlas.vload"(%s15) {dst = 4 : i32, base = 7 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s17 = "atlas.delay"(%s16) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s18 = "atlas.mxu_push"(%s17) {kind = "weight_fp8", unit = 0 : i32, src = 4 : i32, slot = 0 : i32} : (!atlas.state) -> !atlas.state
  %s19 = "atlas.delay"(%s18) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s20 = "atlas.mxu_matmul"(%s19) {unit = 0 : i32, src = 0 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false} : (!atlas.state) -> !atlas.state
  %s21 = "atlas.delay"(%s20) {cycles = 95 : i32} : (!atlas.state) -> !atlas.state
  %s22 = "atlas.alu_imm"(%s21) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 512 : i32} : (!atlas.state) -> !atlas.state
  %s23 = "atlas.upper"(%s22) {kind = "lui", dst = 1 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s24 = "atlas.alu_imm"(%s23) {kind = "addi", dst = 1 : i32, src = 1 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s25 = "atlas.dma"(%s24) {direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s26 = "atlas.dma_wait"(%s25) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s27 = "atlas.vload"(%s26) {dst = 2 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s28 = "atlas.delay"(%s27) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s29 = "atlas.alu_imm"(%s28) {kind = "addi", dst = 7 : i32, src = 0 : i32, immediate = 768 : i32} : (!atlas.state) -> !atlas.state
  %s30 = "atlas.upper"(%s29) {kind = "lui", dst = 1 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s31 = "atlas.alu_imm"(%s30) {kind = "addi", dst = 1 : i32, src = 1 : i32, immediate = -1024 : i32} : (!atlas.state) -> !atlas.state
  %s32 = "atlas.dma"(%s31) {direction = "load", channel = 0 : i32, reg = 7 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s33 = "atlas.dma_wait"(%s32) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s34 = "atlas.vload"(%s33) {dst = 6 : i32, base = 7 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s35 = "atlas.delay"(%s34) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s36 = "atlas.mxu_push"(%s35) {kind = "weight_fp8", unit = 0 : i32, src = 6 : i32, slot = 0 : i32} : (!atlas.state) -> !atlas.state
  %s37 = "atlas.delay"(%s36) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s38 = "atlas.mxu_matmul"(%s37) {unit = 0 : i32, src = 2 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = true} : (!atlas.state) -> !atlas.state
  %s39 = "atlas.delay"(%s38) {cycles = 95 : i32} : (!atlas.state) -> !atlas.state
  %s40 = "atlas.mxu_pop"(%s39) {format = "bf16", unit = 0 : i32, dst = 8 : i32, slot = 0 : i32, scale_reg = 0 : i32} : (!atlas.state) -> !atlas.state
  %s41 = "atlas.delay"(%s40) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s42 = "atlas.alu_imm"(%s41) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s43 = "atlas.vstore"(%s42) {src = 8 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s44 = "atlas.delay"(%s43) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s45 = "atlas.upper"(%s44) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s46 = "atlas.dma"(%s45) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s47 = "atlas.dma_wait"(%s46) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s48 = "atlas.alu_imm"(%s47) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1280 : i32} : (!atlas.state) -> !atlas.state
  %s49 = "atlas.vstore"(%s48) {src = 9 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s50 = "atlas.delay"(%s49) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s51 = "atlas.upper"(%s50) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s52 = "atlas.alu_imm"(%s51) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s53 = "atlas.dma"(%s52) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s54 = "atlas.dma_wait"(%s53) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s55 = "atlas.alu_imm"(%s54) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s56 = "atlas.csr"(%s55) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s57 = "atlas.trap"(%s56) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
