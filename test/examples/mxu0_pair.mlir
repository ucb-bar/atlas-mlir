// Hand-authored MXU0 variant that stores both BF16 output register halves.
// The delays are diagnostic timings, not general target availability bounds.
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
  %s9 = "atlas.alu_imm"(%s8) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s10 = "atlas.vload"(%s9) {dst = 4 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s11 = "atlas.alu_imm"(%s10) {kind = "addi", dst = 7 : i32, src = 0 : i32, immediate = 256 : i32} : (!atlas.state) -> !atlas.state
  %s12 = "atlas.upper"(%s11) {kind = "lui", dst = 1 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s13 = "atlas.alu_imm"(%s12) {kind = "addi", dst = 1 : i32, src = 1 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s14 = "atlas.dma"(%s13) {direction = "load", channel = 0 : i32, reg = 7 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s15 = "atlas.dma_wait"(%s14) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s16 = "atlas.alu_imm"(%s15) {kind = "addi", dst = 7 : i32, src = 0 : i32, immediate = 256 : i32} : (!atlas.state) -> !atlas.state
  %s17 = "atlas.vload"(%s16) {dst = 0 : i32, base = 7 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s18 = "atlas.delay"(%s17) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s19 = "atlas.mxu_push"(%s18) {kind = "weight_fp8", unit = 0 : i32, src = 4 : i32, slot = 0 : i32} : (!atlas.state) -> !atlas.state
  %s20 = "atlas.delay"(%s19) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s21 = "atlas.mxu_matmul"(%s20) {unit = 0 : i32, src = 0 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false} : (!atlas.state) -> !atlas.state
  %s22 = "atlas.delay"(%s21) {cycles = 95 : i32} : (!atlas.state) -> !atlas.state
  %s23 = "atlas.mxu_pop"(%s22) {format = "bf16", unit = 0 : i32, dst = 2 : i32, slot = 0 : i32, scale_reg = 0 : i32} : (!atlas.state) -> !atlas.state
  %s24 = "atlas.delay"(%s23) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s25 = "atlas.alu_imm"(%s24) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 512 : i32} : (!atlas.state) -> !atlas.state
  %s26 = "atlas.vstore"(%s25) {src = 2 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s27 = "atlas.delay"(%s26) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s28 = "atlas.alu_imm"(%s27) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 512 : i32} : (!atlas.state) -> !atlas.state
  %s29 = "atlas.upper"(%s28) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s30 = "atlas.alu_imm"(%s29) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = -2048 : i32} : (!atlas.state) -> !atlas.state
  %s31 = "atlas.dma"(%s30) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s32 = "atlas.dma_wait"(%s31) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s33 = "atlas.alu_imm"(%s32) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 768 : i32} : (!atlas.state) -> !atlas.state
  %s34 = "atlas.vstore"(%s33) {src = 3 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s35 = "atlas.delay"(%s34) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s36 = "atlas.alu_imm"(%s35) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 768 : i32} : (!atlas.state) -> !atlas.state
  %s37 = "atlas.upper"(%s36) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s38 = "atlas.dma"(%s37) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s39 = "atlas.dma_wait"(%s38) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s40 = "atlas.alu_imm"(%s39) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s41 = "atlas.csr"(%s40) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s42 = "atlas.trap"(%s41) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
