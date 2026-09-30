// Diagnostic two-layer MLP instruction sequence: MXU0 -> BF16 ReLU ->
// E8M0 pack at unit scale (SELI 127) -> MXU1. No bias.
// This is not a correct general 32x32 MLP: PACK changes the physical row
// layout before MXU1. See docs/llvm-handoff-examples.md.
// Hand-authored fixed 32x32 machine-stage diagnostic for the selected RTL.
// Inputs: three 1024-byte FP8 tiles at DRAM 0x90000000, +0x400, +0x800.
// Output: two 1024-byte BF16 halves at DRAM +0xC00 and +0x1000.
// Delays are public-example-derived diagnostic spacing, not qualified general bounds.
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
  %s16 = "atlas.vload"(%s15) {dst = 2 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s17 = "atlas.delay"(%s16) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s18 = "atlas.alu_imm"(%s17) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 512 : i32} : (!atlas.state) -> !atlas.state
  %s19 = "atlas.upper"(%s18) {kind = "lui", dst = 1 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s20 = "atlas.alu_imm"(%s19) {kind = "addi", dst = 1 : i32, src = 1 : i32, immediate = -2048 : i32} : (!atlas.state) -> !atlas.state
  %s21 = "atlas.dma"(%s20) {direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s22 = "atlas.dma_wait"(%s21) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s23 = "atlas.vload"(%s22) {dst = 4 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s24 = "atlas.delay"(%s23) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s25 = "atlas.mxu_push"(%s24) {kind = "weight_fp8", unit = 0 : i32, src = 2 : i32, slot = 0 : i32} : (!atlas.state) -> !atlas.state
  %s26 = "atlas.delay"(%s25) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s27 = "atlas.mxu_matmul"(%s26) {unit = 0 : i32, src = 0 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false} : (!atlas.state) -> !atlas.state
  %s28 = "atlas.delay"(%s27) {cycles = 95 : i32} : (!atlas.state) -> !atlas.state
  %s29 = "atlas.mxu_pop"(%s28) {format = "bf16", unit = 0 : i32, dst = 8 : i32, slot = 0 : i32, scale_reg = 0 : i32} : (!atlas.state) -> !atlas.state
  %s30 = "atlas.delay"(%s29) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s31 = "atlas.vpu_unary"(%s30) {kind = "relu", dst = 10 : i32, src = 8 : i32} : (!atlas.state) -> !atlas.state
  %s32 = "atlas.delay"(%s31) {cycles = 64 : i32} : (!atlas.state) -> !atlas.state
  %s33 = "atlas.scalar_load"(%s32) {kind = "seli", dst = 3 : i32, base = 0 : i32, offset = 127 : i32} : (!atlas.state) -> !atlas.state
  %s34 = "atlas.vpu_pack"(%s33) {direction = "bf16_to_fp8", dst = 12 : i32, src = 10 : i32, scale_reg = 3 : i32} : (!atlas.state) -> !atlas.state
  %s35 = "atlas.delay"(%s34) {cycles = 127 : i32} : (!atlas.state) -> !atlas.state
  %s36 = "atlas.mxu_push"(%s35) {kind = "weight_fp8", unit = 1 : i32, src = 4 : i32, slot = 0 : i32} : (!atlas.state) -> !atlas.state
  %s37 = "atlas.delay"(%s36) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s38 = "atlas.mxu_matmul"(%s37) {unit = 1 : i32, src = 12 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false} : (!atlas.state) -> !atlas.state
  %s39 = "atlas.delay"(%s38) {cycles = 95 : i32} : (!atlas.state) -> !atlas.state
  %s40 = "atlas.mxu_pop"(%s39) {format = "bf16", unit = 1 : i32, dst = 14 : i32, slot = 0 : i32, scale_reg = 0 : i32} : (!atlas.state) -> !atlas.state
  %s41 = "atlas.delay"(%s40) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s42 = "atlas.alu_imm"(%s41) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 768 : i32} : (!atlas.state) -> !atlas.state
  %s43 = "atlas.vstore"(%s42) {src = 14 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s44 = "atlas.delay"(%s43) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s45 = "atlas.upper"(%s44) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s46 = "atlas.alu_imm"(%s45) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = -1024 : i32} : (!atlas.state) -> !atlas.state
  %s47 = "atlas.dma"(%s46) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s48 = "atlas.dma_wait"(%s47) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s49 = "atlas.alu_imm"(%s48) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s50 = "atlas.vstore"(%s49) {src = 15 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s51 = "atlas.delay"(%s50) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s52 = "atlas.upper"(%s51) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s53 = "atlas.dma"(%s52) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s54 = "atlas.dma_wait"(%s53) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s55 = "atlas.alu_imm"(%s54) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s56 = "atlas.csr"(%s55) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s57 = "atlas.trap"(%s56) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
