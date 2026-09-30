// Hand-authored fixed 32x32 two-layer quantized MLP diagnostic.
// MXU0(X,W0) -> BF16 ReLU -> unit-scale FP8 pack -> VMEM row-half relayout -> MXU1(H,W1).
// Inputs are three FP8 tiles at DRAM 0x90000000,+0x400,+0x800; BF16 output halves
// are at +0xC00,+0x1000. There is no bias or model frontend.
// The scalar loop repairs the selected PACK physical-row layout for MXU1.
// Delays are diagnostic, not general availability bounds.
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
  %s36 = "atlas.alu_imm"(%s35) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1280 : i32} : (!atlas.state) -> !atlas.state
  %s37 = "atlas.vstore"(%s36) {src = 12 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s38 = "atlas.delay"(%s37) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s39 = "atlas.upper"(%s38) {kind = "lui", dst = 10 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s40 = "atlas.alu_imm"(%s39) {kind = "addi", dst = 10 : i32, src = 10 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s41 = "atlas.upper"(%s40) {kind = "lui", dst = 11 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s42 = "atlas.alu_imm"(%s41) {kind = "addi", dst = 11 : i32, src = 11 : i32, immediate = 1536 : i32} : (!atlas.state) -> !atlas.state
  %s43 = "atlas.upper"(%s42) {kind = "lui", dst = 12 : i32, immediate = 2 : i32} : (!atlas.state) -> !atlas.state
  %s44 = "atlas.alu_imm"(%s43) {kind = "addi", dst = 12 : i32, src = 12 : i32, immediate = -2048 : i32} : (!atlas.state) -> !atlas.state
  %s45 = "atlas.alu_imm"(%s44) {kind = "addi", dst = 13 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s46 = "atlas.alu_imm"(%s45) {kind = "addi", dst = 14 : i32, src = 0 : i32, immediate = 32 : i32} : (!atlas.state) -> !atlas.state
  %s47 = "atlas.scalar_load"(%s46) {kind = "lw", dst = 16 : i32, base = 10 : i32, offset = 0 : i32} : (!atlas.state) -> !atlas.state
  %s48 = "atlas.delay"(%s47) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s49 = "atlas.scalar_store"(%s48) {kind = "sw", src = 16 : i32, base = 12 : i32, offset = 0 : i32} : (!atlas.state) -> !atlas.state
  %s50 = "atlas.scalar_load"(%s49) {kind = "lw", dst = 17 : i32, base = 11 : i32, offset = 0 : i32} : (!atlas.state) -> !atlas.state
  %s51 = "atlas.delay"(%s50) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s52 = "atlas.scalar_store"(%s51) {kind = "sw", src = 17 : i32, base = 12 : i32, offset = 16 : i32} : (!atlas.state) -> !atlas.state
  %s53 = "atlas.scalar_load"(%s52) {kind = "lw", dst = 16 : i32, base = 10 : i32, offset = 4 : i32} : (!atlas.state) -> !atlas.state
  %s54 = "atlas.delay"(%s53) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s55 = "atlas.scalar_store"(%s54) {kind = "sw", src = 16 : i32, base = 12 : i32, offset = 4 : i32} : (!atlas.state) -> !atlas.state
  %s56 = "atlas.scalar_load"(%s55) {kind = "lw", dst = 17 : i32, base = 11 : i32, offset = 4 : i32} : (!atlas.state) -> !atlas.state
  %s57 = "atlas.delay"(%s56) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s58 = "atlas.scalar_store"(%s57) {kind = "sw", src = 17 : i32, base = 12 : i32, offset = 20 : i32} : (!atlas.state) -> !atlas.state
  %s59 = "atlas.scalar_load"(%s58) {kind = "lw", dst = 16 : i32, base = 10 : i32, offset = 8 : i32} : (!atlas.state) -> !atlas.state
  %s60 = "atlas.delay"(%s59) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s61 = "atlas.scalar_store"(%s60) {kind = "sw", src = 16 : i32, base = 12 : i32, offset = 8 : i32} : (!atlas.state) -> !atlas.state
  %s62 = "atlas.scalar_load"(%s61) {kind = "lw", dst = 17 : i32, base = 11 : i32, offset = 8 : i32} : (!atlas.state) -> !atlas.state
  %s63 = "atlas.delay"(%s62) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s64 = "atlas.scalar_store"(%s63) {kind = "sw", src = 17 : i32, base = 12 : i32, offset = 24 : i32} : (!atlas.state) -> !atlas.state
  %s65 = "atlas.scalar_load"(%s64) {kind = "lw", dst = 16 : i32, base = 10 : i32, offset = 12 : i32} : (!atlas.state) -> !atlas.state
  %s66 = "atlas.delay"(%s65) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s67 = "atlas.scalar_store"(%s66) {kind = "sw", src = 16 : i32, base = 12 : i32, offset = 12 : i32} : (!atlas.state) -> !atlas.state
  %s68 = "atlas.scalar_load"(%s67) {kind = "lw", dst = 17 : i32, base = 11 : i32, offset = 12 : i32} : (!atlas.state) -> !atlas.state
  %s69 = "atlas.delay"(%s68) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s70 = "atlas.scalar_store"(%s69) {kind = "sw", src = 17 : i32, base = 12 : i32, offset = 28 : i32} : (!atlas.state) -> !atlas.state
  %s71 = "atlas.alu_imm"(%s70) {kind = "addi", dst = 10 : i32, src = 10 : i32, immediate = 16 : i32} : (!atlas.state) -> !atlas.state
  %s72 = "atlas.alu_imm"(%s71) {kind = "addi", dst = 11 : i32, src = 11 : i32, immediate = 16 : i32} : (!atlas.state) -> !atlas.state
  %s73 = "atlas.alu_imm"(%s72) {kind = "addi", dst = 12 : i32, src = 12 : i32, immediate = 32 : i32} : (!atlas.state) -> !atlas.state
  %s74 = "atlas.alu_imm"(%s73) {kind = "addi", dst = 13 : i32, src = 13 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s75 = "atlas.branch"(%s74) {kind = "blt", lhs = 13 : i32, rhs = 14 : i32, offset_bytes = -56 : i32} : (!atlas.state) -> !atlas.state
  %s76 = "atlas.alu_imm"(%s75) {kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s77 = "atlas.delay"(%s76) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s78 = "atlas.alu_imm"(%s77) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 1536 : i32} : (!atlas.state) -> !atlas.state
  %s79 = "atlas.vload"(%s78) {dst = 12 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s80 = "atlas.delay"(%s79) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s81 = "atlas.mxu_push"(%s80) {kind = "weight_fp8", unit = 1 : i32, src = 4 : i32, slot = 0 : i32} : (!atlas.state) -> !atlas.state
  %s82 = "atlas.delay"(%s81) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s83 = "atlas.mxu_matmul"(%s82) {unit = 1 : i32, src = 12 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false} : (!atlas.state) -> !atlas.state
  %s84 = "atlas.delay"(%s83) {cycles = 95 : i32} : (!atlas.state) -> !atlas.state
  %s85 = "atlas.mxu_pop"(%s84) {format = "bf16", unit = 1 : i32, dst = 14 : i32, slot = 0 : i32, scale_reg = 0 : i32} : (!atlas.state) -> !atlas.state
  %s86 = "atlas.delay"(%s85) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s87 = "atlas.alu_imm"(%s86) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 768 : i32} : (!atlas.state) -> !atlas.state
  %s88 = "atlas.vstore"(%s87) {src = 14 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s89 = "atlas.delay"(%s88) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s90 = "atlas.upper"(%s89) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s91 = "atlas.alu_imm"(%s90) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = -1024 : i32} : (!atlas.state) -> !atlas.state
  %s92 = "atlas.dma"(%s91) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s93 = "atlas.dma_wait"(%s92) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s94 = "atlas.alu_imm"(%s93) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s95 = "atlas.vstore"(%s94) {src = 15 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s96 = "atlas.delay"(%s95) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s97 = "atlas.upper"(%s96) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s98 = "atlas.dma"(%s97) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s99 = "atlas.dma_wait"(%s98) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s100 = "atlas.alu_imm"(%s99) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s101 = "atlas.csr"(%s100) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s102 = "atlas.trap"(%s101) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
