// Hand-authored fixed 32x32 unmasked attention diagnostic.
// QK^T -> BF16 row normalization via exp2 -> unit-scale FP8 pack ->
// VMEM row-half relayout -> PV. No mask, causal state, or model frontend.
// Inputs are FP8 Q,K^T,V tiles at DRAM 0x90000000,+0x400,+0x800;
// BF16 output halves are at +0xC00,+0x1000.
// BF16 raw 0x3e83 approximates log2(e)/sqrt(32); this is a selected precision policy.
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
  %s31 = "atlas.vpu_reduce"(%s30) {kind = "row_max", dst = 10 : i32, src = 8 : i32} : (!atlas.state) -> !atlas.state
  %s32 = "atlas.delay"(%s31) {cycles = 64 : i32} : (!atlas.state) -> !atlas.state
  %s33 = "atlas.vpu_binary"(%s32) {kind = "sub", dst = 12 : i32, lhs = 8 : i32, rhs = 10 : i32} : (!atlas.state) -> !atlas.state
  %s34 = "atlas.delay"(%s33) {cycles = 64 : i32} : (!atlas.state) -> !atlas.state
  %s35 = "atlas.vli"(%s34) {mode = "all", dst = 14 : i32, immediate = 16003 : i32} : (!atlas.state) -> !atlas.state
  %s36 = "atlas.delay"(%s35) {cycles = 64 : i32} : (!atlas.state) -> !atlas.state
  %s37 = "atlas.vpu_binary"(%s36) {kind = "mul", dst = 16 : i32, lhs = 12 : i32, rhs = 14 : i32} : (!atlas.state) -> !atlas.state
  %s38 = "atlas.delay"(%s37) {cycles = 64 : i32} : (!atlas.state) -> !atlas.state
  %s39 = "atlas.vpu_unary"(%s38) {kind = "exp2", dst = 18 : i32, src = 16 : i32} : (!atlas.state) -> !atlas.state
  %s40 = "atlas.delay"(%s39) {cycles = 64 : i32} : (!atlas.state) -> !atlas.state
  %s41 = "atlas.vpu_reduce"(%s40) {kind = "row_sum", dst = 20 : i32, src = 18 : i32} : (!atlas.state) -> !atlas.state
  %s42 = "atlas.delay"(%s41) {cycles = 64 : i32} : (!atlas.state) -> !atlas.state
  %s43 = "atlas.vpu_unary"(%s42) {kind = "recip", dst = 22 : i32, src = 20 : i32} : (!atlas.state) -> !atlas.state
  %s44 = "atlas.delay"(%s43) {cycles = 64 : i32} : (!atlas.state) -> !atlas.state
  %s45 = "atlas.vpu_binary"(%s44) {kind = "mul", dst = 24 : i32, lhs = 18 : i32, rhs = 22 : i32} : (!atlas.state) -> !atlas.state
  %s46 = "atlas.delay"(%s45) {cycles = 64 : i32} : (!atlas.state) -> !atlas.state
  %s47 = "atlas.scalar_load"(%s46) {kind = "seli", dst = 3 : i32, base = 0 : i32, offset = 127 : i32} : (!atlas.state) -> !atlas.state
  %s48 = "atlas.vpu_pack"(%s47) {direction = "bf16_to_fp8", dst = 26 : i32, src = 24 : i32, scale_reg = 3 : i32} : (!atlas.state) -> !atlas.state
  %s49 = "atlas.delay"(%s48) {cycles = 127 : i32} : (!atlas.state) -> !atlas.state
  %s50 = "atlas.alu_imm"(%s49) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1280 : i32} : (!atlas.state) -> !atlas.state
  %s51 = "atlas.vstore"(%s50) {src = 26 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s52 = "atlas.delay"(%s51) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s53 = "atlas.upper"(%s52) {kind = "lui", dst = 10 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s54 = "atlas.alu_imm"(%s53) {kind = "addi", dst = 10 : i32, src = 10 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s55 = "atlas.upper"(%s54) {kind = "lui", dst = 11 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s56 = "atlas.alu_imm"(%s55) {kind = "addi", dst = 11 : i32, src = 11 : i32, immediate = 1536 : i32} : (!atlas.state) -> !atlas.state
  %s57 = "atlas.upper"(%s56) {kind = "lui", dst = 12 : i32, immediate = 2 : i32} : (!atlas.state) -> !atlas.state
  %s58 = "atlas.alu_imm"(%s57) {kind = "addi", dst = 12 : i32, src = 12 : i32, immediate = -2048 : i32} : (!atlas.state) -> !atlas.state
  %s59 = "atlas.alu_imm"(%s58) {kind = "addi", dst = 13 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s60 = "atlas.alu_imm"(%s59) {kind = "addi", dst = 14 : i32, src = 0 : i32, immediate = 32 : i32} : (!atlas.state) -> !atlas.state
  %s61 = "atlas.scalar_load"(%s60) {kind = "lw", dst = 16 : i32, base = 10 : i32, offset = 0 : i32} : (!atlas.state) -> !atlas.state
  %s62 = "atlas.delay"(%s61) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s63 = "atlas.scalar_store"(%s62) {kind = "sw", src = 16 : i32, base = 12 : i32, offset = 0 : i32} : (!atlas.state) -> !atlas.state
  %s64 = "atlas.scalar_load"(%s63) {kind = "lw", dst = 17 : i32, base = 11 : i32, offset = 0 : i32} : (!atlas.state) -> !atlas.state
  %s65 = "atlas.delay"(%s64) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s66 = "atlas.scalar_store"(%s65) {kind = "sw", src = 17 : i32, base = 12 : i32, offset = 16 : i32} : (!atlas.state) -> !atlas.state
  %s67 = "atlas.scalar_load"(%s66) {kind = "lw", dst = 16 : i32, base = 10 : i32, offset = 4 : i32} : (!atlas.state) -> !atlas.state
  %s68 = "atlas.delay"(%s67) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s69 = "atlas.scalar_store"(%s68) {kind = "sw", src = 16 : i32, base = 12 : i32, offset = 4 : i32} : (!atlas.state) -> !atlas.state
  %s70 = "atlas.scalar_load"(%s69) {kind = "lw", dst = 17 : i32, base = 11 : i32, offset = 4 : i32} : (!atlas.state) -> !atlas.state
  %s71 = "atlas.delay"(%s70) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s72 = "atlas.scalar_store"(%s71) {kind = "sw", src = 17 : i32, base = 12 : i32, offset = 20 : i32} : (!atlas.state) -> !atlas.state
  %s73 = "atlas.scalar_load"(%s72) {kind = "lw", dst = 16 : i32, base = 10 : i32, offset = 8 : i32} : (!atlas.state) -> !atlas.state
  %s74 = "atlas.delay"(%s73) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s75 = "atlas.scalar_store"(%s74) {kind = "sw", src = 16 : i32, base = 12 : i32, offset = 8 : i32} : (!atlas.state) -> !atlas.state
  %s76 = "atlas.scalar_load"(%s75) {kind = "lw", dst = 17 : i32, base = 11 : i32, offset = 8 : i32} : (!atlas.state) -> !atlas.state
  %s77 = "atlas.delay"(%s76) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s78 = "atlas.scalar_store"(%s77) {kind = "sw", src = 17 : i32, base = 12 : i32, offset = 24 : i32} : (!atlas.state) -> !atlas.state
  %s79 = "atlas.scalar_load"(%s78) {kind = "lw", dst = 16 : i32, base = 10 : i32, offset = 12 : i32} : (!atlas.state) -> !atlas.state
  %s80 = "atlas.delay"(%s79) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s81 = "atlas.scalar_store"(%s80) {kind = "sw", src = 16 : i32, base = 12 : i32, offset = 12 : i32} : (!atlas.state) -> !atlas.state
  %s82 = "atlas.scalar_load"(%s81) {kind = "lw", dst = 17 : i32, base = 11 : i32, offset = 12 : i32} : (!atlas.state) -> !atlas.state
  %s83 = "atlas.delay"(%s82) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s84 = "atlas.scalar_store"(%s83) {kind = "sw", src = 17 : i32, base = 12 : i32, offset = 28 : i32} : (!atlas.state) -> !atlas.state
  %s85 = "atlas.alu_imm"(%s84) {kind = "addi", dst = 10 : i32, src = 10 : i32, immediate = 16 : i32} : (!atlas.state) -> !atlas.state
  %s86 = "atlas.alu_imm"(%s85) {kind = "addi", dst = 11 : i32, src = 11 : i32, immediate = 16 : i32} : (!atlas.state) -> !atlas.state
  %s87 = "atlas.alu_imm"(%s86) {kind = "addi", dst = 12 : i32, src = 12 : i32, immediate = 32 : i32} : (!atlas.state) -> !atlas.state
  %s88 = "atlas.alu_imm"(%s87) {kind = "addi", dst = 13 : i32, src = 13 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s89 = "atlas.branch"(%s88) {kind = "blt", lhs = 13 : i32, rhs = 14 : i32, offset_bytes = -56 : i32} : (!atlas.state) -> !atlas.state
  %s90 = "atlas.alu_imm"(%s89) {kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s91 = "atlas.delay"(%s90) {cycles = 4 : i32} : (!atlas.state) -> !atlas.state
  %s92 = "atlas.alu_imm"(%s91) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 1536 : i32} : (!atlas.state) -> !atlas.state
  %s93 = "atlas.vload"(%s92) {dst = 26 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s94 = "atlas.delay"(%s93) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s95 = "atlas.mxu_push"(%s94) {kind = "weight_fp8", unit = 1 : i32, src = 4 : i32, slot = 0 : i32} : (!atlas.state) -> !atlas.state
  %s96 = "atlas.delay"(%s95) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s97 = "atlas.mxu_matmul"(%s96) {unit = 1 : i32, src = 26 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false} : (!atlas.state) -> !atlas.state
  %s98 = "atlas.delay"(%s97) {cycles = 95 : i32} : (!atlas.state) -> !atlas.state
  %s99 = "atlas.mxu_pop"(%s98) {format = "bf16", unit = 1 : i32, dst = 28 : i32, slot = 0 : i32, scale_reg = 0 : i32} : (!atlas.state) -> !atlas.state
  %s100 = "atlas.delay"(%s99) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s101 = "atlas.alu_imm"(%s100) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 768 : i32} : (!atlas.state) -> !atlas.state
  %s102 = "atlas.vstore"(%s101) {src = 28 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s103 = "atlas.delay"(%s102) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s104 = "atlas.upper"(%s103) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s105 = "atlas.alu_imm"(%s104) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = -1024 : i32} : (!atlas.state) -> !atlas.state
  %s106 = "atlas.dma"(%s105) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s107 = "atlas.dma_wait"(%s106) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s108 = "atlas.alu_imm"(%s107) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s109 = "atlas.vstore"(%s108) {src = 29 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s110 = "atlas.delay"(%s109) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s111 = "atlas.upper"(%s110) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s112 = "atlas.dma"(%s111) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s113 = "atlas.dma_wait"(%s112) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s114 = "atlas.alu_imm"(%s113) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s115 = "atlas.csr"(%s114) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s116 = "atlas.trap"(%s115) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
