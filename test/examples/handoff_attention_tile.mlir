// Diagnostic unmasked attention instruction sequence: QK^T -> BF16 row
// normalization via exp2 -> E8M0 pack at unit scale -> PV.
// This is not a correct general 32x32 attention implementation: PACK changes
// the row layout before PV. See docs/llvm-handoff-examples.md.
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
  %s50 = "atlas.mxu_push"(%s49) {kind = "weight_fp8", unit = 1 : i32, src = 4 : i32, slot = 0 : i32} : (!atlas.state) -> !atlas.state
  %s51 = "atlas.delay"(%s50) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s52 = "atlas.mxu_matmul"(%s51) {unit = 1 : i32, src = 26 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false} : (!atlas.state) -> !atlas.state
  %s53 = "atlas.delay"(%s52) {cycles = 95 : i32} : (!atlas.state) -> !atlas.state
  %s54 = "atlas.mxu_pop"(%s53) {format = "bf16", unit = 1 : i32, dst = 28 : i32, slot = 0 : i32, scale_reg = 0 : i32} : (!atlas.state) -> !atlas.state
  %s55 = "atlas.delay"(%s54) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s56 = "atlas.alu_imm"(%s55) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 768 : i32} : (!atlas.state) -> !atlas.state
  %s57 = "atlas.vstore"(%s56) {src = 28 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s58 = "atlas.delay"(%s57) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s59 = "atlas.upper"(%s58) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s60 = "atlas.alu_imm"(%s59) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = -1024 : i32} : (!atlas.state) -> !atlas.state
  %s61 = "atlas.dma"(%s60) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s62 = "atlas.dma_wait"(%s61) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s63 = "atlas.alu_imm"(%s62) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s64 = "atlas.vstore"(%s63) {src = 29 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s65 = "atlas.delay"(%s64) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s66 = "atlas.upper"(%s65) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s67 = "atlas.dma"(%s66) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s68 = "atlas.dma_wait"(%s67) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s69 = "atlas.alu_imm"(%s68) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s70 = "atlas.csr"(%s69) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s71 = "atlas.trap"(%s70) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
