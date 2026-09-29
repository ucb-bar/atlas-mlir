// Bounded E8M0 pack/unpack and downstream MXU1 FP8-consumer stream.
// Set SELI e3 to the selected raw scale code; paired DRAM inputs at +0/+0x400.
// Outputs: packed FP8 +0x800, unpacked BF16 +0xC00/+0x1000,
// MXU1 BF16 +0x1400/+0x1800. Delays are diagnostic, not qualified bounds.
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
  %s18 = "atlas.scalar_load"(%s17) {kind = "seli", dst = 3 : i32, base = 0 : i32, offset = 128 : i32} : (!atlas.state) -> !atlas.state
  %s19 = "atlas.vpu_pack"(%s18) {direction = "bf16_to_fp8", dst = 2 : i32, src = 0 : i32, scale_reg = 3 : i32} : (!atlas.state) -> !atlas.state
  %s20 = "atlas.delay"(%s19) {cycles = 95 : i32} : (!atlas.state) -> !atlas.state
  %s21 = "atlas.vpu_pack"(%s20) {direction = "fp8_to_bf16", dst = 4 : i32, src = 2 : i32, scale_reg = 3 : i32} : (!atlas.state) -> !atlas.state
  %s22 = "atlas.delay"(%s21) {cycles = 127 : i32} : (!atlas.state) -> !atlas.state
  %s23 = "atlas.mxu_push"(%s22) {kind = "weight_fp8", unit = 1 : i32, src = 2 : i32, slot = 0 : i32} : (!atlas.state) -> !atlas.state
  %s24 = "atlas.delay"(%s23) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s25 = "atlas.mxu_matmul"(%s24) {unit = 1 : i32, src = 2 : i32, weight_slot = 0 : i32, acc_slot = 0 : i32, accumulate = false} : (!atlas.state) -> !atlas.state
  %s26 = "atlas.delay"(%s25) {cycles = 95 : i32} : (!atlas.state) -> !atlas.state
  %s27 = "atlas.mxu_pop"(%s26) {format = "bf16", unit = 1 : i32, dst = 6 : i32, slot = 0 : i32, scale_reg = 0 : i32} : (!atlas.state) -> !atlas.state
  %s28 = "atlas.delay"(%s27) {cycles = 31 : i32} : (!atlas.state) -> !atlas.state
  %s29 = "atlas.alu_imm"(%s28) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 512 : i32} : (!atlas.state) -> !atlas.state
  %s30 = "atlas.vstore"(%s29) {src = 2 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s31 = "atlas.delay"(%s30) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s32 = "atlas.upper"(%s31) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s33 = "atlas.alu_imm"(%s32) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = -2048 : i32} : (!atlas.state) -> !atlas.state
  %s34 = "atlas.dma"(%s33) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s35 = "atlas.dma_wait"(%s34) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s36 = "atlas.alu_imm"(%s35) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 768 : i32} : (!atlas.state) -> !atlas.state
  %s37 = "atlas.vstore"(%s36) {src = 4 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s38 = "atlas.delay"(%s37) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s39 = "atlas.upper"(%s38) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s40 = "atlas.alu_imm"(%s39) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = -1024 : i32} : (!atlas.state) -> !atlas.state
  %s41 = "atlas.dma"(%s40) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s42 = "atlas.dma_wait"(%s41) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s43 = "atlas.alu_imm"(%s42) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s44 = "atlas.vstore"(%s43) {src = 5 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s45 = "atlas.delay"(%s44) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s46 = "atlas.upper"(%s45) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s47 = "atlas.dma"(%s46) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s48 = "atlas.dma_wait"(%s47) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s49 = "atlas.alu_imm"(%s48) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1280 : i32} : (!atlas.state) -> !atlas.state
  %s50 = "atlas.vstore"(%s49) {src = 6 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s51 = "atlas.delay"(%s50) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s52 = "atlas.upper"(%s51) {kind = "lui", dst = 3 : i32, immediate = 589825 : i32} : (!atlas.state) -> !atlas.state
  %s53 = "atlas.alu_imm"(%s52) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s54 = "atlas.dma"(%s53) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s55 = "atlas.dma_wait"(%s54) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s56 = "atlas.alu_imm"(%s55) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 1536 : i32} : (!atlas.state) -> !atlas.state
  %s57 = "atlas.vstore"(%s56) {src = 7 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s58 = "atlas.delay"(%s57) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s59 = "atlas.upper"(%s58) {kind = "lui", dst = 3 : i32, immediate = 589826 : i32} : (!atlas.state) -> !atlas.state
  %s60 = "atlas.alu_imm"(%s59) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = -2048 : i32} : (!atlas.state) -> !atlas.state
  %s61 = "atlas.dma"(%s60) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s62 = "atlas.dma_wait"(%s61) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s63 = "atlas.alu_imm"(%s62) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s64 = "atlas.csr"(%s63) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s65 = "atlas.trap"(%s64) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
