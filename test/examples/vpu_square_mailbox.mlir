// Selected standalone AtlasCore reset-entry mailbox. Runtime supplies two
// little-endian u32 DRAM byte addresses at 0x90000000: input at +0, output at
// +4. Both buffers are 2048 bytes and must be 32-byte aligned, disjoint, and
// inside the selected 1 MiB DRAM window. The program does not embed their
// addresses. The mailbox is DMA-loaded to VMEM, then read by scalar LW.
// DELAY 8 after each LW is bounded evidence for this selected core, not a
// general scalar-load completion protocol. ECALL is the launch completion.
module {
  %s0 = "atlas.start"() : () -> !atlas.state
  %s1 = "atlas.alu_imm"(%s0) {kind = "addi", dst = 28 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s2 = "atlas.alu_imm"(%s1) {kind = "addi", dst = 5 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s3 = "atlas.dma_config"(%s2) {channel = 0 : i32, base_reg = 5 : i32} : (!atlas.state) -> !atlas.state
  %s4 = "atlas.alu_imm"(%s3) {kind = "addi", dst = 2 : i32, src = 0 : i32, immediate = 32 : i32} : (!atlas.state) -> !atlas.state
  %s5 = "atlas.alu_imm"(%s4) {kind = "addi", dst = 6 : i32, src = 0 : i32, immediate = 0 : i32} : (!atlas.state) -> !atlas.state
  %s6 = "atlas.upper"(%s5) {kind = "lui", dst = 1 : i32, immediate = 589824 : i32} : (!atlas.state) -> !atlas.state
  %s7 = "atlas.dma"(%s6) {direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s8 = "atlas.dma_wait"(%s7) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s9 = "atlas.scalar_load"(%s8) {kind = "lw", dst = 1 : i32, base = 0 : i32, offset = 0 : i32} : (!atlas.state) -> !atlas.state
  %s10 = "atlas.delay"(%s9) {cycles = 8 : i32} : (!atlas.state) -> !atlas.state
  %s11 = "atlas.scalar_load"(%s10) {kind = "lw", dst = 3 : i32, base = 0 : i32, offset = 4 : i32} : (!atlas.state) -> !atlas.state
  %s12 = "atlas.delay"(%s11) {cycles = 8 : i32} : (!atlas.state) -> !atlas.state
  %s13 = "atlas.alu_imm"(%s12) {kind = "addi", dst = 2 : i32, src = 0 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s14 = "atlas.dma"(%s13) {direction = "load", channel = 0 : i32, reg = 6 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s15 = "atlas.dma_wait"(%s14) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s16 = "atlas.vload"(%s15) {dst = 0 : i32, base = 6 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s17 = "atlas.delay"(%s16) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s18 = "atlas.alu_imm"(%s17) {kind = "addi", dst = 7 : i32, src = 0 : i32, immediate = 256 : i32} : (!atlas.state) -> !atlas.state
  %s19 = "atlas.alu_imm"(%s18) {kind = "addi", dst = 1 : i32, src = 1 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s20 = "atlas.dma"(%s19) {direction = "load", channel = 0 : i32, reg = 7 : i32, dram = 1 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s21 = "atlas.dma_wait"(%s20) {channel = 0 : i32} : (!atlas.state) -> !atlas.state
  %s22 = "atlas.vload"(%s21) {dst = 1 : i32, base = 7 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s23 = "atlas.delay"(%s22) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s24 = "atlas.vpu_unary"(%s23) {kind = "square", dst = 4 : i32, src = 0 : i32} : (!atlas.state) -> !atlas.state
  %s25 = "atlas.delay"(%s24) {cycles = 64 : i32} : (!atlas.state) -> !atlas.state
  %s26 = "atlas.alu_imm"(%s25) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 512 : i32} : (!atlas.state) -> !atlas.state
  %s27 = "atlas.vstore"(%s26) {src = 4 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s28 = "atlas.delay"(%s27) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s29 = "atlas.dma"(%s28) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s30 = "atlas.dma_wait"(%s29) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s31 = "atlas.alu_imm"(%s30) {kind = "addi", dst = 8 : i32, src = 0 : i32, immediate = 768 : i32} : (!atlas.state) -> !atlas.state
  %s32 = "atlas.vstore"(%s31) {src = 5 : i32, base = 8 : i32, offset = 0 : i32, format = "raw"} : (!atlas.state) -> !atlas.state
  %s33 = "atlas.delay"(%s32) {cycles = 33 : i32} : (!atlas.state) -> !atlas.state
  %s34 = "atlas.alu_imm"(%s33) {kind = "addi", dst = 3 : i32, src = 3 : i32, immediate = 1024 : i32} : (!atlas.state) -> !atlas.state
  %s35 = "atlas.dma"(%s34) {direction = "store", channel = 1 : i32, reg = 8 : i32, dram = 3 : i32, size = 2 : i32} : (!atlas.state) -> !atlas.state
  %s36 = "atlas.dma_wait"(%s35) {channel = 1 : i32} : (!atlas.state) -> !atlas.state
  %s37 = "atlas.alu_imm"(%s36) {kind = "addi", dst = 1 : i32, src = 0 : i32, immediate = 1 : i32} : (!atlas.state) -> !atlas.state
  %s38 = "atlas.csr"(%s37) {kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32} : (!atlas.state) -> !atlas.state
  %s39 = "atlas.trap"(%s38) {kind = "ecall"} : (!atlas.state) -> !atlas.state
}
