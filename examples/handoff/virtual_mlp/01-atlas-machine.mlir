module attributes {atlas.generated_from_virtual, atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415927296 : i64, atlas.scalar_arg_regs = array<i32>} {
  %0 = "atlas.start"() : () -> !atlas.state
  %1 = "atlas.alu_imm"(%0) <{dst = 28 : i32, immediate = 1 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %2 = "atlas.alu_imm"(%1) <{dst = 5 : i32, immediate = 0 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %3 = "atlas.dma_config"(%2) <{base_reg = 5 : i32, channel = 0 : i32}> : (!atlas.state) -> !atlas.state
  %4 = "atlas.dma_config"(%3) <{base_reg = 5 : i32, channel = 1 : i32}> : (!atlas.state) -> !atlas.state
  %5 = "atlas.alu_imm"(%4) <{dst = 2 : i32, immediate = 1024 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %6 = "atlas.alu_imm"(%5) <{dst = 6 : i32, immediate = 0 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %7 = "atlas.upper"(%6) <{dst = 1 : i32, immediate = 589824 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %8 = "atlas.alu_imm"(%7) <{dst = 1 : i32, immediate = 0 : i32, kind = "addi", src = 1 : i32}> : (!atlas.state) -> !atlas.state
  %9 = "atlas.dma"(%8) <{channel = 0 : i32, direction = "load", dram = 1 : i32, reg = 6 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %10 = "atlas.dma_wait"(%9) <{channel = 0 : i32}> : (!atlas.state) -> !atlas.state
  %11 = "atlas.vload"(%10) <{base = 6 : i32, dst = 1 : i32, format = "raw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %12 = "atlas.delay"(%11) <{cycles = 256 : i32}> {atlas.delay_reason = "vload_completion"} : (!atlas.state) -> !atlas.state
  %13 = "atlas.alu_imm"(%12) <{dst = 6 : i32, immediate = 512 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %14 = "atlas.upper"(%13) <{dst = 1 : i32, immediate = 589825 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %15 = "atlas.alu_imm"(%14) <{dst = 1 : i32, immediate = -2048 : i32, kind = "addi", src = 1 : i32}> : (!atlas.state) -> !atlas.state
  %16 = "atlas.dma"(%15) <{channel = 0 : i32, direction = "load", dram = 1 : i32, reg = 6 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %17 = "atlas.dma_wait"(%16) <{channel = 0 : i32}> : (!atlas.state) -> !atlas.state
  %18 = "atlas.vload"(%17) <{base = 6 : i32, dst = 2 : i32, format = "raw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %19 = "atlas.delay"(%18) <{cycles = 256 : i32}> {atlas.delay_reason = "vload_completion"} : (!atlas.state) -> !atlas.state
  %20 = "atlas.alu_imm"(%19) <{dst = 6 : i32, immediate = 1024 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %21 = "atlas.upper"(%20) <{dst = 1 : i32, immediate = 589825 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %22 = "atlas.alu_imm"(%21) <{dst = 1 : i32, immediate = 0 : i32, kind = "addi", src = 1 : i32}> : (!atlas.state) -> !atlas.state
  %23 = "atlas.dma"(%22) <{channel = 0 : i32, direction = "load", dram = 1 : i32, reg = 6 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %24 = "atlas.dma_wait"(%23) <{channel = 0 : i32}> : (!atlas.state) -> !atlas.state
  %25 = "atlas.vload"(%24) <{base = 6 : i32, dst = 0 : i32, format = "raw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %26 = "atlas.delay"(%25) <{cycles = 256 : i32}> {atlas.delay_reason = "vload_completion"} : (!atlas.state) -> !atlas.state
  %27 = "atlas.mxu_push"(%26) <{kind = "weight_fp8", slot = 0 : i32, src = 2 : i32, unit = 0 : i32}> : (!atlas.state) -> !atlas.state
  %28 = "atlas.delay"(%27) <{cycles = 256 : i32}> {atlas.delay_reason = "mxu_weight_completion"} : (!atlas.state) -> !atlas.state
  %29 = "atlas.mxu_matmul"(%28) <{acc_slot = 0 : i32, accumulate = false, src = 1 : i32, unit = 0 : i32, weight_slot = 0 : i32}> : (!atlas.state) -> !atlas.state
  %30 = "atlas.delay"(%29) <{cycles = 256 : i32}> {atlas.delay_reason = "mxu_matmul_completion"} : (!atlas.state) -> !atlas.state
  %31 = "atlas.mxu_pop"(%30) <{dst = 32 : i32, format = "bf16", scale_reg = 0 : i32, slot = 0 : i32, unit = 0 : i32}> : (!atlas.state) -> !atlas.state
  %32 = "atlas.delay"(%31) <{cycles = 256 : i32}> {atlas.delay_reason = "mxu_readout_completion"} : (!atlas.state) -> !atlas.state
  %33 = "atlas.vpu_unary"(%32) <{dst = 34 : i32, kind = "relu", src = 32 : i32}> : (!atlas.state) -> !atlas.state
  %34 = "atlas.delay"(%33) <{cycles = 256 : i32}> {atlas.delay_reason = "vpu_completion"} : (!atlas.state) -> !atlas.state
  %35 = "atlas.scalar_load"(%34) <{base = 0 : i32, dst = 3 : i32, kind = "seli", offset = 127 : i32}> : (!atlas.state) -> !atlas.state
  %36 = "atlas.vpu_pack"(%35) <{direction = "bf16_to_fp8", dst = 1 : i32, scale_reg = 3 : i32, src = 34 : i32}> : (!atlas.state) -> !atlas.state
  %37 = "atlas.delay"(%36) <{cycles = 256 : i32}> {atlas.delay_reason = "vpu_pack_completion"} : (!atlas.state) -> !atlas.state
  %38 = "atlas.upper"(%37) <{dst = 8 : i32, immediate = 8 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %39 = "atlas.alu_imm"(%38) <{dst = 8 : i32, immediate = 0 : i32, kind = "addi", src = 8 : i32}> : (!atlas.state) -> !atlas.state
  %40 = "atlas.vstore"(%39) <{base = 8 : i32, format = "raw", offset = 0 : i32, src = 1 : i32}> : (!atlas.state) -> !atlas.state
  %41 = "atlas.delay"(%40) <{cycles = 256 : i32}> {atlas.delay_reason = "pack_vstore_completion"} : (!atlas.state) -> !atlas.state
  %42 = "atlas.upper"(%41) <{dst = 10 : i32, immediate = 32 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %43 = "atlas.alu_imm"(%42) <{dst = 10 : i32, immediate = 0 : i32, kind = "addi", src = 10 : i32}> : (!atlas.state) -> !atlas.state
  %44 = "atlas.upper"(%43) <{dst = 11 : i32, immediate = 32 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %45 = "atlas.alu_imm"(%44) <{dst = 11 : i32, immediate = 512 : i32, kind = "addi", src = 11 : i32}> : (!atlas.state) -> !atlas.state
  %46 = "atlas.upper"(%45) <{dst = 12 : i32, immediate = 32 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %47 = "atlas.alu_imm"(%46) <{dst = 12 : i32, immediate = 1024 : i32, kind = "addi", src = 12 : i32}> : (!atlas.state) -> !atlas.state
  %48 = "atlas.alu_imm"(%47) <{dst = 13 : i32, immediate = 0 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %49 = "atlas.alu_imm"(%48) <{dst = 14 : i32, immediate = 32 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %50 = "atlas.scalar_load"(%49) <{base = 10 : i32, dst = 16 : i32, kind = "lw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %51 = "atlas.delay"(%50) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %52 = "atlas.scalar_store"(%51) <{base = 12 : i32, kind = "sw", offset = 0 : i32, src = 16 : i32}> : (!atlas.state) -> !atlas.state
  %53 = "atlas.scalar_load"(%52) <{base = 11 : i32, dst = 17 : i32, kind = "lw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %54 = "atlas.delay"(%53) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %55 = "atlas.scalar_store"(%54) <{base = 12 : i32, kind = "sw", offset = 16 : i32, src = 17 : i32}> : (!atlas.state) -> !atlas.state
  %56 = "atlas.scalar_load"(%55) <{base = 10 : i32, dst = 16 : i32, kind = "lw", offset = 4 : i32}> : (!atlas.state) -> !atlas.state
  %57 = "atlas.delay"(%56) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %58 = "atlas.scalar_store"(%57) <{base = 12 : i32, kind = "sw", offset = 4 : i32, src = 16 : i32}> : (!atlas.state) -> !atlas.state
  %59 = "atlas.scalar_load"(%58) <{base = 11 : i32, dst = 17 : i32, kind = "lw", offset = 4 : i32}> : (!atlas.state) -> !atlas.state
  %60 = "atlas.delay"(%59) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %61 = "atlas.scalar_store"(%60) <{base = 12 : i32, kind = "sw", offset = 20 : i32, src = 17 : i32}> : (!atlas.state) -> !atlas.state
  %62 = "atlas.scalar_load"(%61) <{base = 10 : i32, dst = 16 : i32, kind = "lw", offset = 8 : i32}> : (!atlas.state) -> !atlas.state
  %63 = "atlas.delay"(%62) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %64 = "atlas.scalar_store"(%63) <{base = 12 : i32, kind = "sw", offset = 8 : i32, src = 16 : i32}> : (!atlas.state) -> !atlas.state
  %65 = "atlas.scalar_load"(%64) <{base = 11 : i32, dst = 17 : i32, kind = "lw", offset = 8 : i32}> : (!atlas.state) -> !atlas.state
  %66 = "atlas.delay"(%65) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %67 = "atlas.scalar_store"(%66) <{base = 12 : i32, kind = "sw", offset = 24 : i32, src = 17 : i32}> : (!atlas.state) -> !atlas.state
  %68 = "atlas.scalar_load"(%67) <{base = 10 : i32, dst = 16 : i32, kind = "lw", offset = 12 : i32}> : (!atlas.state) -> !atlas.state
  %69 = "atlas.delay"(%68) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %70 = "atlas.scalar_store"(%69) <{base = 12 : i32, kind = "sw", offset = 12 : i32, src = 16 : i32}> : (!atlas.state) -> !atlas.state
  %71 = "atlas.scalar_load"(%70) <{base = 11 : i32, dst = 17 : i32, kind = "lw", offset = 12 : i32}> : (!atlas.state) -> !atlas.state
  %72 = "atlas.delay"(%71) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %73 = "atlas.scalar_store"(%72) <{base = 12 : i32, kind = "sw", offset = 28 : i32, src = 17 : i32}> : (!atlas.state) -> !atlas.state
  %74 = "atlas.alu_imm"(%73) <{dst = 10 : i32, immediate = 16 : i32, kind = "addi", src = 10 : i32}> : (!atlas.state) -> !atlas.state
  %75 = "atlas.alu_imm"(%74) <{dst = 11 : i32, immediate = 16 : i32, kind = "addi", src = 11 : i32}> : (!atlas.state) -> !atlas.state
  %76 = "atlas.alu_imm"(%75) <{dst = 12 : i32, immediate = 32 : i32, kind = "addi", src = 12 : i32}> : (!atlas.state) -> !atlas.state
  %77 = "atlas.alu_imm"(%76) <{dst = 13 : i32, immediate = 1 : i32, kind = "addi", src = 13 : i32}> : (!atlas.state) -> !atlas.state
  %78 = "atlas.branch"(%77) <{kind = "blt", lhs = 13 : i32, offset_bytes = -56 : i32, rhs = 14 : i32}> : (!atlas.state) -> !atlas.state
  %79 = "atlas.alu_imm"(%78) <{dst = 0 : i32, immediate = 0 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %80 = "atlas.delay"(%79) <{cycles = 256 : i32}> {atlas.delay_reason = "scalar_relayout_completion"} : (!atlas.state) -> !atlas.state
  %81 = "atlas.upper"(%80) <{dst = 6 : i32, immediate = 8 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %82 = "atlas.alu_imm"(%81) <{dst = 6 : i32, immediate = 256 : i32, kind = "addi", src = 6 : i32}> : (!atlas.state) -> !atlas.state
  %83 = "atlas.vload"(%82) <{base = 6 : i32, dst = 1 : i32, format = "raw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %84 = "atlas.delay"(%83) <{cycles = 256 : i32}> {atlas.delay_reason = "pack_relayout_vload_completion"} : (!atlas.state) -> !atlas.state
  %85 = "atlas.mxu_push"(%84) <{kind = "weight_fp8", slot = 0 : i32, src = 0 : i32, unit = 1 : i32}> : (!atlas.state) -> !atlas.state
  %86 = "atlas.delay"(%85) <{cycles = 256 : i32}> {atlas.delay_reason = "mxu_weight_completion"} : (!atlas.state) -> !atlas.state
  %87 = "atlas.mxu_matmul"(%86) <{acc_slot = 0 : i32, accumulate = false, src = 1 : i32, unit = 1 : i32, weight_slot = 0 : i32}> : (!atlas.state) -> !atlas.state
  %88 = "atlas.delay"(%87) <{cycles = 256 : i32}> {atlas.delay_reason = "mxu_matmul_completion"} : (!atlas.state) -> !atlas.state
  %89 = "atlas.mxu_pop"(%88) <{dst = 32 : i32, format = "bf16", scale_reg = 0 : i32, slot = 0 : i32, unit = 1 : i32}> : (!atlas.state) -> !atlas.state
  %90 = "atlas.delay"(%89) <{cycles = 256 : i32}> {atlas.delay_reason = "mxu_readout_completion"} : (!atlas.state) -> !atlas.state
  %91 = "atlas.upper"(%90) <{dst = 8 : i32, immediate = 16 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %92 = "atlas.alu_imm"(%91) <{dst = 8 : i32, immediate = 0 : i32, kind = "addi", src = 8 : i32}> : (!atlas.state) -> !atlas.state
  %93 = "atlas.vstore"(%92) <{base = 8 : i32, format = "raw", offset = 0 : i32, src = 32 : i32}> : (!atlas.state) -> !atlas.state
  %94 = "atlas.delay"(%93) <{cycles = 256 : i32}> {atlas.delay_reason = "vstore_completion"} : (!atlas.state) -> !atlas.state
  %95 = "atlas.upper"(%94) <{dst = 3 : i32, immediate = 589826 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %96 = "atlas.alu_imm"(%95) <{dst = 3 : i32, immediate = 0 : i32, kind = "addi", src = 3 : i32}> : (!atlas.state) -> !atlas.state
  %97 = "atlas.dma"(%96) <{channel = 1 : i32, direction = "store", dram = 3 : i32, reg = 8 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %98 = "atlas.dma_wait"(%97) <{channel = 1 : i32}> : (!atlas.state) -> !atlas.state
  %99 = "atlas.upper"(%98) <{dst = 8 : i32, immediate = 16 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %100 = "atlas.alu_imm"(%99) <{dst = 8 : i32, immediate = 256 : i32, kind = "addi", src = 8 : i32}> : (!atlas.state) -> !atlas.state
  %101 = "atlas.vstore"(%100) <{base = 8 : i32, format = "raw", offset = 0 : i32, src = 33 : i32}> : (!atlas.state) -> !atlas.state
  %102 = "atlas.delay"(%101) <{cycles = 256 : i32}> {atlas.delay_reason = "vstore_completion"} : (!atlas.state) -> !atlas.state
  %103 = "atlas.upper"(%102) <{dst = 3 : i32, immediate = 589826 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %104 = "atlas.alu_imm"(%103) <{dst = 3 : i32, immediate = 1024 : i32, kind = "addi", src = 3 : i32}> : (!atlas.state) -> !atlas.state
  %105 = "atlas.dma"(%104) <{channel = 1 : i32, direction = "store", dram = 3 : i32, reg = 8 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %106 = "atlas.dma_wait"(%105) <{channel = 1 : i32}> : (!atlas.state) -> !atlas.state
  %107 = "atlas.alu_imm"(%106) <{dst = 1 : i32, immediate = 1 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %108 = "atlas.csr"(%107) <{address = 3088 : i32, dst = 0 : i32, kind = "rrw", source = 1 : i32}> : (!atlas.state) -> !atlas.state
  %109 = "atlas.trap"(%108) <{kind = "ecall"}> : (!atlas.state) -> !atlas.state
}
