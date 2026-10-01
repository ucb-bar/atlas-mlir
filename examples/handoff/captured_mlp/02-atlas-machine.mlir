module attributes {atlas.generated_from_virtual, atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415984640 : i64, atlas.scalar_arg_regs = array<i32>} {
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
  %11 = "atlas.vload"(%10) <{base = 6 : i32, dst = 0 : i32, format = "raw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %12 = "atlas.delay"(%11) <{cycles = 256 : i32}> {atlas.delay_reason = "vload_completion"} : (!atlas.state) -> !atlas.state
  %13 = "atlas.alu_imm"(%12) <{dst = 6 : i32, immediate = 512 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %14 = "atlas.upper"(%13) <{dst = 1 : i32, immediate = 589825 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %15 = "atlas.alu_imm"(%14) <{dst = 1 : i32, immediate = -2048 : i32, kind = "addi", src = 1 : i32}> : (!atlas.state) -> !atlas.state
  %16 = "atlas.dma"(%15) <{channel = 0 : i32, direction = "load", dram = 1 : i32, reg = 6 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %17 = "atlas.dma_wait"(%16) <{channel = 0 : i32}> : (!atlas.state) -> !atlas.state
  %18 = "atlas.vload"(%17) <{base = 6 : i32, dst = 1 : i32, format = "raw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %19 = "atlas.delay"(%18) <{cycles = 256 : i32}> {atlas.delay_reason = "vload_completion"} : (!atlas.state) -> !atlas.state
  %20 = "atlas.mxu_push"(%19) <{kind = "weight_fp8", slot = 0 : i32, src = 1 : i32, unit = 0 : i32}> : (!atlas.state) -> !atlas.state
  %21 = "atlas.delay"(%20) <{cycles = 256 : i32}> {atlas.delay_reason = "mxu_weight_completion"} : (!atlas.state) -> !atlas.state
  %22 = "atlas.mxu_matmul"(%21) <{acc_slot = 0 : i32, accumulate = false, src = 0 : i32, unit = 0 : i32, weight_slot = 0 : i32}> : (!atlas.state) -> !atlas.state
  %23 = "atlas.delay"(%22) <{cycles = 256 : i32}> {atlas.delay_reason = "mxu_matmul_completion"} : (!atlas.state) -> !atlas.state
  %24 = "atlas.mxu_pop"(%23) <{dst = 34 : i32, format = "bf16", scale_reg = 0 : i32, slot = 0 : i32, unit = 0 : i32}> : (!atlas.state) -> !atlas.state
  %25 = "atlas.delay"(%24) <{cycles = 256 : i32}> {atlas.delay_reason = "mxu_readout_completion"} : (!atlas.state) -> !atlas.state
  %26 = "atlas.alu_imm"(%25) <{dst = 6 : i32, immediate = 1024 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %27 = "atlas.upper"(%26) <{dst = 1 : i32, immediate = 589825 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %28 = "atlas.alu_imm"(%27) <{dst = 1 : i32, immediate = 0 : i32, kind = "addi", src = 1 : i32}> : (!atlas.state) -> !atlas.state
  %29 = "atlas.dma"(%28) <{channel = 0 : i32, direction = "load", dram = 1 : i32, reg = 6 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %30 = "atlas.dma_wait"(%29) <{channel = 0 : i32}> : (!atlas.state) -> !atlas.state
  %31 = "atlas.vload"(%30) <{base = 6 : i32, dst = 36 : i32, format = "raw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %32 = "atlas.delay"(%31) <{cycles = 256 : i32}> {atlas.delay_reason = "vload_completion"} : (!atlas.state) -> !atlas.state
  %33 = "atlas.alu_imm"(%32) <{dst = 6 : i32, immediate = 1280 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %34 = "atlas.upper"(%33) <{dst = 1 : i32, immediate = 589825 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %35 = "atlas.alu_imm"(%34) <{dst = 1 : i32, immediate = 1024 : i32, kind = "addi", src = 1 : i32}> : (!atlas.state) -> !atlas.state
  %36 = "atlas.dma"(%35) <{channel = 0 : i32, direction = "load", dram = 1 : i32, reg = 6 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %37 = "atlas.dma_wait"(%36) <{channel = 0 : i32}> : (!atlas.state) -> !atlas.state
  %38 = "atlas.vload"(%37) <{base = 6 : i32, dst = 37 : i32, format = "raw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %39 = "atlas.delay"(%38) <{cycles = 256 : i32}> {atlas.delay_reason = "vload_completion"} : (!atlas.state) -> !atlas.state
  %40 = "atlas.vpu_binary"(%39) <{dst = 32 : i32, kind = "add", lhs = 34 : i32, rhs = 36 : i32}> : (!atlas.state) -> !atlas.state
  %41 = "atlas.delay"(%40) <{cycles = 256 : i32}> {atlas.delay_reason = "vpu_completion"} : (!atlas.state) -> !atlas.state
  %42 = "atlas.vpu_unary"(%41) <{dst = 34 : i32, kind = "relu", src = 32 : i32}> : (!atlas.state) -> !atlas.state
  %43 = "atlas.delay"(%42) <{cycles = 256 : i32}> {atlas.delay_reason = "vpu_completion"} : (!atlas.state) -> !atlas.state
  %44 = "atlas.scalar_load"(%43) <{base = 0 : i32, dst = 3 : i32, kind = "seli", offset = 127 : i32}> : (!atlas.state) -> !atlas.state
  %45 = "atlas.vpu_pack"(%44) <{direction = "bf16_to_fp8", dst = 0 : i32, scale_reg = 3 : i32, src = 34 : i32}> : (!atlas.state) -> !atlas.state
  %46 = "atlas.delay"(%45) <{cycles = 256 : i32}> {atlas.delay_reason = "vpu_pack_completion"} : (!atlas.state) -> !atlas.state
  %47 = "atlas.upper"(%46) <{dst = 8 : i32, immediate = 8 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %48 = "atlas.alu_imm"(%47) <{dst = 8 : i32, immediate = 0 : i32, kind = "addi", src = 8 : i32}> : (!atlas.state) -> !atlas.state
  %49 = "atlas.vstore"(%48) <{base = 8 : i32, format = "raw", offset = 0 : i32, src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %50 = "atlas.delay"(%49) <{cycles = 256 : i32}> {atlas.delay_reason = "pack_vstore_completion"} : (!atlas.state) -> !atlas.state
  %51 = "atlas.upper"(%50) <{dst = 10 : i32, immediate = 32 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %52 = "atlas.alu_imm"(%51) <{dst = 10 : i32, immediate = 0 : i32, kind = "addi", src = 10 : i32}> : (!atlas.state) -> !atlas.state
  %53 = "atlas.upper"(%52) <{dst = 11 : i32, immediate = 32 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %54 = "atlas.alu_imm"(%53) <{dst = 11 : i32, immediate = 512 : i32, kind = "addi", src = 11 : i32}> : (!atlas.state) -> !atlas.state
  %55 = "atlas.upper"(%54) <{dst = 12 : i32, immediate = 32 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %56 = "atlas.alu_imm"(%55) <{dst = 12 : i32, immediate = 1024 : i32, kind = "addi", src = 12 : i32}> : (!atlas.state) -> !atlas.state
  %57 = "atlas.alu_imm"(%56) <{dst = 13 : i32, immediate = 0 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %58 = "atlas.alu_imm"(%57) <{dst = 14 : i32, immediate = 32 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %59 = "atlas.scalar_load"(%58) <{base = 10 : i32, dst = 16 : i32, kind = "lw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %60 = "atlas.delay"(%59) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %61 = "atlas.scalar_store"(%60) <{base = 12 : i32, kind = "sw", offset = 0 : i32, src = 16 : i32}> : (!atlas.state) -> !atlas.state
  %62 = "atlas.scalar_load"(%61) <{base = 11 : i32, dst = 17 : i32, kind = "lw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %63 = "atlas.delay"(%62) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %64 = "atlas.scalar_store"(%63) <{base = 12 : i32, kind = "sw", offset = 16 : i32, src = 17 : i32}> : (!atlas.state) -> !atlas.state
  %65 = "atlas.scalar_load"(%64) <{base = 10 : i32, dst = 16 : i32, kind = "lw", offset = 4 : i32}> : (!atlas.state) -> !atlas.state
  %66 = "atlas.delay"(%65) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %67 = "atlas.scalar_store"(%66) <{base = 12 : i32, kind = "sw", offset = 4 : i32, src = 16 : i32}> : (!atlas.state) -> !atlas.state
  %68 = "atlas.scalar_load"(%67) <{base = 11 : i32, dst = 17 : i32, kind = "lw", offset = 4 : i32}> : (!atlas.state) -> !atlas.state
  %69 = "atlas.delay"(%68) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %70 = "atlas.scalar_store"(%69) <{base = 12 : i32, kind = "sw", offset = 20 : i32, src = 17 : i32}> : (!atlas.state) -> !atlas.state
  %71 = "atlas.scalar_load"(%70) <{base = 10 : i32, dst = 16 : i32, kind = "lw", offset = 8 : i32}> : (!atlas.state) -> !atlas.state
  %72 = "atlas.delay"(%71) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %73 = "atlas.scalar_store"(%72) <{base = 12 : i32, kind = "sw", offset = 8 : i32, src = 16 : i32}> : (!atlas.state) -> !atlas.state
  %74 = "atlas.scalar_load"(%73) <{base = 11 : i32, dst = 17 : i32, kind = "lw", offset = 8 : i32}> : (!atlas.state) -> !atlas.state
  %75 = "atlas.delay"(%74) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %76 = "atlas.scalar_store"(%75) <{base = 12 : i32, kind = "sw", offset = 24 : i32, src = 17 : i32}> : (!atlas.state) -> !atlas.state
  %77 = "atlas.scalar_load"(%76) <{base = 10 : i32, dst = 16 : i32, kind = "lw", offset = 12 : i32}> : (!atlas.state) -> !atlas.state
  %78 = "atlas.delay"(%77) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %79 = "atlas.scalar_store"(%78) <{base = 12 : i32, kind = "sw", offset = 12 : i32, src = 16 : i32}> : (!atlas.state) -> !atlas.state
  %80 = "atlas.scalar_load"(%79) <{base = 11 : i32, dst = 17 : i32, kind = "lw", offset = 12 : i32}> : (!atlas.state) -> !atlas.state
  %81 = "atlas.delay"(%80) <{cycles = 8 : i32}> {atlas.delay_reason = "scalar_load_completion"} : (!atlas.state) -> !atlas.state
  %82 = "atlas.scalar_store"(%81) <{base = 12 : i32, kind = "sw", offset = 28 : i32, src = 17 : i32}> : (!atlas.state) -> !atlas.state
  %83 = "atlas.alu_imm"(%82) <{dst = 10 : i32, immediate = 16 : i32, kind = "addi", src = 10 : i32}> : (!atlas.state) -> !atlas.state
  %84 = "atlas.alu_imm"(%83) <{dst = 11 : i32, immediate = 16 : i32, kind = "addi", src = 11 : i32}> : (!atlas.state) -> !atlas.state
  %85 = "atlas.alu_imm"(%84) <{dst = 12 : i32, immediate = 32 : i32, kind = "addi", src = 12 : i32}> : (!atlas.state) -> !atlas.state
  %86 = "atlas.alu_imm"(%85) <{dst = 13 : i32, immediate = 1 : i32, kind = "addi", src = 13 : i32}> : (!atlas.state) -> !atlas.state
  %87 = "atlas.branch"(%86) <{kind = "blt", lhs = 13 : i32, offset_bytes = -56 : i32, rhs = 14 : i32}> : (!atlas.state) -> !atlas.state
  %88 = "atlas.alu_imm"(%87) <{dst = 0 : i32, immediate = 0 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %89 = "atlas.delay"(%88) <{cycles = 256 : i32}> {atlas.delay_reason = "scalar_relayout_completion"} : (!atlas.state) -> !atlas.state
  %90 = "atlas.upper"(%89) <{dst = 6 : i32, immediate = 8 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %91 = "atlas.alu_imm"(%90) <{dst = 6 : i32, immediate = 256 : i32, kind = "addi", src = 6 : i32}> : (!atlas.state) -> !atlas.state
  %92 = "atlas.vload"(%91) <{base = 6 : i32, dst = 0 : i32, format = "raw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %93 = "atlas.delay"(%92) <{cycles = 256 : i32}> {atlas.delay_reason = "pack_relayout_vload_completion"} : (!atlas.state) -> !atlas.state
  %94 = "atlas.alu_imm"(%93) <{dst = 6 : i32, immediate = 1536 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %95 = "atlas.upper"(%94) <{dst = 1 : i32, immediate = 589826 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %96 = "atlas.alu_imm"(%95) <{dst = 1 : i32, immediate = -2048 : i32, kind = "addi", src = 1 : i32}> : (!atlas.state) -> !atlas.state
  %97 = "atlas.dma"(%96) <{channel = 0 : i32, direction = "load", dram = 1 : i32, reg = 6 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %98 = "atlas.dma_wait"(%97) <{channel = 0 : i32}> : (!atlas.state) -> !atlas.state
  %99 = "atlas.vload"(%98) <{base = 6 : i32, dst = 1 : i32, format = "raw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %100 = "atlas.delay"(%99) <{cycles = 256 : i32}> {atlas.delay_reason = "vload_completion"} : (!atlas.state) -> !atlas.state
  %101 = "atlas.mxu_push"(%100) <{kind = "weight_fp8", slot = 0 : i32, src = 1 : i32, unit = 1 : i32}> : (!atlas.state) -> !atlas.state
  %102 = "atlas.delay"(%101) <{cycles = 256 : i32}> {atlas.delay_reason = "mxu_weight_completion"} : (!atlas.state) -> !atlas.state
  %103 = "atlas.mxu_matmul"(%102) <{acc_slot = 0 : i32, accumulate = false, src = 0 : i32, unit = 1 : i32, weight_slot = 0 : i32}> : (!atlas.state) -> !atlas.state
  %104 = "atlas.delay"(%103) <{cycles = 256 : i32}> {atlas.delay_reason = "mxu_matmul_completion"} : (!atlas.state) -> !atlas.state
  %105 = "atlas.mxu_pop"(%104) <{dst = 32 : i32, format = "bf16", scale_reg = 0 : i32, slot = 0 : i32, unit = 1 : i32}> : (!atlas.state) -> !atlas.state
  %106 = "atlas.delay"(%105) <{cycles = 256 : i32}> {atlas.delay_reason = "mxu_readout_completion"} : (!atlas.state) -> !atlas.state
  %107 = "atlas.upper"(%106) <{dst = 6 : i32, immediate = 1 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %108 = "atlas.alu_imm"(%107) <{dst = 6 : i32, immediate = -2048 : i32, kind = "addi", src = 6 : i32}> : (!atlas.state) -> !atlas.state
  %109 = "atlas.upper"(%108) <{dst = 1 : i32, immediate = 589826 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %110 = "atlas.alu_imm"(%109) <{dst = 1 : i32, immediate = 0 : i32, kind = "addi", src = 1 : i32}> : (!atlas.state) -> !atlas.state
  %111 = "atlas.dma"(%110) <{channel = 0 : i32, direction = "load", dram = 1 : i32, reg = 6 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %112 = "atlas.dma_wait"(%111) <{channel = 0 : i32}> : (!atlas.state) -> !atlas.state
  %113 = "atlas.vload"(%112) <{base = 6 : i32, dst = 34 : i32, format = "raw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %114 = "atlas.delay"(%113) <{cycles = 256 : i32}> {atlas.delay_reason = "vload_completion"} : (!atlas.state) -> !atlas.state
  %115 = "atlas.upper"(%114) <{dst = 6 : i32, immediate = 1 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %116 = "atlas.alu_imm"(%115) <{dst = 6 : i32, immediate = -1792 : i32, kind = "addi", src = 6 : i32}> : (!atlas.state) -> !atlas.state
  %117 = "atlas.upper"(%116) <{dst = 1 : i32, immediate = 589826 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %118 = "atlas.alu_imm"(%117) <{dst = 1 : i32, immediate = 1024 : i32, kind = "addi", src = 1 : i32}> : (!atlas.state) -> !atlas.state
  %119 = "atlas.dma"(%118) <{channel = 0 : i32, direction = "load", dram = 1 : i32, reg = 6 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %120 = "atlas.dma_wait"(%119) <{channel = 0 : i32}> : (!atlas.state) -> !atlas.state
  %121 = "atlas.vload"(%120) <{base = 6 : i32, dst = 35 : i32, format = "raw", offset = 0 : i32}> : (!atlas.state) -> !atlas.state
  %122 = "atlas.delay"(%121) <{cycles = 256 : i32}> {atlas.delay_reason = "vload_completion"} : (!atlas.state) -> !atlas.state
  %123 = "atlas.vpu_binary"(%122) <{dst = 36 : i32, kind = "add", lhs = 32 : i32, rhs = 34 : i32}> : (!atlas.state) -> !atlas.state
  %124 = "atlas.delay"(%123) <{cycles = 256 : i32}> {atlas.delay_reason = "vpu_completion"} : (!atlas.state) -> !atlas.state
  %125 = "atlas.upper"(%124) <{dst = 8 : i32, immediate = 16 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %126 = "atlas.alu_imm"(%125) <{dst = 8 : i32, immediate = 0 : i32, kind = "addi", src = 8 : i32}> : (!atlas.state) -> !atlas.state
  %127 = "atlas.vstore"(%126) <{base = 8 : i32, format = "raw", offset = 0 : i32, src = 36 : i32}> : (!atlas.state) -> !atlas.state
  %128 = "atlas.delay"(%127) <{cycles = 256 : i32}> {atlas.delay_reason = "vstore_completion"} : (!atlas.state) -> !atlas.state
  %129 = "atlas.upper"(%128) <{dst = 3 : i32, immediate = 589840 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %130 = "atlas.alu_imm"(%129) <{dst = 3 : i32, immediate = 0 : i32, kind = "addi", src = 3 : i32}> : (!atlas.state) -> !atlas.state
  %131 = "atlas.dma"(%130) <{channel = 1 : i32, direction = "store", dram = 3 : i32, reg = 8 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %132 = "atlas.dma_wait"(%131) <{channel = 1 : i32}> : (!atlas.state) -> !atlas.state
  %133 = "atlas.upper"(%132) <{dst = 8 : i32, immediate = 16 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %134 = "atlas.alu_imm"(%133) <{dst = 8 : i32, immediate = 256 : i32, kind = "addi", src = 8 : i32}> : (!atlas.state) -> !atlas.state
  %135 = "atlas.vstore"(%134) <{base = 8 : i32, format = "raw", offset = 0 : i32, src = 37 : i32}> : (!atlas.state) -> !atlas.state
  %136 = "atlas.delay"(%135) <{cycles = 256 : i32}> {atlas.delay_reason = "vstore_completion"} : (!atlas.state) -> !atlas.state
  %137 = "atlas.upper"(%136) <{dst = 3 : i32, immediate = 589840 : i32, kind = "lui"}> : (!atlas.state) -> !atlas.state
  %138 = "atlas.alu_imm"(%137) <{dst = 3 : i32, immediate = 1024 : i32, kind = "addi", src = 3 : i32}> : (!atlas.state) -> !atlas.state
  %139 = "atlas.dma"(%138) <{channel = 1 : i32, direction = "store", dram = 3 : i32, reg = 8 : i32, size = 2 : i32}> : (!atlas.state) -> !atlas.state
  %140 = "atlas.dma_wait"(%139) <{channel = 1 : i32}> : (!atlas.state) -> !atlas.state
  %141 = "atlas.alu_imm"(%140) <{dst = 1 : i32, immediate = 1 : i32, kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state
  %142 = "atlas.csr"(%141) <{address = 3088 : i32, dst = 0 : i32, kind = "rrw", source = 1 : i32}> : (!atlas.state) -> !atlas.state
  %143 = "atlas.trap"(%142) <{kind = "ecall"}> : (!atlas.state) -> !atlas.state
}
