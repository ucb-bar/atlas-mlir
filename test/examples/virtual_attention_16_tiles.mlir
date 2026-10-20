module {
  func.func @attention_tiles() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415984640 : i64} {
    %io0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %q_addr_t0 = arith.constant -1879044096 : i32
    %k_addr_t0 = arith.constant -1879047168 : i32
    %v_addr_t0 = arith.constant -1879046144 : i32
    %out_addr_t0 = arith.constant -1878982656 : i32
    %fp8_size_t0 = arith.constant 1024 : i32
    %bf16_size_t0 = arith.constant 2048 : i32
    %io1_t0, %q_t0 = "atlas.virtual_dma_load_fp8"(%io0, %q_addr_t0, %fp8_size_t0) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t0, %k_t0 = "atlas.virtual_dma_load_fp8"(%io1_t0, %k_addr_t0, %fp8_size_t0) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t0, %v_t0 = "atlas.virtual_dma_load_fp8"(%io2_t0, %v_addr_t0, %fp8_size_t0) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t0, %k_weight_t0 = "atlas.virtual_mxu_load_weight"(%io3_t0, %k_t0) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t0, %scores_acc_t0 = "atlas.virtual_mxu_reset"(%io4_t0, %q_t0, %k_weight_t0) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t0, %s_t0 = "atlas.virtual_mxu_readout_bf16"(%io5_t0, %scores_acc_t0) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t0 = "atlas.virtual_vpu_reduce"(%s_t0) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t0 = "atlas.virtual_vpu_binary"(%s_t0, %max_t0) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t0 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t0 = "atlas.virtual_vpu_binary"(%shifted_t0, %scale_t0) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t0 = "atlas.virtual_vpu_unary"(%scaled_t0) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t0 = "atlas.virtual_vpu_reduce"(%e_t0) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t0 = "atlas.virtual_vpu_unary"(%sum_t0) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t0 = "atlas.virtual_vpu_binary"(%e_t0, %inverse_t0) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t0 = "atlas.virtual_pack_fp8"(%p_t0) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t0, %v_weight_t0 = "atlas.virtual_mxu_load_weight"(%io6_t0, %v_t0) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t0, %out_acc_t0 = "atlas.virtual_mxu_reset"(%io7_t0, %p8_t0, %v_weight_t0) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t0, %out_t0 = "atlas.virtual_mxu_readout_bf16"(%io8_t0, %out_acc_t0) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t0 = "atlas.virtual_dma_store_bf16"(%io9_t0, %out_t0, %out_addr_t0, %bf16_size_t0) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t1 = arith.constant -1879042048 : i32
    %k_addr_t1 = arith.constant -1879047168 : i32
    %v_addr_t1 = arith.constant -1879046144 : i32
    %out_addr_t1 = arith.constant -1878980608 : i32
    %fp8_size_t1 = arith.constant 1024 : i32
    %bf16_size_t1 = arith.constant 2048 : i32
    %io1_t1, %q_t1 = "atlas.virtual_dma_load_fp8"(%io10_t0, %q_addr_t1, %fp8_size_t1) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t1, %k_t1 = "atlas.virtual_dma_load_fp8"(%io1_t1, %k_addr_t1, %fp8_size_t1) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t1, %v_t1 = "atlas.virtual_dma_load_fp8"(%io2_t1, %v_addr_t1, %fp8_size_t1) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t1, %k_weight_t1 = "atlas.virtual_mxu_load_weight"(%io3_t1, %k_t1) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t1, %scores_acc_t1 = "atlas.virtual_mxu_reset"(%io4_t1, %q_t1, %k_weight_t1) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t1, %s_t1 = "atlas.virtual_mxu_readout_bf16"(%io5_t1, %scores_acc_t1) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t1 = "atlas.virtual_vpu_reduce"(%s_t1) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t1 = "atlas.virtual_vpu_binary"(%s_t1, %max_t1) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t1 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t1 = "atlas.virtual_vpu_binary"(%shifted_t1, %scale_t1) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t1 = "atlas.virtual_vpu_unary"(%scaled_t1) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t1 = "atlas.virtual_vpu_reduce"(%e_t1) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t1 = "atlas.virtual_vpu_unary"(%sum_t1) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t1 = "atlas.virtual_vpu_binary"(%e_t1, %inverse_t1) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t1 = "atlas.virtual_pack_fp8"(%p_t1) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t1, %v_weight_t1 = "atlas.virtual_mxu_load_weight"(%io6_t1, %v_t1) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t1, %out_acc_t1 = "atlas.virtual_mxu_reset"(%io7_t1, %p8_t1, %v_weight_t1) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t1, %out_t1 = "atlas.virtual_mxu_readout_bf16"(%io8_t1, %out_acc_t1) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t1 = "atlas.virtual_dma_store_bf16"(%io9_t1, %out_t1, %out_addr_t1, %bf16_size_t1) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t2 = arith.constant -1879040000 : i32
    %k_addr_t2 = arith.constant -1879047168 : i32
    %v_addr_t2 = arith.constant -1879046144 : i32
    %out_addr_t2 = arith.constant -1878978560 : i32
    %fp8_size_t2 = arith.constant 1024 : i32
    %bf16_size_t2 = arith.constant 2048 : i32
    %io1_t2, %q_t2 = "atlas.virtual_dma_load_fp8"(%io10_t1, %q_addr_t2, %fp8_size_t2) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t2, %k_t2 = "atlas.virtual_dma_load_fp8"(%io1_t2, %k_addr_t2, %fp8_size_t2) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t2, %v_t2 = "atlas.virtual_dma_load_fp8"(%io2_t2, %v_addr_t2, %fp8_size_t2) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t2, %k_weight_t2 = "atlas.virtual_mxu_load_weight"(%io3_t2, %k_t2) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t2, %scores_acc_t2 = "atlas.virtual_mxu_reset"(%io4_t2, %q_t2, %k_weight_t2) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t2, %s_t2 = "atlas.virtual_mxu_readout_bf16"(%io5_t2, %scores_acc_t2) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t2 = "atlas.virtual_vpu_reduce"(%s_t2) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t2 = "atlas.virtual_vpu_binary"(%s_t2, %max_t2) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t2 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t2 = "atlas.virtual_vpu_binary"(%shifted_t2, %scale_t2) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t2 = "atlas.virtual_vpu_unary"(%scaled_t2) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t2 = "atlas.virtual_vpu_reduce"(%e_t2) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t2 = "atlas.virtual_vpu_unary"(%sum_t2) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t2 = "atlas.virtual_vpu_binary"(%e_t2, %inverse_t2) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t2 = "atlas.virtual_pack_fp8"(%p_t2) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t2, %v_weight_t2 = "atlas.virtual_mxu_load_weight"(%io6_t2, %v_t2) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t2, %out_acc_t2 = "atlas.virtual_mxu_reset"(%io7_t2, %p8_t2, %v_weight_t2) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t2, %out_t2 = "atlas.virtual_mxu_readout_bf16"(%io8_t2, %out_acc_t2) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t2 = "atlas.virtual_dma_store_bf16"(%io9_t2, %out_t2, %out_addr_t2, %bf16_size_t2) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t3 = arith.constant -1879037952 : i32
    %k_addr_t3 = arith.constant -1879047168 : i32
    %v_addr_t3 = arith.constant -1879046144 : i32
    %out_addr_t3 = arith.constant -1878976512 : i32
    %fp8_size_t3 = arith.constant 1024 : i32
    %bf16_size_t3 = arith.constant 2048 : i32
    %io1_t3, %q_t3 = "atlas.virtual_dma_load_fp8"(%io10_t2, %q_addr_t3, %fp8_size_t3) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t3, %k_t3 = "atlas.virtual_dma_load_fp8"(%io1_t3, %k_addr_t3, %fp8_size_t3) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t3, %v_t3 = "atlas.virtual_dma_load_fp8"(%io2_t3, %v_addr_t3, %fp8_size_t3) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t3, %k_weight_t3 = "atlas.virtual_mxu_load_weight"(%io3_t3, %k_t3) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t3, %scores_acc_t3 = "atlas.virtual_mxu_reset"(%io4_t3, %q_t3, %k_weight_t3) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t3, %s_t3 = "atlas.virtual_mxu_readout_bf16"(%io5_t3, %scores_acc_t3) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t3 = "atlas.virtual_vpu_reduce"(%s_t3) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t3 = "atlas.virtual_vpu_binary"(%s_t3, %max_t3) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t3 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t3 = "atlas.virtual_vpu_binary"(%shifted_t3, %scale_t3) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t3 = "atlas.virtual_vpu_unary"(%scaled_t3) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t3 = "atlas.virtual_vpu_reduce"(%e_t3) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t3 = "atlas.virtual_vpu_unary"(%sum_t3) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t3 = "atlas.virtual_vpu_binary"(%e_t3, %inverse_t3) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t3 = "atlas.virtual_pack_fp8"(%p_t3) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t3, %v_weight_t3 = "atlas.virtual_mxu_load_weight"(%io6_t3, %v_t3) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t3, %out_acc_t3 = "atlas.virtual_mxu_reset"(%io7_t3, %p8_t3, %v_weight_t3) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t3, %out_t3 = "atlas.virtual_mxu_readout_bf16"(%io8_t3, %out_acc_t3) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t3 = "atlas.virtual_dma_store_bf16"(%io9_t3, %out_t3, %out_addr_t3, %bf16_size_t3) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t4 = arith.constant -1879035904 : i32
    %k_addr_t4 = arith.constant -1879047168 : i32
    %v_addr_t4 = arith.constant -1879046144 : i32
    %out_addr_t4 = arith.constant -1878974464 : i32
    %fp8_size_t4 = arith.constant 1024 : i32
    %bf16_size_t4 = arith.constant 2048 : i32
    %io1_t4, %q_t4 = "atlas.virtual_dma_load_fp8"(%io10_t3, %q_addr_t4, %fp8_size_t4) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t4, %k_t4 = "atlas.virtual_dma_load_fp8"(%io1_t4, %k_addr_t4, %fp8_size_t4) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t4, %v_t4 = "atlas.virtual_dma_load_fp8"(%io2_t4, %v_addr_t4, %fp8_size_t4) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t4, %k_weight_t4 = "atlas.virtual_mxu_load_weight"(%io3_t4, %k_t4) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t4, %scores_acc_t4 = "atlas.virtual_mxu_reset"(%io4_t4, %q_t4, %k_weight_t4) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t4, %s_t4 = "atlas.virtual_mxu_readout_bf16"(%io5_t4, %scores_acc_t4) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t4 = "atlas.virtual_vpu_reduce"(%s_t4) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t4 = "atlas.virtual_vpu_binary"(%s_t4, %max_t4) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t4 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t4 = "atlas.virtual_vpu_binary"(%shifted_t4, %scale_t4) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t4 = "atlas.virtual_vpu_unary"(%scaled_t4) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t4 = "atlas.virtual_vpu_reduce"(%e_t4) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t4 = "atlas.virtual_vpu_unary"(%sum_t4) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t4 = "atlas.virtual_vpu_binary"(%e_t4, %inverse_t4) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t4 = "atlas.virtual_pack_fp8"(%p_t4) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t4, %v_weight_t4 = "atlas.virtual_mxu_load_weight"(%io6_t4, %v_t4) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t4, %out_acc_t4 = "atlas.virtual_mxu_reset"(%io7_t4, %p8_t4, %v_weight_t4) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t4, %out_t4 = "atlas.virtual_mxu_readout_bf16"(%io8_t4, %out_acc_t4) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t4 = "atlas.virtual_dma_store_bf16"(%io9_t4, %out_t4, %out_addr_t4, %bf16_size_t4) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t5 = arith.constant -1879033856 : i32
    %k_addr_t5 = arith.constant -1879047168 : i32
    %v_addr_t5 = arith.constant -1879046144 : i32
    %out_addr_t5 = arith.constant -1878972416 : i32
    %fp8_size_t5 = arith.constant 1024 : i32
    %bf16_size_t5 = arith.constant 2048 : i32
    %io1_t5, %q_t5 = "atlas.virtual_dma_load_fp8"(%io10_t4, %q_addr_t5, %fp8_size_t5) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t5, %k_t5 = "atlas.virtual_dma_load_fp8"(%io1_t5, %k_addr_t5, %fp8_size_t5) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t5, %v_t5 = "atlas.virtual_dma_load_fp8"(%io2_t5, %v_addr_t5, %fp8_size_t5) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t5, %k_weight_t5 = "atlas.virtual_mxu_load_weight"(%io3_t5, %k_t5) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t5, %scores_acc_t5 = "atlas.virtual_mxu_reset"(%io4_t5, %q_t5, %k_weight_t5) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t5, %s_t5 = "atlas.virtual_mxu_readout_bf16"(%io5_t5, %scores_acc_t5) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t5 = "atlas.virtual_vpu_reduce"(%s_t5) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t5 = "atlas.virtual_vpu_binary"(%s_t5, %max_t5) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t5 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t5 = "atlas.virtual_vpu_binary"(%shifted_t5, %scale_t5) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t5 = "atlas.virtual_vpu_unary"(%scaled_t5) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t5 = "atlas.virtual_vpu_reduce"(%e_t5) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t5 = "atlas.virtual_vpu_unary"(%sum_t5) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t5 = "atlas.virtual_vpu_binary"(%e_t5, %inverse_t5) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t5 = "atlas.virtual_pack_fp8"(%p_t5) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t5, %v_weight_t5 = "atlas.virtual_mxu_load_weight"(%io6_t5, %v_t5) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t5, %out_acc_t5 = "atlas.virtual_mxu_reset"(%io7_t5, %p8_t5, %v_weight_t5) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t5, %out_t5 = "atlas.virtual_mxu_readout_bf16"(%io8_t5, %out_acc_t5) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t5 = "atlas.virtual_dma_store_bf16"(%io9_t5, %out_t5, %out_addr_t5, %bf16_size_t5) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t6 = arith.constant -1879031808 : i32
    %k_addr_t6 = arith.constant -1879047168 : i32
    %v_addr_t6 = arith.constant -1879046144 : i32
    %out_addr_t6 = arith.constant -1878970368 : i32
    %fp8_size_t6 = arith.constant 1024 : i32
    %bf16_size_t6 = arith.constant 2048 : i32
    %io1_t6, %q_t6 = "atlas.virtual_dma_load_fp8"(%io10_t5, %q_addr_t6, %fp8_size_t6) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t6, %k_t6 = "atlas.virtual_dma_load_fp8"(%io1_t6, %k_addr_t6, %fp8_size_t6) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t6, %v_t6 = "atlas.virtual_dma_load_fp8"(%io2_t6, %v_addr_t6, %fp8_size_t6) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t6, %k_weight_t6 = "atlas.virtual_mxu_load_weight"(%io3_t6, %k_t6) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t6, %scores_acc_t6 = "atlas.virtual_mxu_reset"(%io4_t6, %q_t6, %k_weight_t6) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t6, %s_t6 = "atlas.virtual_mxu_readout_bf16"(%io5_t6, %scores_acc_t6) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t6 = "atlas.virtual_vpu_reduce"(%s_t6) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t6 = "atlas.virtual_vpu_binary"(%s_t6, %max_t6) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t6 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t6 = "atlas.virtual_vpu_binary"(%shifted_t6, %scale_t6) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t6 = "atlas.virtual_vpu_unary"(%scaled_t6) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t6 = "atlas.virtual_vpu_reduce"(%e_t6) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t6 = "atlas.virtual_vpu_unary"(%sum_t6) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t6 = "atlas.virtual_vpu_binary"(%e_t6, %inverse_t6) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t6 = "atlas.virtual_pack_fp8"(%p_t6) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t6, %v_weight_t6 = "atlas.virtual_mxu_load_weight"(%io6_t6, %v_t6) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t6, %out_acc_t6 = "atlas.virtual_mxu_reset"(%io7_t6, %p8_t6, %v_weight_t6) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t6, %out_t6 = "atlas.virtual_mxu_readout_bf16"(%io8_t6, %out_acc_t6) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t6 = "atlas.virtual_dma_store_bf16"(%io9_t6, %out_t6, %out_addr_t6, %bf16_size_t6) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t7 = arith.constant -1879029760 : i32
    %k_addr_t7 = arith.constant -1879047168 : i32
    %v_addr_t7 = arith.constant -1879046144 : i32
    %out_addr_t7 = arith.constant -1878968320 : i32
    %fp8_size_t7 = arith.constant 1024 : i32
    %bf16_size_t7 = arith.constant 2048 : i32
    %io1_t7, %q_t7 = "atlas.virtual_dma_load_fp8"(%io10_t6, %q_addr_t7, %fp8_size_t7) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t7, %k_t7 = "atlas.virtual_dma_load_fp8"(%io1_t7, %k_addr_t7, %fp8_size_t7) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t7, %v_t7 = "atlas.virtual_dma_load_fp8"(%io2_t7, %v_addr_t7, %fp8_size_t7) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t7, %k_weight_t7 = "atlas.virtual_mxu_load_weight"(%io3_t7, %k_t7) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t7, %scores_acc_t7 = "atlas.virtual_mxu_reset"(%io4_t7, %q_t7, %k_weight_t7) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t7, %s_t7 = "atlas.virtual_mxu_readout_bf16"(%io5_t7, %scores_acc_t7) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t7 = "atlas.virtual_vpu_reduce"(%s_t7) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t7 = "atlas.virtual_vpu_binary"(%s_t7, %max_t7) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t7 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t7 = "atlas.virtual_vpu_binary"(%shifted_t7, %scale_t7) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t7 = "atlas.virtual_vpu_unary"(%scaled_t7) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t7 = "atlas.virtual_vpu_reduce"(%e_t7) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t7 = "atlas.virtual_vpu_unary"(%sum_t7) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t7 = "atlas.virtual_vpu_binary"(%e_t7, %inverse_t7) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t7 = "atlas.virtual_pack_fp8"(%p_t7) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t7, %v_weight_t7 = "atlas.virtual_mxu_load_weight"(%io6_t7, %v_t7) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t7, %out_acc_t7 = "atlas.virtual_mxu_reset"(%io7_t7, %p8_t7, %v_weight_t7) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t7, %out_t7 = "atlas.virtual_mxu_readout_bf16"(%io8_t7, %out_acc_t7) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t7 = "atlas.virtual_dma_store_bf16"(%io9_t7, %out_t7, %out_addr_t7, %bf16_size_t7) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t8 = arith.constant -1879027712 : i32
    %k_addr_t8 = arith.constant -1879047168 : i32
    %v_addr_t8 = arith.constant -1879046144 : i32
    %out_addr_t8 = arith.constant -1878966272 : i32
    %fp8_size_t8 = arith.constant 1024 : i32
    %bf16_size_t8 = arith.constant 2048 : i32
    %io1_t8, %q_t8 = "atlas.virtual_dma_load_fp8"(%io10_t7, %q_addr_t8, %fp8_size_t8) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t8, %k_t8 = "atlas.virtual_dma_load_fp8"(%io1_t8, %k_addr_t8, %fp8_size_t8) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t8, %v_t8 = "atlas.virtual_dma_load_fp8"(%io2_t8, %v_addr_t8, %fp8_size_t8) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t8, %k_weight_t8 = "atlas.virtual_mxu_load_weight"(%io3_t8, %k_t8) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t8, %scores_acc_t8 = "atlas.virtual_mxu_reset"(%io4_t8, %q_t8, %k_weight_t8) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t8, %s_t8 = "atlas.virtual_mxu_readout_bf16"(%io5_t8, %scores_acc_t8) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t8 = "atlas.virtual_vpu_reduce"(%s_t8) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t8 = "atlas.virtual_vpu_binary"(%s_t8, %max_t8) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t8 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t8 = "atlas.virtual_vpu_binary"(%shifted_t8, %scale_t8) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t8 = "atlas.virtual_vpu_unary"(%scaled_t8) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t8 = "atlas.virtual_vpu_reduce"(%e_t8) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t8 = "atlas.virtual_vpu_unary"(%sum_t8) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t8 = "atlas.virtual_vpu_binary"(%e_t8, %inverse_t8) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t8 = "atlas.virtual_pack_fp8"(%p_t8) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t8, %v_weight_t8 = "atlas.virtual_mxu_load_weight"(%io6_t8, %v_t8) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t8, %out_acc_t8 = "atlas.virtual_mxu_reset"(%io7_t8, %p8_t8, %v_weight_t8) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t8, %out_t8 = "atlas.virtual_mxu_readout_bf16"(%io8_t8, %out_acc_t8) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t8 = "atlas.virtual_dma_store_bf16"(%io9_t8, %out_t8, %out_addr_t8, %bf16_size_t8) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t9 = arith.constant -1879025664 : i32
    %k_addr_t9 = arith.constant -1879047168 : i32
    %v_addr_t9 = arith.constant -1879046144 : i32
    %out_addr_t9 = arith.constant -1878964224 : i32
    %fp8_size_t9 = arith.constant 1024 : i32
    %bf16_size_t9 = arith.constant 2048 : i32
    %io1_t9, %q_t9 = "atlas.virtual_dma_load_fp8"(%io10_t8, %q_addr_t9, %fp8_size_t9) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t9, %k_t9 = "atlas.virtual_dma_load_fp8"(%io1_t9, %k_addr_t9, %fp8_size_t9) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t9, %v_t9 = "atlas.virtual_dma_load_fp8"(%io2_t9, %v_addr_t9, %fp8_size_t9) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t9, %k_weight_t9 = "atlas.virtual_mxu_load_weight"(%io3_t9, %k_t9) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t9, %scores_acc_t9 = "atlas.virtual_mxu_reset"(%io4_t9, %q_t9, %k_weight_t9) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t9, %s_t9 = "atlas.virtual_mxu_readout_bf16"(%io5_t9, %scores_acc_t9) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t9 = "atlas.virtual_vpu_reduce"(%s_t9) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t9 = "atlas.virtual_vpu_binary"(%s_t9, %max_t9) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t9 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t9 = "atlas.virtual_vpu_binary"(%shifted_t9, %scale_t9) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t9 = "atlas.virtual_vpu_unary"(%scaled_t9) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t9 = "atlas.virtual_vpu_reduce"(%e_t9) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t9 = "atlas.virtual_vpu_unary"(%sum_t9) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t9 = "atlas.virtual_vpu_binary"(%e_t9, %inverse_t9) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t9 = "atlas.virtual_pack_fp8"(%p_t9) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t9, %v_weight_t9 = "atlas.virtual_mxu_load_weight"(%io6_t9, %v_t9) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t9, %out_acc_t9 = "atlas.virtual_mxu_reset"(%io7_t9, %p8_t9, %v_weight_t9) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t9, %out_t9 = "atlas.virtual_mxu_readout_bf16"(%io8_t9, %out_acc_t9) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t9 = "atlas.virtual_dma_store_bf16"(%io9_t9, %out_t9, %out_addr_t9, %bf16_size_t9) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t10 = arith.constant -1879023616 : i32
    %k_addr_t10 = arith.constant -1879047168 : i32
    %v_addr_t10 = arith.constant -1879046144 : i32
    %out_addr_t10 = arith.constant -1878962176 : i32
    %fp8_size_t10 = arith.constant 1024 : i32
    %bf16_size_t10 = arith.constant 2048 : i32
    %io1_t10, %q_t10 = "atlas.virtual_dma_load_fp8"(%io10_t9, %q_addr_t10, %fp8_size_t10) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t10, %k_t10 = "atlas.virtual_dma_load_fp8"(%io1_t10, %k_addr_t10, %fp8_size_t10) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t10, %v_t10 = "atlas.virtual_dma_load_fp8"(%io2_t10, %v_addr_t10, %fp8_size_t10) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t10, %k_weight_t10 = "atlas.virtual_mxu_load_weight"(%io3_t10, %k_t10) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t10, %scores_acc_t10 = "atlas.virtual_mxu_reset"(%io4_t10, %q_t10, %k_weight_t10) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t10, %s_t10 = "atlas.virtual_mxu_readout_bf16"(%io5_t10, %scores_acc_t10) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t10 = "atlas.virtual_vpu_reduce"(%s_t10) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t10 = "atlas.virtual_vpu_binary"(%s_t10, %max_t10) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t10 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t10 = "atlas.virtual_vpu_binary"(%shifted_t10, %scale_t10) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t10 = "atlas.virtual_vpu_unary"(%scaled_t10) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t10 = "atlas.virtual_vpu_reduce"(%e_t10) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t10 = "atlas.virtual_vpu_unary"(%sum_t10) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t10 = "atlas.virtual_vpu_binary"(%e_t10, %inverse_t10) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t10 = "atlas.virtual_pack_fp8"(%p_t10) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t10, %v_weight_t10 = "atlas.virtual_mxu_load_weight"(%io6_t10, %v_t10) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t10, %out_acc_t10 = "atlas.virtual_mxu_reset"(%io7_t10, %p8_t10, %v_weight_t10) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t10, %out_t10 = "atlas.virtual_mxu_readout_bf16"(%io8_t10, %out_acc_t10) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t10 = "atlas.virtual_dma_store_bf16"(%io9_t10, %out_t10, %out_addr_t10, %bf16_size_t10) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t11 = arith.constant -1879021568 : i32
    %k_addr_t11 = arith.constant -1879047168 : i32
    %v_addr_t11 = arith.constant -1879046144 : i32
    %out_addr_t11 = arith.constant -1878960128 : i32
    %fp8_size_t11 = arith.constant 1024 : i32
    %bf16_size_t11 = arith.constant 2048 : i32
    %io1_t11, %q_t11 = "atlas.virtual_dma_load_fp8"(%io10_t10, %q_addr_t11, %fp8_size_t11) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t11, %k_t11 = "atlas.virtual_dma_load_fp8"(%io1_t11, %k_addr_t11, %fp8_size_t11) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t11, %v_t11 = "atlas.virtual_dma_load_fp8"(%io2_t11, %v_addr_t11, %fp8_size_t11) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t11, %k_weight_t11 = "atlas.virtual_mxu_load_weight"(%io3_t11, %k_t11) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t11, %scores_acc_t11 = "atlas.virtual_mxu_reset"(%io4_t11, %q_t11, %k_weight_t11) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t11, %s_t11 = "atlas.virtual_mxu_readout_bf16"(%io5_t11, %scores_acc_t11) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t11 = "atlas.virtual_vpu_reduce"(%s_t11) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t11 = "atlas.virtual_vpu_binary"(%s_t11, %max_t11) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t11 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t11 = "atlas.virtual_vpu_binary"(%shifted_t11, %scale_t11) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t11 = "atlas.virtual_vpu_unary"(%scaled_t11) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t11 = "atlas.virtual_vpu_reduce"(%e_t11) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t11 = "atlas.virtual_vpu_unary"(%sum_t11) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t11 = "atlas.virtual_vpu_binary"(%e_t11, %inverse_t11) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t11 = "atlas.virtual_pack_fp8"(%p_t11) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t11, %v_weight_t11 = "atlas.virtual_mxu_load_weight"(%io6_t11, %v_t11) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t11, %out_acc_t11 = "atlas.virtual_mxu_reset"(%io7_t11, %p8_t11, %v_weight_t11) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t11, %out_t11 = "atlas.virtual_mxu_readout_bf16"(%io8_t11, %out_acc_t11) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t11 = "atlas.virtual_dma_store_bf16"(%io9_t11, %out_t11, %out_addr_t11, %bf16_size_t11) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t12 = arith.constant -1879019520 : i32
    %k_addr_t12 = arith.constant -1879047168 : i32
    %v_addr_t12 = arith.constant -1879046144 : i32
    %out_addr_t12 = arith.constant -1878958080 : i32
    %fp8_size_t12 = arith.constant 1024 : i32
    %bf16_size_t12 = arith.constant 2048 : i32
    %io1_t12, %q_t12 = "atlas.virtual_dma_load_fp8"(%io10_t11, %q_addr_t12, %fp8_size_t12) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t12, %k_t12 = "atlas.virtual_dma_load_fp8"(%io1_t12, %k_addr_t12, %fp8_size_t12) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t12, %v_t12 = "atlas.virtual_dma_load_fp8"(%io2_t12, %v_addr_t12, %fp8_size_t12) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t12, %k_weight_t12 = "atlas.virtual_mxu_load_weight"(%io3_t12, %k_t12) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t12, %scores_acc_t12 = "atlas.virtual_mxu_reset"(%io4_t12, %q_t12, %k_weight_t12) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t12, %s_t12 = "atlas.virtual_mxu_readout_bf16"(%io5_t12, %scores_acc_t12) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t12 = "atlas.virtual_vpu_reduce"(%s_t12) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t12 = "atlas.virtual_vpu_binary"(%s_t12, %max_t12) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t12 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t12 = "atlas.virtual_vpu_binary"(%shifted_t12, %scale_t12) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t12 = "atlas.virtual_vpu_unary"(%scaled_t12) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t12 = "atlas.virtual_vpu_reduce"(%e_t12) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t12 = "atlas.virtual_vpu_unary"(%sum_t12) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t12 = "atlas.virtual_vpu_binary"(%e_t12, %inverse_t12) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t12 = "atlas.virtual_pack_fp8"(%p_t12) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t12, %v_weight_t12 = "atlas.virtual_mxu_load_weight"(%io6_t12, %v_t12) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t12, %out_acc_t12 = "atlas.virtual_mxu_reset"(%io7_t12, %p8_t12, %v_weight_t12) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t12, %out_t12 = "atlas.virtual_mxu_readout_bf16"(%io8_t12, %out_acc_t12) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t12 = "atlas.virtual_dma_store_bf16"(%io9_t12, %out_t12, %out_addr_t12, %bf16_size_t12) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t13 = arith.constant -1879017472 : i32
    %k_addr_t13 = arith.constant -1879047168 : i32
    %v_addr_t13 = arith.constant -1879046144 : i32
    %out_addr_t13 = arith.constant -1878956032 : i32
    %fp8_size_t13 = arith.constant 1024 : i32
    %bf16_size_t13 = arith.constant 2048 : i32
    %io1_t13, %q_t13 = "atlas.virtual_dma_load_fp8"(%io10_t12, %q_addr_t13, %fp8_size_t13) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t13, %k_t13 = "atlas.virtual_dma_load_fp8"(%io1_t13, %k_addr_t13, %fp8_size_t13) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t13, %v_t13 = "atlas.virtual_dma_load_fp8"(%io2_t13, %v_addr_t13, %fp8_size_t13) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t13, %k_weight_t13 = "atlas.virtual_mxu_load_weight"(%io3_t13, %k_t13) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t13, %scores_acc_t13 = "atlas.virtual_mxu_reset"(%io4_t13, %q_t13, %k_weight_t13) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t13, %s_t13 = "atlas.virtual_mxu_readout_bf16"(%io5_t13, %scores_acc_t13) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t13 = "atlas.virtual_vpu_reduce"(%s_t13) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t13 = "atlas.virtual_vpu_binary"(%s_t13, %max_t13) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t13 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t13 = "atlas.virtual_vpu_binary"(%shifted_t13, %scale_t13) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t13 = "atlas.virtual_vpu_unary"(%scaled_t13) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t13 = "atlas.virtual_vpu_reduce"(%e_t13) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t13 = "atlas.virtual_vpu_unary"(%sum_t13) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t13 = "atlas.virtual_vpu_binary"(%e_t13, %inverse_t13) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t13 = "atlas.virtual_pack_fp8"(%p_t13) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t13, %v_weight_t13 = "atlas.virtual_mxu_load_weight"(%io6_t13, %v_t13) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t13, %out_acc_t13 = "atlas.virtual_mxu_reset"(%io7_t13, %p8_t13, %v_weight_t13) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t13, %out_t13 = "atlas.virtual_mxu_readout_bf16"(%io8_t13, %out_acc_t13) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t13 = "atlas.virtual_dma_store_bf16"(%io9_t13, %out_t13, %out_addr_t13, %bf16_size_t13) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t14 = arith.constant -1879015424 : i32
    %k_addr_t14 = arith.constant -1879047168 : i32
    %v_addr_t14 = arith.constant -1879046144 : i32
    %out_addr_t14 = arith.constant -1878953984 : i32
    %fp8_size_t14 = arith.constant 1024 : i32
    %bf16_size_t14 = arith.constant 2048 : i32
    %io1_t14, %q_t14 = "atlas.virtual_dma_load_fp8"(%io10_t13, %q_addr_t14, %fp8_size_t14) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t14, %k_t14 = "atlas.virtual_dma_load_fp8"(%io1_t14, %k_addr_t14, %fp8_size_t14) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t14, %v_t14 = "atlas.virtual_dma_load_fp8"(%io2_t14, %v_addr_t14, %fp8_size_t14) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t14, %k_weight_t14 = "atlas.virtual_mxu_load_weight"(%io3_t14, %k_t14) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t14, %scores_acc_t14 = "atlas.virtual_mxu_reset"(%io4_t14, %q_t14, %k_weight_t14) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t14, %s_t14 = "atlas.virtual_mxu_readout_bf16"(%io5_t14, %scores_acc_t14) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t14 = "atlas.virtual_vpu_reduce"(%s_t14) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t14 = "atlas.virtual_vpu_binary"(%s_t14, %max_t14) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t14 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t14 = "atlas.virtual_vpu_binary"(%shifted_t14, %scale_t14) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t14 = "atlas.virtual_vpu_unary"(%scaled_t14) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t14 = "atlas.virtual_vpu_reduce"(%e_t14) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t14 = "atlas.virtual_vpu_unary"(%sum_t14) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t14 = "atlas.virtual_vpu_binary"(%e_t14, %inverse_t14) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t14 = "atlas.virtual_pack_fp8"(%p_t14) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t14, %v_weight_t14 = "atlas.virtual_mxu_load_weight"(%io6_t14, %v_t14) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t14, %out_acc_t14 = "atlas.virtual_mxu_reset"(%io7_t14, %p8_t14, %v_weight_t14) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t14, %out_t14 = "atlas.virtual_mxu_readout_bf16"(%io8_t14, %out_acc_t14) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t14 = "atlas.virtual_dma_store_bf16"(%io9_t14, %out_t14, %out_addr_t14, %bf16_size_t14) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    %q_addr_t15 = arith.constant -1879013376 : i32
    %k_addr_t15 = arith.constant -1879047168 : i32
    %v_addr_t15 = arith.constant -1879046144 : i32
    %out_addr_t15 = arith.constant -1878951936 : i32
    %fp8_size_t15 = arith.constant 1024 : i32
    %bf16_size_t15 = arith.constant 2048 : i32
    %io1_t15, %q_t15 = "atlas.virtual_dma_load_fp8"(%io10_t14, %q_addr_t15, %fp8_size_t15) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io2_t15, %k_t15 = "atlas.virtual_dma_load_fp8"(%io1_t15, %k_addr_t15, %fp8_size_t15) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io3_t15, %v_t15 = "atlas.virtual_dma_load_fp8"(%io2_t15, %v_addr_t15, %fp8_size_t15) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %io4_t15, %k_weight_t15 = "atlas.virtual_mxu_load_weight"(%io3_t15, %k_t15) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %io5_t15, %scores_acc_t15 = "atlas.virtual_mxu_reset"(%io4_t15, %q_t15, %k_weight_t15) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %io6_t15, %s_t15 = "atlas.virtual_mxu_readout_bf16"(%io5_t15, %scores_acc_t15) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %max_t15 = "atlas.virtual_vpu_reduce"(%s_t15) {kind = "row_max"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %shifted_t15 = "atlas.virtual_vpu_binary"(%s_t15, %max_t15) {kind = "sub"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %scale_t15 = "atlas.virtual_vpu_splat"() {bits = 16003 : i32} : () -> !atlas.virtual_bf16
    %scaled_t15 = "atlas.virtual_vpu_binary"(%shifted_t15, %scale_t15) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %e_t15 = "atlas.virtual_vpu_unary"(%scaled_t15) {kind = "exp2"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %sum_t15 = "atlas.virtual_vpu_reduce"(%e_t15) {kind = "row_sum"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %inverse_t15 = "atlas.virtual_vpu_unary"(%sum_t15) {kind = "recip"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p_t15 = "atlas.virtual_vpu_binary"(%e_t15, %inverse_t15) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p8_t15 = "atlas.virtual_pack_fp8"(%p_t15) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %io7_t15, %v_weight_t15 = "atlas.virtual_mxu_load_weight"(%io6_t15, %v_t15) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<1>)
    %io8_t15, %out_acc_t15 = "atlas.virtual_mxu_reset"(%io7_t15, %p8_t15, %v_weight_t15) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<1>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %io9_t15, %out_t15 = "atlas.virtual_mxu_readout_bf16"(%io8_t15, %out_acc_t15) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %io10_t15 = "atlas.virtual_dma_store_bf16"(%io9_t15, %out_t15, %out_addr_t15, %bf16_size_t15) : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    return %io10_t15 : !atlas.virtual_state
  }
}
