module {
func.func @many_scales() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415984640 : i64} {
%s = "atlas.virtual_start"() : () -> !atlas.virtual_state
%x = "atlas.virtual_vpu_splat"() {bits = 16256 : i32} : () -> !atlas.virtual_bf16
%p0 = "atlas.virtual_pack_fp8"(%x) {scale_code = 122 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l0, %a0 = "atlas.virtual_mxu_load_acc_fp8"(%s, %p0) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r0, %b0 = "atlas.virtual_mxu_readout_bf16"(%l0, %a0) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o0 = "atlas.virtual_output_bf16"(%r0, %b0) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p1 = "atlas.virtual_pack_fp8"(%x) {scale_code = 123 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l1, %a1 = "atlas.virtual_mxu_load_acc_fp8"(%o0, %p1) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r1, %b1 = "atlas.virtual_mxu_readout_bf16"(%l1, %a1) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o1 = "atlas.virtual_output_bf16"(%r1, %b1) {index = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p2 = "atlas.virtual_pack_fp8"(%x) {scale_code = 124 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l2, %a2 = "atlas.virtual_mxu_load_acc_fp8"(%o1, %p2) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r2, %b2 = "atlas.virtual_mxu_readout_bf16"(%l2, %a2) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o2 = "atlas.virtual_output_bf16"(%r2, %b2) {index = 2 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p3 = "atlas.virtual_pack_fp8"(%x) {scale_code = 125 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l3, %a3 = "atlas.virtual_mxu_load_acc_fp8"(%o2, %p3) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r3, %b3 = "atlas.virtual_mxu_readout_bf16"(%l3, %a3) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o3 = "atlas.virtual_output_bf16"(%r3, %b3) {index = 3 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p4 = "atlas.virtual_pack_fp8"(%x) {scale_code = 126 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l4, %a4 = "atlas.virtual_mxu_load_acc_fp8"(%o3, %p4) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r4, %b4 = "atlas.virtual_mxu_readout_bf16"(%l4, %a4) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o4 = "atlas.virtual_output_bf16"(%r4, %b4) {index = 4 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p5 = "atlas.virtual_pack_fp8"(%x) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l5, %a5 = "atlas.virtual_mxu_load_acc_fp8"(%o4, %p5) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r5, %b5 = "atlas.virtual_mxu_readout_bf16"(%l5, %a5) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o5 = "atlas.virtual_output_bf16"(%r5, %b5) {index = 5 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p6 = "atlas.virtual_pack_fp8"(%x) {scale_code = 128 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l6, %a6 = "atlas.virtual_mxu_load_acc_fp8"(%o5, %p6) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r6, %b6 = "atlas.virtual_mxu_readout_bf16"(%l6, %a6) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o6 = "atlas.virtual_output_bf16"(%r6, %b6) {index = 6 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p7 = "atlas.virtual_pack_fp8"(%x) {scale_code = 122 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l7, %a7 = "atlas.virtual_mxu_load_acc_fp8"(%o6, %p7) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r7, %b7 = "atlas.virtual_mxu_readout_bf16"(%l7, %a7) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o7 = "atlas.virtual_output_bf16"(%r7, %b7) {index = 7 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p8 = "atlas.virtual_pack_fp8"(%x) {scale_code = 123 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l8, %a8 = "atlas.virtual_mxu_load_acc_fp8"(%o7, %p8) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r8, %b8 = "atlas.virtual_mxu_readout_bf16"(%l8, %a8) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o8 = "atlas.virtual_output_bf16"(%r8, %b8) {index = 8 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p9 = "atlas.virtual_pack_fp8"(%x) {scale_code = 124 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l9, %a9 = "atlas.virtual_mxu_load_acc_fp8"(%o8, %p9) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r9, %b9 = "atlas.virtual_mxu_readout_bf16"(%l9, %a9) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o9 = "atlas.virtual_output_bf16"(%r9, %b9) {index = 9 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p10 = "atlas.virtual_pack_fp8"(%x) {scale_code = 125 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l10, %a10 = "atlas.virtual_mxu_load_acc_fp8"(%o9, %p10) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r10, %b10 = "atlas.virtual_mxu_readout_bf16"(%l10, %a10) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o10 = "atlas.virtual_output_bf16"(%r10, %b10) {index = 10 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p11 = "atlas.virtual_pack_fp8"(%x) {scale_code = 126 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l11, %a11 = "atlas.virtual_mxu_load_acc_fp8"(%o10, %p11) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r11, %b11 = "atlas.virtual_mxu_readout_bf16"(%l11, %a11) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o11 = "atlas.virtual_output_bf16"(%r11, %b11) {index = 11 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p12 = "atlas.virtual_pack_fp8"(%x) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l12, %a12 = "atlas.virtual_mxu_load_acc_fp8"(%o11, %p12) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r12, %b12 = "atlas.virtual_mxu_readout_bf16"(%l12, %a12) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o12 = "atlas.virtual_output_bf16"(%r12, %b12) {index = 12 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
%p13 = "atlas.virtual_pack_fp8"(%x) {scale_code = 128 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
%l13, %a13 = "atlas.virtual_mxu_load_acc_fp8"(%o12, %p13) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
%r13, %b13 = "atlas.virtual_mxu_readout_bf16"(%l13, %a13) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
%o13 = "atlas.virtual_output_bf16"(%r13, %b13) {index = 13 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
return %o13 : !atlas.virtual_state
}
}
