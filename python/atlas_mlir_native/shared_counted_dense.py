"""Counted Atlas K traversal of one checked shared output-tile schedule.

Revalidate and consume the actual shared calls through the existing expander.
Only the complete ascending one-tile trace below has a counted realization.
The loop carries exact BF16 readout/restore state across CFG edges; no independent partial dot products or host summation are introduced.
"""

from .shared_dense import INPUT_BASE, SLOT_BYTES, expand_dense, virtual_program
from .virtual_builder import VirtualBuilder


def counted_dense_program(selected, scheduled, *, output_base):
    if selected.m != 32 or selected.n != 32 or selected.k % 32:
        raise ValueError("counted lowering requires one complete output tile")
    depth = selected.k // 32
    if depth < 1 or depth > 256:
        raise ValueError("counted reduction outside explicit checking budget")
    trace = []
    builder = VirtualBuilder(INPUT_BASE, output_base)

    def load(role, row, col):
        trace.append((role, row, col))
        index = col if role == "lhs" else depth + row
        return builder.emit(
            "virtual_input_fp8",
            [],
            [],
            "!atlas.virtual_fp8",
            f" {{index = {index} : i32}}",
        )

    def store(row, col, value):
        trace.append(("store", row, col))

    expand_dense(selected, scheduled, builder, load, store)
    expected = [v for k in range(depth) for v in (("lhs", 0, k), ("rhs", k, 0))] + [
        ("store", 0, 0)
    ]
    if trace != expected:
        raise ValueError("actual shared schedule differs from counted K traversal")
    if depth == 1:
        return virtual_program(selected, scheduled, output_base=output_base)
    if output_base < INPUT_BASE + 2 * depth * SLOT_BYTES:
        raise ValueError("counted inputs overlap output")
    return f"""module {{
  func.func @counted_dense() -> !atlas.virtual_state attributes {{atlas.input_dram_base = {INPUT_BASE} : i64, atlas.output_dram_base = {output_base} : i64}} {{
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %size = arith.constant 1024 : i32
    %step = arith.constant {SLOT_BYTES} : i32
    %one = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    %limit = arith.constant {depth - 1} : i32
    %a0 = arith.constant {INPUT_BASE} : i32
    %b0 = arith.constant {INPUT_BASE + depth * SLOT_BYTES} : i32
    %s1, %x0 = "atlas.virtual_dma_load_fp8"(%s, %a0, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s2, %y0 = "atlas.virtual_dma_load_fp8"(%s1, %b0, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s3, %w0 = "atlas.virtual_mxu_load_weight"(%s2, %y0) {{unit = 0 : i32}} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %s4, %acc0 = "atlas.virtual_mxu_reset"(%s3, %x0, %w0) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %sinit, %initial = "atlas.virtual_mxu_readout_bf16"(%s4, %acc0) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %a1 = arith.addi %a0, %step : i32
    %b1 = arith.addi %b0, %step : i32
    cf.br ^loop(%sinit, %initial, %a1, %b1, %zero : !atlas.virtual_state, !atlas.virtual_bf16, i32, i32, i32)
  ^loop(%ls: !atlas.virtual_state, %acc: !atlas.virtual_bf16, %a: i32, %b: i32, %i: i32):
    %more = arith.cmpi ult, %i, %limit : i32
    cf.cond_br %more, ^body(%ls : !atlas.virtual_state), ^exit(%ls, %acc : !atlas.virtual_state, !atlas.virtual_bf16)
  ^body(%bs: !atlas.virtual_state):
    %s5, %x = "atlas.virtual_dma_load_fp8"(%bs, %a, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s6, %y = "atlas.virtual_dma_load_fp8"(%s5, %b, %size) : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s7, %w = "atlas.virtual_mxu_load_weight"(%s6, %y) {{unit = 0 : i32}} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %seedstate, %ba = "atlas.virtual_mxu_load_acc_bf16"(%s7, %acc) {{unit = 0 : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s8, %next = "atlas.virtual_mxu_accumulate"(%seedstate, %x, %w, %ba) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %readstate, %tile = "atlas.virtual_mxu_readout_bf16"(%s8, %next) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %an = arith.addi %a, %step : i32
    %bn = arith.addi %b, %step : i32
    %jn = arith.addi %i, %one : i32
    cf.br ^loop(%readstate, %tile, %an, %bn, %jn : !atlas.virtual_state, !atlas.virtual_bf16, i32, i32, i32)
  ^exit(%es: !atlas.virtual_state, %out: !atlas.virtual_bf16):
    %s10 = "atlas.virtual_output_bf16"(%es, %out) {{index = 0 : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s10 : !atlas.virtual_state
  }}
}}
"""
