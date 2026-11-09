"""Ordered K panels retaining FP8 activation tiles across output columns."""
from __future__ import annotations


# The selected target has 32 FP8 registers. Reserve eight for streamed weights
# and lowering temporaries, keeping each bounded activation panel in the rest.
RESIDENT_A_TILES = 32 - 8


def schedule_record(plan):
    panels = (plan.counts[2] + RESIDENT_A_TILES - 1) // RESIDENT_A_TILES
    return dict(operand_reuse="resident-a", resident_a_panel_tiles=RESIDENT_A_TILES,
                resident_a_panels=panels, reduction_unroll_scope="complete_bounded_resident_panel",
                partial_accumulator="raw_bf16_output_buffer_between_ordered_panels")


def virtual_resident_a(plan, addresses, output, epilogue=None, *, seed_address=None,
                       panel_tiles=RESIDENT_A_TILES):
    """Emit typed DMA/MXU operations with exactly the original buffer ABI.

    A preload dominates its N loop, so its FP8 values need no CFG edge
    arguments. The lowerer stages unread loads before reusing their VMEM
    windows, and stages the remainder before the preload block's terminator.
    """
    if type(panel_tiles) is not int or not 1 <= panel_tiles <= RESIDENT_A_TILES:
        raise ValueError("resident activation panel exceeds the selected FP8 register budget")
    mc, nc, kc = plan.counts
    ab, wb, ob = plan.tile_bytes
    a, w = addresses
    state, fp8, bf16 = "!atlas.virtual_state", "!atlas.virtual_fp8", "!atlas.virtual_bf16"
    weight, accum = "!atlas.virtual_mxu_weight<0>", "!atlas.virtual_mxu_acc<0>"
    lines = ["module {",
             f"  func.func @resident_looped_matmul() -> {state} attributes {{atlas.retain_bf16_accumulators, atlas.input_dram_base = {a} : i64, atlas.output_dram_base = {output} : i64, atlas.output_dram_bytes = {plan.buffer_bytes[2]} : i64}} {{",
             f'    %start = "atlas.virtual_start"() : () -> {state}']

    def c(name, value):
        bits = value if value < 1 << 31 else value - (1 << 32)
        lines.append(f"    %{name} = arith.constant {bits} : i32")

    def add(name, base, offset):
        c(name + "_offset", offset)
        lines.append(f"    %{name} = arith.addi %{base}, %{name}_offset : i32")

    def effect(name, before, after, args, types, result=None, attr=""):
        lhs = f"%{after}, %{result[0]}" if result else f"%{after}"
        returns = f"({state}, {result[1]})" if result else state
        lines.append(f'    {lhs} = "atlas.virtual_{name}"(%{before}, '
                     + ", ".join(f"%{v}" for v in args) + f") {attr} : ({state}, "
                     + ", ".join(types) + f") -> {returns}")

    c("abase", a)
    c("ybase", output)
    if seed_address is not None:
        c("seedbase", seed_address)
    seed_type = ", i32" if seed_address is not None else ""
    seed_init = ", %seedbase" if seed_address is not None else ""
    if mc > 1:
        c("zero_m", 0)
        lines.extend([f"    cf.br ^m_header(%start, %zero_m, %abase, %ybase{seed_init} : {state}, i32, i32, i32{seed_type})",
                      f"  ^m_header(%ms: {state}, %m: i32, %arow: i32, %yrow: i32" + (", %seedrow: i32" if seed_address is not None else "") + "):"])
        c("m_count", mc)
        lines.extend(["    %m_more = arith.cmpi ult, %m, %m_count : i32",
                      f"    cf.cond_br %m_more, ^m_body(%ms : {state}), ^done(%ms : {state})",
                      f"  ^m_body(%mb: {state}):"])
    else:
        c("arow", a)
        c("yrow", output)
        if seed_address is not None:
            c("seedrow", seed_address)
        lines.extend([f"    cf.br ^m_body(%start : {state})", f"  ^m_body(%mb: {state}):"])
    panels = list(range(0, kc, panel_tiles))
    lines.append(f"    cf.br ^p0_preload(%mb : {state})")
    for panel, first in enumerate(panels):
        prefix = f"p{panel}"
        count = min(panel_tiles, kc - first)
        final = panel + 1 == len(panels)
        lines.append(f"  ^{prefix}_preload(%{prefix}_ps: {state}):")
        before = prefix + "_ps"
        for index in range(count):
            stem = f"{prefix}_a{index}"
            add(stem + "_addr", "arow", (first + index) * ab)
            c(stem + "_bytes", ab)
            effect("dma_load_fp8", before, stem + "_s", [stem + "_addr", stem + "_bytes"],
                   ["i32", "i32"], (stem, fp8))
            before = stem + "_s"
        c(prefix + "_zero", 0)
        c(prefix + "_wbase", w + first * wb)
        external_seed = seed_address is not None and panel == 0
        n_seed_type = ", i32" if external_seed else ""
        n_seed_init = ", %seedrow" if external_seed else ""
        lines.extend([f"    cf.br ^{prefix}_n_header(%{before}, %{prefix}_zero, %{prefix}_wbase, %yrow{n_seed_init} : {state}, i32, i32, i32{n_seed_type})",
                      f"  ^{prefix}_n_header(%{prefix}_ns: {state}, %{prefix}_n: i32, %{prefix}_wrow: i32, %{prefix}_ytile: i32" + (f", %{prefix}_seedtile: i32" if external_seed else "") + "):"])
        c(prefix + "_n_count", nc)
        lines.append(f"    %{prefix}_n_more = arith.cmpi ult, %{prefix}_n, %{prefix}_n_count : i32")
        next_panel = f"p{panel + 1}_preload" if not final else "m_next"
        lines.extend([f"    cf.cond_br %{prefix}_n_more, ^{prefix}_n_body(%{prefix}_ns : {state}), ^{next_panel}(%{prefix}_ns : {state})",
                      f"  ^{prefix}_n_body(%{prefix}_nb: {state}):"])
        before = prefix + "_nb"
        seed = None
        if external_seed or panel > 0:
            c(prefix + "_seed_bytes", ob)
            seed_pointer = prefix + ("_seedtile" if external_seed else "_ytile")
            effect("dma_load_bf16", before, prefix + "_seed_s", [seed_pointer, prefix + "_seed_bytes"],
                   ["i32", "i32"], (prefix + "_seed", bf16))
            before, seed = prefix + "_seed_s", prefix + "_seed"
        previous = None
        for group in range(0, count, 4):
            for index in range(group, min(group + 4, count)):
                stem = f"{prefix}_k{index}"
                add(stem + "_wp", prefix + "_wrow", index * wb)
                c(stem + "_bytes", wb)
                effect("dma_load_fp8", before, stem + "_w_s", [stem + "_wp", stem + "_bytes"],
                       ["i32", "i32"], (stem + "_w", fp8))
                before = stem + "_w_s"
            for index in range(group, min(group + 4, count)):
                stem = f"{prefix}_k{index}"
                effect("mxu_load_weight", before, stem + "_weight_s", [stem + "_w"], [fp8],
                       (stem + "_weight", weight), "{unit = 0 : i32}")
                before = stem + "_weight_s"
                if index == 0 and seed is not None:
                    effect("mxu_load_acc_bf16", before, prefix + "_seeded_s", [seed], [bf16],
                           (prefix + "_seeded", accum), "{unit = 0 : i32}")
                    before, previous = prefix + "_seeded_s", prefix + "_seeded"
                if previous is None:
                    effect("mxu_reset", before, stem + "_dot_s", [f"{prefix}_a{index}", stem + "_weight"],
                           [fp8, weight], (stem + "_acc", accum))
                else:
                    effect("mxu_accumulate", before, stem + "_dot_s", [f"{prefix}_a{index}", stem + "_weight", previous],
                           [fp8, weight, accum], (stem + "_acc", accum))
                before, previous = stem + "_dot_s", stem + "_acc"
        effect("mxu_readout_bf16", before, prefix + "_result_s", [previous], [accum], (prefix + "_result", bf16))
        result = prefix + "_result"
        if epilogue is not None and final:
            lines.append(f'    %{prefix}_activated = "atlas.virtual_vpu_unary"(%{result}) {{kind = "relu"}} : ({bf16}) -> {bf16}')
            result = prefix + "_activated"
        c(prefix + "_out_bytes", ob)
        effect("dma_store_bf16", prefix + "_result_s", prefix + "_stored", [result, prefix + "_ytile", prefix + "_out_bytes"],
               [bf16, "i32", "i32"])
        add(prefix + "_nn", prefix + "_n", 1)
        add(prefix + "_wn", prefix + "_wrow", kc * wb)
        add(prefix + "_yn", prefix + "_ytile", ob)
        if external_seed:
            add(prefix + "_seedn", prefix + "_seedtile", ob)
        next_seed = f", %{prefix}_seedn" if external_seed else ""
        lines.append(f"    cf.br ^{prefix}_n_header(%{prefix}_stored, %{prefix}_nn, %{prefix}_wn, %{prefix}_yn{next_seed} : {state}, i32, i32, i32{n_seed_type})")
    lines.append(f"  ^m_next(%mn: {state}):")
    if mc > 1:
        add("mn_next", "m", 1)
        add("arow_next", "arow", kc * ab)
        add("yrow_next", "yrow", nc * ob)
        if seed_address is not None:
            add("seedrow_next", "seedrow", nc * ob)
        m_seed = ", %seedrow_next" if seed_address is not None else ""
        lines.append(f"    cf.br ^m_header(%mn, %mn_next, %arow_next, %yrow_next{m_seed} : {state}, i32, i32, i32{seed_type})")
    else:
        lines.append(f"    cf.br ^done(%mn : {state})")
    lines.extend([f"  ^done(%end: {state}):", f"    return %end : {state}", "  }", "}"])
    return "\n".join(lines) + "\n"
