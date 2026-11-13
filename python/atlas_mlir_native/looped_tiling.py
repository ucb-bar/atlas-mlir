"""Lower ordered matrix contractions to compact Atlas M/N/K loops."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from merlin.semantic_compiler.model import (
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.ordered_k_tiling import (
    OrderedKTilingError,
    _validated_oriented_source,
    partition_ordered_matmul_k,
)
from merlin.semantic_compiler.ordered_mnk_tiling import LoopedMNKTiling
from merlin.semantic_compiler.target_binding import NativeCompilationError


def is_looped_request(request: KernelRequest) -> bool:
    return (
        len(request.nodes) in (4, 5)
        and tuple(node.op for node in request.nodes[:4]) == ("input", "input", "input", "matmul_accumulate")
        and (len(request.nodes) == 4 or request.nodes[-1].op == "relu")
    ) or (
        len(request.nodes) in (3, 4)
        and tuple(node.op for node in request.nodes[:3]) == ("input", "input", "matmul")
        and (len(request.nodes) == 3 or request.nodes[-1].op == "relu")
        and any(node.type.shape != (32, 32) for node in request.nodes)
    )


def split_seeded_request(request):
    """Separate an explicit completed BF16 accumulator from loop geometry."""
    if len(request.nodes) not in (4, 5) or request.nodes[3].op != "matmul_accumulate":
        return request, None
    from .binding import MXU0_POLICY, MXU0_SEED_ATTRS, _mxu0_accumulate_maps, _mxu0_maps

    a, w, seed, result = request.nodes[:4]
    if (
        seed.op != "input"
        or seed.effect != "input"
        or seed.attrs != MXU0_SEED_ATTRS
        or seed.index_maps
        or seed.type != result.type
        or seed.type.dtype != "bf16"
        or seed.type.numerical_policy != MXU0_POLICY
        or result.inputs != (a.id, w.id, seed.id)
        or result.attrs
        or result.effect != "pure"
        or result.index_maps != _mxu0_accumulate_maps()
        or dict(request.input_storages).get(seed.id) != "external"
    ):
        raise NativeCompilationError(
            "unsupported_semantics", "loop accumulator must be an explicit completed BF16 state"
        )
    reset = replace(result, op="matmul", inputs=(a.id, w.id), index_maps=_mxu0_maps())
    geometry = replace(
        request,
        nodes=(a, w, reset, *request.nodes[4:]),
        input_storages=tuple(item for item in request.input_storages if item[0] != seed.id),
    )
    return geometry, seed


def split_epilogue(request):
    """Keep a terminal pointwise operation outside every ordered K step."""
    if len(request.nodes) != 4:
        return request, None
    from .binding import BF16_RELU_ATTRS

    result, epilogue = request.nodes[-2:]
    if (
        epilogue.op != "relu"
        or epilogue.effect != "pure"
        or dict(epilogue.attrs) != dict(BF16_RELU_ATTRS)
        or epilogue.inputs != (result.id,)
        or epilogue.type != TensorType(result.type.shape, "bf16", "raw-bit-pattern")
        or epilogue.index_maps != pointwise_identity_maps(2, 1)
        or request.outputs != (epilogue.id,)
    ):
        raise NativeCompilationError("unsupported_semantics", "looped contraction has an unsupported epilogue")
    return replace(request, nodes=request.nodes[:-1], outputs=(result.id,)), epilogue


def prepare_plan(request: KernelRequest) -> LoopedMNKTiling:
    from .binding import (
        MXU0_INPUT_DOMAIN,
        MXU0_POLICY,
        TARGET_IDENTITY,
        AtlasNativeBinding,
    )

    request, _ = split_seeded_request(request)
    request, _ = split_epilogue(request)
    if (
        len(request.nodes) != 3
        or request.target_identity != TARGET_IDENTITY
        or request.lowering_policy != "diagnostic"
        or any(
            node.type.dtype != "fp8_e4m3"
            or node.type.numerical_policy != "raw-bit-pattern"
            or node.attrs != (("input_domain", MXU0_INPUT_DOMAIN),)
            for node in request.nodes[:2]
        )
        or request.nodes[-1].type.dtype != "bf16"
    ):
        raise NativeCompilationError(
            "unsupported_semantics", "looped contraction needs the admitted MXU0 source policy"
        )
    try:
        _validated_oriented_source(request, MXU0_POLICY)
    except OrderedKTilingError as error:
        raise NativeCompilationError("unsupported_semantics", str(error)) from error
    descriptor = next(d for d in AtlasNativeBinding.profile().descriptors if d.name == "atlas.mxu0_reset_acc_tile")
    tm, tn = (axis.minimum for axis in descriptor.output_axis_bounds)
    tk = descriptor.input_axis_bounds[0][1].minimum
    m, n, k = request.nodes[0].type.shape[0], request.nodes[1].type.shape[0], request.nodes[0].type.shape[1]
    return LoopedMNKTiling(
        request, tm, tn, tk, MXU0_POLICY, 1, 2, b"\0" if m % tm or n % tn else None, b"\0" if k % tk else None
    )


def prepare_execution_plan(request):
    """Keep the selected DRAM budget and stream independent output panels."""
    from merlin.semantic_compiler.streamed_mnk_tiling import (
        plan_streamed_k_reduction,
        plan_streamed_mnk,
    )

    from .binding import TILE_BYTES, dram_slots

    full = prepare_plan(request)
    _, seed = split_seeded_request(request)
    if seed is not None:
        sizes = (*full.buffer_bytes, full.buffer_bytes[2])
        if sum((size + TILE_BYTES - 1) // TILE_BYTES * TILE_BYTES for size in sizes) > dram_slots() * TILE_BYTES:
            raise NativeCompilationError("resource_limit", "seeded loop buffers exceed the selected DRAM window")
        return full, full, None
    try:
        a, w, out = full.tile_bytes
        minimum = (full.counts[2] * a, full.counts[2] * w, out)
        needs_k = sum((size + TILE_BYTES - 1) // TILE_BYTES * TILE_BYTES for size in minimum) > dram_slots() * TILE_BYTES
        planner = plan_streamed_k_reduction if needs_k else plan_streamed_mnk
        streaming = planner(
            full,
            capacity_bytes=dram_slots() * TILE_BYTES,
            alignment_bytes=TILE_BYTES,
            pad_element=b"\0",
        )
    except ValueError as error:
        raise NativeCompilationError("resource_limit", str(error)) from error
    return full, streaming.body if streaming else full, streaming


def _virtual(
    plan: LoopedMNKTiling,
    addresses: tuple[int, int],
    output: int,
    epilogue=None,
    *,
    seed_address=None,
    k_unroll=1,
    unit=0,
    batch_count=1,
    shared_weight=True,
    exponent_pages=(),
    restoration=None,
    output_window_bytes=None,
    cache_a=False,
    cache_a_tiles=None,
    wide_cache_a=False,
) -> str:
    """Emit only typed operations; scheduling and physical placement stay downstream."""
    if type(batch_count) is not int or batch_count < 1 or type(shared_weight) is not bool:
        raise ValueError("batch count must be positive and weight sharing must be boolean")
    if batch_count > 1 and seed_address is not None:
        raise ValueError("batched external accumulator seeds require a separate ABI")
    if unit not in (0, 1):
        raise ValueError("MXU unit must be 0 or 1")
    if type(k_unroll) is not int or not 1 <= k_unroll <= 64:
        raise ValueError("K unroll factor must be an integer in [1,64]")
    if type(cache_a) is not bool:
        raise ValueError("activation residency must be boolean")
    if type(wide_cache_a) is not bool or (wide_cache_a and not cache_a):
        raise ValueError("wide activation residency requires an explicit active cache")
    mc, nc, kc = plan.counts
    cached_tiles = kc if cache_a_tiles is None else cache_a_tiles
    if cache_a_tiles is not None and (not cache_a or type(cache_a_tiles) is not int):
        raise ValueError("a cached prefix requires activation residency and an integer tile count")
    from .quantized_operand_reuse import WIDE_CACHE_TILES
    cache_limit = WIDE_CACHE_TILES if wide_cache_a else 64
    if cache_a and (unit != 1 or not 1 <= cached_tiles <= min(kc, cache_limit)):
        raise ValueError("quantized activation residency requires MXU1 and a bounded tile count")
    ab, wb, ob = plan.tile_bytes
    a, w = addresses
    state, fp8, bf16 = "!atlas.virtual_state", "!atlas.virtual_fp8", "!atlas.virtual_bf16"
    retention = "atlas.retain_bf16_accumulators, " if k_unroll > 1 or cache_a else ""
    output_window_bytes = plan.buffer_bytes[2] if output_window_bytes is None else output_window_bytes
    lines = [
        "module {",
        f"  func.func @looped_matmul() -> {state} attributes {{{retention}atlas.input_dram_base = {a} : i64, atlas.output_dram_base = {output} : i64, atlas.output_dram_bytes = {output_window_bytes} : i64}} {{",
        f'    %start = "atlas.virtual_start"() : () -> {state}',
    ]

    def c(name, value):
        bits = value if value < 1 << 31 else value - (1 << 32)
        lines.append(f"    %{name} = arith.constant {bits} : i32")

    def effect(name, before, after, args, types, result=None, attr=""):
        lhs = f"%{after}, %{result[0]}" if result else f"%{after}"
        returns = f"({state}, {result[1]})" if result else state
        lines.append(
            f'    {lhs} = "atlas.virtual_{name}"(%{before}, '
            + ", ".join(f"%{v}" for v in args)
            + f") {attr} : ({state}, "
            + ", ".join(types)
            + f") -> {returns}"
        )

    slot = "!atlas.virtual_spill"
    if cache_a:
        for index in range(cached_tiles):
            lines.append(f'    %cached_a{index} = "atlas.virtual_slot"() '
                         f'{{kind = "fp8", words = {ab // 4} : i32, slot_word = -1 : i32, purpose = "scratch"}} : () -> {slot}')

    def tile_operands(prefix, before, ap, wp, cached=None):
        c(prefix + "_bytes", ab)
        if cached is None:
            effect("dma_load_fp8", before, prefix + "_a_s", [ap, prefix + "_bytes"], ["i32", "i32"], (prefix + "_a", fp8))
        else:
            effect("restore", before, prefix + "_a_s", [f"cached_a{cached}"], [slot], (prefix + "_a", fp8))
        effect(
            "dma_load_fp8",
            prefix + "_a_s",
            prefix + "_w_s",
            [wp, prefix + "_bytes"],
            ["i32", "i32"],
            (prefix + "_w", fp8),
        )
        return prefix + "_w_s"

    def tile_product(prefix, before, ap, wp, acc=None, *, loaded=False, cached=None):
        if not loaded:
            before = tile_operands(prefix, before, ap, wp, cached)
        weight, accum = f"!atlas.virtual_mxu_weight<{unit}>", f"!atlas.virtual_mxu_acc<{unit}>"
        effect(
            "mxu_load_weight",
            before,
            prefix + "_weight_s",
            [prefix + "_w"],
            [fp8],
            (prefix + "_weight", weight),
            f"{{unit = {unit} : i32}}",
        )
        if acc is None:
            effect(
                "mxu_reset",
                prefix + "_weight_s",
                prefix + "_dot_s",
                [prefix + "_a", prefix + "_weight"],
                [fp8, weight],
                (prefix + "_acc", accum),
            )
        else:
            effect(
                "mxu_load_acc_bf16",
                prefix + "_weight_s",
                prefix + "_seed_s",
                [acc],
                [bf16],
                (prefix + "_seed", accum),
                f"{{unit = {unit} : i32}}",
            )
            effect(
                "mxu_accumulate",
                prefix + "_seed_s",
                prefix + "_dot_s",
                [prefix + "_a", prefix + "_weight", prefix + "_seed"],
                [fp8, weight, accum],
                (prefix + "_acc", accum),
            )
        effect(
            "mxu_readout_bf16",
            prefix + "_dot_s",
            prefix + "_result_s",
            [prefix + "_acc"],
            [accum],
            (prefix + "_result", bf16),
        )

    # The loop owner carries all affine page addresses beside its operand
    # addresses. Hooks emit typed readout operations without rewriting this CFG.
    if exponent_pages and len(exponent_pages) != 2:
        raise ValueError("exponent pages require left and right descriptors")
    batch_pages = [i for i, page in enumerate(exponent_pages) if batch_count > 1 and page[2]]
    def page_name(i, suffix):
        return "restore_" + ("l" if i == 0 else "r") + suffix
    def values(names):
        return "".join(", %" + name for name in names)
    def parameters(names):
        return "".join(", %" + name + ": i32" for name in names)
    batch_w_type = "" if shared_weight else ", i32"
    batch_w_initial = "" if shared_weight else ", %w_initial"
    batch_w_parameter = "" if shared_weight else ", %batch_w: i32"
    batch_w_next = "" if shared_weight else ", %batch_wn"
    batch_types = ", i32" * len(batch_pages)
    row_types = ", i32" if exponent_pages and exponent_pages[0][1] else ""
    col_types = ", i32" if exponent_pages and exponent_pages[1][1] else ""
    entry_state = "start"
    if restoration is not None:
        lines.extend(restoration.entry())
        entry_state = "restore_start"
    for i, (base, tile_stride, batch_stride) in enumerate(exponent_pages):
        c(page_name(i, "base"), base)
    c("zero_m", 0)
    c("a_initial", a)
    c("w_initial", w)
    c("y_initial", output)
    if batch_count > 1:
        lines += [
            f"    cf.br ^batch_header(%{entry_state}, %a_initial{batch_w_initial}, %y_initial{values(page_name(i, "base") for i in batch_pages)} : {state}, i32{batch_w_type}, i32{batch_types})",
            f"  ^batch_header(%bs: {state}, %batch_a: i32{batch_w_parameter}, %batch_y: i32{parameters(page_name(i, "batch") for i in batch_pages)}):",
        ]
        c("batch_end", a + batch_count * mc * kc * ab)
        lines += [
            "    %batch_more = arith.cmpi ult, %batch_a, %batch_end : i32",
            f"    cf.cond_br %batch_more, ^batch_body(%bs : {state}), ^all_done(%bs : {state})",
            f"  ^batch_body(%batch_start: {state}):",
        ]
    else:
        lines += [f"    cf.br ^batch_body(%{entry_state} : {state})", f"  ^batch_body(%batch_start: {state}):"]
    a_origin = "batch_a" if batch_count > 1 else "a_initial"
    w_origin = "batch_w" if batch_count > 1 and not shared_weight else "w_initial"
    y_origin = "batch_y" if batch_count > 1 else "y_initial"
    a_row = "arow" if mc > 1 else a_origin
    y_row = "yrow" if mc > 1 else y_origin
    w_row = "wrow" if nc > 1 else w_origin
    y_tile = "ytile" if nc > 1 else y_row
    page_origins = [page_name(i, "batch" if i in batch_pages else "base") for i in range(len(exponent_pages))]
    batch_ready = 'batch_start'
    if restoration is not None:
        prefetch, batch_ready = restoration.batch_entry(batch_ready, page_origins)
        lines.extend(prefetch)
    page_addresses = [page_name(i, "page") if page[1] and (mc if i == 0 else nc) > 1 else page_origins[i]
                      for i, page in enumerate(exponent_pages)]
    if seed_address is not None:
        c("seed_base", seed_address)
    seed_type = ", i32" if seed_address is not None else ""
    seed_entry = ", %seed_base" if seed_address is not None else ""
    seed_m_parameter = ", %seed_row: i32" if seed_address is not None else ""
    seed_m_value = ", %seed_row" if seed_address is not None else ""
    seed_n_parameter = ", %seed_tile: i32" if seed_address is not None else ""
    if mc > 1:
        lines += [
            f"    cf.br ^m_header(%{batch_ready}, %zero_m, %{a_origin}, %{y_origin}{seed_entry}{values([page_origins[0]]) if row_types else ""} : {state}, i32, i32, i32{seed_type}{row_types})",
            f"  ^m_header(%ms: {state}, %m: i32, %arow: i32, %yrow: i32{seed_m_parameter}{parameters([page_name(0, "page")]) if row_types else ""}):",
        ]
        c("m_count", mc)
        lines += [
            "    %m_more = arith.cmpi ult, %m, %m_count : i32",
            f"    cf.cond_br %m_more, ^m_body(%ms : {state}), ^done(%ms : {state})",
            f"  ^m_body(%mb: {state}):",
        ]
    else:
        if seed_address is not None:
            c("seed_row", seed_address)
        lines += [f"    cf.br ^m_body(%{batch_ready} : {state})", f"  ^m_body(%mb: {state}):"]
    row_ready = "mb"
    if cache_a:
        # A row is immutable across N. Refresh every slot for each M row
        # and batch before entering the column loop; no cross-launch state.
        for index in range(cached_tiles):
            prefix = f"cache_a{index}"
            c(prefix + "_offset", index * ab)
            c(prefix + "_bytes", ab)
            lines.append(f"    %{prefix}_address = arith.addi %{a_row}, %{prefix}_offset : i32")
            effect("dma_load_fp8", row_ready, prefix + "_loaded",
                   [prefix + "_address", prefix + "_bytes"], ["i32", "i32"], (prefix + "_value", fp8))
            effect("save", prefix + "_loaded", prefix + "_saved",
                   [f"cached_a{index}", prefix + "_value"], [slot, fp8])
            row_ready = prefix + "_saved"
    c("zero_n", 0)
    if nc > 1:
        lines += [
            f"    cf.br ^n_header(%{row_ready}, %zero_n, %{w_origin}, %{y_row}{seed_m_value}{values([page_origins[1]]) if col_types else ""} : {state}, i32, i32, i32{seed_type}{col_types})",
            f"  ^n_header(%ns: {state}, %n: i32, %wrow: i32, %ytile: i32{seed_n_parameter}{parameters([page_name(1, "page")]) if col_types else ""}):",
        ]
        c("n_count", nc)
        lines += [
            "    %n_more = arith.cmpi ult, %n, %n_count : i32",
            f"    cf.cond_br %n_more, ^n_body(%ns : {state}), ^m_next(%ns : {state})",
            f"  ^n_body(%nb: {state}):",
        ]
    else:
        # A one-tile column interval uses the same output base.
        if seed_address is not None:
            lines.append("    %seed_tile = arith.addi %seed_row, %zero_n : i32")
        lines += [f"    cf.br ^n_body(%{row_ready} : {state})", f"  ^n_body(%nb: {state}):"]
    if seed_address is None:
        tile_product("first", "nb", a_row, w_row, cached=0 if cache_a else None)
    else:
        c("seed_bytes", ob)
        effect("dma_load_bf16", "nb", "seed_loaded", ["seed_tile", "seed_bytes"], ["i32", "i32"], ("seed_value", bf16))
        tile_product("first", "seed_loaded", a_row, w_row, "seed_value", cached=0 if cache_a else None)
    def ordered_steps(prefix, count, before, ap, wp, value, cached_start=None):
        # Stage at most four tile pairs before consuming them. Eight
        # transfers fit the generic DMA staging windows. Their first
        # uses still govern waits, so later operand transfers can run
        # while earlier ordered BF16 contractions execute.
        prior_right = wp
        for first in range(0, count, 4):
            steps = []
            prior_left = ap
            if cached_start is None:
                prior_right = wp
            for index in range(first, min(first + 4, count)):
                step = f"{prefix}_{index}"
                left, right = ap, wp
                if cached_start is not None and index:
                    c(step + "_w_offset", wb)
                    lines.append(f"    %{step}_wp = arith.addi %{prior_right}, %{step}_w_offset : i32")
                    right = step + "_wp"
                elif index:
                    # Consecutive operands in one DMA staging group use
                    # one tile-sized increment. This keeps the constants
                    # in RV32's immediate range and shortens pointer lives.
                    increment = index != first
                    c(step + "_a_offset", ab if increment else index * ab)
                    c(step + "_w_offset", wb if increment else index * wb)
                    lines.extend(
                        (
                            f"    %{step}_ap = arith.addi %{prior_left if increment else ap}, %{step}_a_offset : i32",
                            f"    %{step}_wp = arith.addi %{prior_right if increment else wp}, %{step}_w_offset : i32",
                        )
                    )
                    left, right = step + "_ap", step + "_wp"
                before = tile_operands(step, before, left, right,
                                       None if cached_start is None else cached_start + index)
                prior_left, prior_right = left, right
                steps.append(step)
            for step in steps:
                tile_product(step, before, None, None, value, loaded=True)
                before, value = step + "_result_s", step + "_result"
        return before, value

    # Consume the resident prefix first, then stream the remaining K tiles
    # through the existing ordered loop. The BF16 value crosses that boundary
    # exactly as it does between ordinary counted groups.
    entry_state, entry_value, k_start = "first_result_s", "first_result", 1
    if cache_a:
        c("cached_w_step", wb)
        lines.append(f"    %cached_wfirst = arith.addi %{w_row}, %cached_w_step : i32")
        current, value = ordered_steps("cached_step", cached_tiles - 1, "first_result_s", None,
                                       "cached_wfirst", "first_result", cached_start=1)
        entry_state, entry_value, k_start = current, value, cached_tiles
    if kc == k_start:
        lines.append(f"    cf.br ^store(%{entry_state}, %{entry_value} : {state}, {bf16})")
    elif k_unroll > 1:
        # Replicate complete, ordered tile steps. The reset remains first, and
        # each step consumes the previous completed BF16 value. This changes
        # loop control only; it does not reassociate the reduction or extend
        # the live MXU state across iterations.
        factor = min(k_unroll, kc - k_start)
        remainder = (kc - k_start) % factor
        c("zero_k", 0)
        c("a_step", k_start * ab)
        c("w_step", k_start * wb)
        lines += [
            f"    %afirst = arith.addi %{a_row}, %a_step : i32",
            f"    %wfirst = arith.addi %{w_row}, %w_step : i32",
            f"    cf.br ^ku_header(%{entry_state}, %zero_k, %afirst, %wfirst, %{entry_value} : {state}, i32, i32, i32, {bf16})",
            f"  ^ku_header(%kus: {state}, %ku: i32, %ku_ap: i32, %ku_wp: i32, %ku_value: {bf16}):",
        ]
        c("ku_count", (kc - k_start) // factor)
        lines += [
            "    %ku_more = arith.cmpi ult, %ku, %ku_count : i32",
            f"    cf.cond_br %ku_more, ^ku_body(%kus : {state}), ^ku_tail(%kus, %ku_value : {state}, {bf16})",
            f"  ^ku_body(%kub: {state}):",
        ]

        current, value = ordered_steps("ku_step", factor, "kub", "ku_ap", "ku_wp", "ku_value")
        c("ku_stride", 1)
        c("ku_a_stride", factor * ab)
        c("ku_w_stride", factor * wb)
        lines += [
            "    %ku_next = arith.addi %ku, %ku_stride : i32",
            "    %ku_an = arith.addi %ku_ap, %ku_a_stride : i32",
            "    %ku_wn = arith.addi %ku_wp, %ku_w_stride : i32",
            f"    cf.br ^ku_header(%{current}, %ku_next, %ku_an, %ku_wn, %{value} : {state}, i32, i32, i32, {bf16})",
            f"  ^ku_tail(%kut: {state}, %kt_value: {bf16}):",
        ]
        if remainder:
            completed_tiles = kc - remainder
            c("kt_a_offset", completed_tiles * ab)
            c("kt_w_offset", completed_tiles * wb)
            lines.extend(
                (
                    f"    %kt_ap = arith.addi %{a_row}, %kt_a_offset : i32",
                    f"    %kt_wp = arith.addi %{w_row}, %kt_w_offset : i32",
                )
            )
        current, value = ordered_steps("kt_step", remainder, "kut", "kt_ap", "kt_wp", "kt_value")
        lines.append(f"    cf.br ^store(%{current}, %{value} : {state}, {bf16})")
    elif kc > 1:
        c("zero_k", 0)
        c("a_step", k_start * ab)
        c("w_step", k_start * wb)
        lines += [
            f"    %afirst = arith.addi %{a_row}, %a_step : i32",
            f"    %wfirst = arith.addi %{w_row}, %w_step : i32",
            f"    cf.br ^k_header(%{entry_state}, %zero_k, %afirst, %wfirst, %{entry_value} : {state}, i32, i32, i32, {bf16})",
            f"  ^k_header(%ks: {state}, %k: i32, %ap: i32, %wp: i32, %carried: {bf16}):",
        ]
        c("k_count", kc - k_start)
        lines += [
            "    %k_more = arith.cmpi ult, %k, %k_count : i32",
            f"    cf.cond_br %k_more, ^k_body(%ks : {state}), ^store(%ks, %carried : {state}, {bf16})",
            f"  ^k_body(%kb: {state}):",
        ]
        tile_product("step", "kb", "ap", "wp", "carried")
        c("one_k", 1)
        lines += [
            "    %kn = arith.addi %k, %one_k : i32",
            "    %an = arith.addi %ap, %step_bytes : i32",
            "    %wn = arith.addi %wp, %step_bytes : i32",
            f"    cf.br ^k_header(%step_result_s, %kn, %an, %wn, %step_result : {state}, i32, i32, i32, {bf16})",
        ]
    else:
        lines.append(f"    cf.br ^store(%first_result_s, %first_result : {state}, {bf16})")
    lines.append(f"  ^store(%ss: {state}, %result: {bf16}):")
    output_value = "result"
    if epilogue is not None:
        lines.append(f'    %activated = "atlas.virtual_vpu_unary"(%result) {{kind = "relu"}} : ({bf16}) -> {bf16}')
        output_value = "activated"
    c("out_bytes", ob)
    if restoration is None:
        effect("dma_store_bf16", "ss", "stored", [output_value, y_tile, "out_bytes"], [bf16, "i32", "i32"])
    else:
        lines.extend(restoration.tile(output_address=y_tile, exponent_addresses=page_addresses))
    if nc > 1:
        c("one_n", 1)
        c("w_row_step", kc * wb)
    if seed_address is not None and nc > 1:
        lines.append("    %seed_next = arith.addi %seed_tile, %out_bytes : i32")
    if nc > 1 and col_types:
        c("restore_rstride", exponent_pages[1][1])
        lines.append("    %restore_rpage_next = arith.addi %restore_rpage, %restore_rstride : i32")
    seed_n_next = ", %seed_next" if seed_address is not None else ""
    n_next = [
        "    %nn = arith.addi %n, %one_n : i32",
        f"    %wnext = arith.addi %{w_row}, %w_row_step : i32",
        f"    %ynext = arith.addi %{y_tile}, %out_bytes : i32",
        f"    cf.br ^n_header(%stored, %nn, %wnext, %ynext{seed_n_next}{values(["restore_rpage_next"]) if col_types else ""} : {state}, i32, i32, i32{seed_type}{col_types})",
    ]
    if nc == 1:
        n_next = [f"    cf.br ^m_next(%stored : {state})"]
    lines += [*n_next, f"  ^m_next(%mn: {state}):"]
    if mc > 1:
        c("one_m", 1)
        c("a_row_step", kc * ab)
        c("y_row_step", nc * ob)
    if seed_address is not None and mc > 1:
        lines.append("    %seed_row_next = arith.addi %seed_row, %y_row_step : i32")
    if mc > 1 and row_types:
        c("restore_lstride", exponent_pages[0][1])
        lines.append("    %restore_lpage_next = arith.addi %restore_lpage, %restore_lstride : i32")
    seed_m_next = ", %seed_row_next" if seed_address is not None else ""
    m_next = [
        "    %mm = arith.addi %m, %one_m : i32",
        f"    %arow_next = arith.addi %{a_row}, %a_row_step : i32",
        f"    %yrow_next = arith.addi %{y_row}, %y_row_step : i32",
        f"    cf.br ^m_header(%mn, %mm, %arow_next, %yrow_next{seed_m_next}{values(["restore_lpage_next"]) if row_types else ""} : {state}, i32, i32, i32{seed_type}{row_types})",
    ]
    if mc == 1:
        m_next = [f"    cf.br ^done(%mn : {state})"]
    lines += [*m_next, f"  ^done(%end: {state}):"]
    if batch_count > 1:
        c("batch_a_stride", mc * kc * ab)
        c("batch_y_stride", mc * nc * ob)
        if not shared_weight:
            c("batch_w_stride", nc * kc * wb)
            lines.append("    %batch_wn = arith.addi %batch_w, %batch_w_stride : i32")
        for i in batch_pages:
            c(page_name(i, "batchstride"), exponent_pages[i][2])
            lines.append(f"    %{page_name(i, 'batch_next')} = arith.addi %{page_name(i, 'batch')}, %{page_name(i, 'batchstride')} : i32")
        lines += [
            "    %batch_an = arith.addi %batch_a, %batch_a_stride : i32",
            "    %batch_yn = arith.addi %batch_y, %batch_y_stride : i32",
            f"    cf.br ^batch_header(%end, %batch_an{batch_w_next}, %batch_yn{values(page_name(i, "batch_next") for i in batch_pages)} : {state}, i32{batch_w_type}, i32{batch_types})",
            f"  ^all_done(%final: {state}):",
            *(restoration.finish("final") if restoration is not None else [f"    return %final : {state}"]),
        ]
    else:
        lines.extend(restoration.finish("end") if restoration is not None else [f"    return %end : {state}"])
    lines += ["  }", "}"]
    return "\n".join(lines) + "\n"


def tuning_key(snapshot, request, fixed_inputs, fixed_outputs, limits):
    """Specialization identity, excluding the choice being measured."""
    import hashlib
    import os
    from pathlib import Path

    from merlin.semantic_compiler import codegen

    from .binding import _native_pipeline
    from .private_staging import storage_policy
    from .exponent_page_cache import enabled as exponent_page_cache_enabled

    tool = Path(os.environ["ATLAS_OOT_BIN_DIR"]) / "atlas-opt"
    staging_storage = storage_policy()
    return {
        "request_digest": request.digest(),
        "target_identity": request.target_identity,
        "snapshot_sha256": hashlib.sha256((snapshot.root / "manifest.json").read_bytes()).hexdigest(),
        "atlas_opt_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        "atlas_emit_sha256": hashlib.sha256((tool.parent / "atlas-emit").read_bytes()).hexdigest(),
        "provider_sources": {
            path.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "codegen_options_sha256": hashlib.sha256(Path(codegen.__file__).read_bytes()).hexdigest(),
        "fixed_inputs": fixed_inputs,
        "fixed_outputs": list(fixed_outputs),
        "pipeline": _native_pipeline(),
        "operand_reuse": _operand_reuse_policy(),
        **({"staging_storage": staging_storage} if staging_storage != "dram" else {}),
        **({"exponent_page_cache": True} if exponent_page_cache_enabled() else {}),
        "search_limits": limits.record(),
        "program_size_limit_bytes": codegen.current_codegen_options().program_size_limit_bytes,
        **({"execution_instruction_limit": codegen.current_codegen_options().execution_instruction_limit}
           if codegen.current_codegen_options().execution_instruction_limit is not None else {}),
    }


def _operand_reuse_policy():
    policy = os.environ.get("ATLAS_LOOPED_OPERAND_REUSE", "ordered")
    if policy not in ("ordered", "resident-a"):
        raise NativeCompilationError(
            "unsupported_semantics", "ATLAS_LOOPED_OPERAND_REUSE must be ordered or resident-a"
        )
    return policy


def _virtual_schedule(plan, addresses, output, epilogue=None, *, seed_address=None, k_unroll=1):
    policy = _operand_reuse_policy()
    record = {"operand_reuse_requested": policy}
    if policy == "resident-a" and plan.counts[1] > 1:
        from .binding import _native_pipeline
        from .resident_looped_tiling import schedule_record, virtual_resident_a

        if _native_pipeline() != "mandatory":
            raise NativeCompilationError(
                "unsupported_semantics", "resident-a is qualified with ATLAS_NATIVE_PIPELINE=mandatory only"
            )
        record.update(schedule_record(plan))
        virtual = virtual_resident_a(plan, addresses, output, epilogue, seed_address=seed_address)
    else:
        record["operand_reuse"] = "ordered"
        if policy == "resident-a":
            record["operand_reuse_fallback"] = "one output-column tile provides no activation reuse"
        virtual = _virtual(plan, addresses, output, epilogue, seed_address=seed_address, k_unroll=k_unroll)
    return virtual, record


def compile_looped(snapshot, request, *, fixed_inputs, fixed_outputs, target_source, destination, limits):
    from merlin.semantic_compiler.codegen import choose_reduction_unroll

    from .binding import (
        DRAM_BASE,
        NATIVE_CHECKS,
        NATIVE_PIPELINES,
        RTL_REVISION,
        TILE_BYTES,
        _native_pipeline,
        _run,
        dram_slots,
    )

    unroll, codegen = choose_reduction_unroll(tuning_key(snapshot, request, fixed_inputs, fixed_outputs, limits))
    geometry, seed = split_seeded_request(request)
    contraction, epilogue = split_epilogue(geometry)
    full_plan, plan, streaming = prepare_execution_plan(request)
    if snapshot.profile.target_identity != request.target_identity:
        raise NativeCompilationError("compile_error", "loop source differs from selected native target")
    if streaming is not None and streaming.k_block_tiles is not None:
        from .chunked_tiling import compile_chunked

        return compile_chunked(
            snapshot,
            request,
            streaming,
            fixed_inputs=fixed_inputs,
            fixed_outputs=fixed_outputs,
            target_source=target_source,
            destination=destination,
            limits=limits,
        )
    sources = (*request.nodes[:2], *((seed,) if seed is not None else ()))
    if set(fixed_inputs) != {n.id for n in sources} or len(fixed_outputs) != 1:
        raise NativeCompilationError("compile_error", "loop ABI must bind all inputs and one output")
    slots = tuple(fixed_inputs[n.id] for n in sources) + tuple(fixed_outputs)
    sizes = (*plan.buffer_bytes[:2], *((plan.buffer_bytes[2],) if seed else ()), plan.buffer_bytes[2])
    if any(
        type(slot) is not int or slot < 0 or slot * TILE_BYTES + size > dram_slots() * TILE_BYTES
        for slot, size in zip(slots, sizes, strict=True)
    ):
        raise NativeCompilationError("resource_limit", "loop buffers exceed the selected diagnostic DRAM window")
    ranges = [
        (DRAM_BASE + slot * TILE_BYTES, DRAM_BASE + slot * TILE_BYTES + size)
        for slot, size in zip(slots, sizes, strict=True)
    ]
    if any(a < d and c < b for i, (a, b) in enumerate(ranges) for c, d in ranges[i + 1 :]):
        raise NativeCompilationError("infeasible_candidate", "loop input/output buffers overlap")
    # Select a bounded representative reset/continuation body through the native
    # engine. Outer output iteration and byte packing stay in the tiling plan.
    representative_k = plan.tile_k * (2 if plan.counts[2] > 1 else 1)
    nodes = (
        *[
            replace(n, type=replace(n.type, shape=(rows, representative_k)))
            for n, rows in zip(sources[:2], (plan.tile_m, plan.tile_n), strict=True)
        ],
        replace(
            contraction.nodes[-1],
            type=TensorType((plan.tile_m, plan.tile_n), "bf16", plan.qualified_policy),
        ),
    )
    representative = replace(contraction, nodes=nodes)
    body = partition_ordered_matmul_k(
        representative, tile_k=plan.tile_k, qualified_policy=plan.qualified_policy
    ).request
    if seed is not None:
        from .binding import _mxu0_accumulate_maps

        inputs = tuple(node for node in body.nodes if node.effect == "input")
        reductions = tuple(node for node in body.nodes if node.effect == "pure")
        seed_tile = replace(seed, type=replace(seed.type, shape=(plan.tile_m, plan.tile_n)))
        first = replace(
            reductions[0],
            op="matmul_accumulate",
            inputs=(*reductions[0].inputs, seed_tile.id),
            index_maps=_mxu0_accumulate_maps(),
        )
        body = replace(
            body,
            nodes=(*inputs, seed_tile, first, *reductions[1:]),
            input_storages=(*body.input_storages, (seed_tile.id, "external")),
        )
    selected = snapshot.select(body, limits=limits)
    if selected.status != "selected" or not selected.graph or not selected.rules or not selected.allocation:
        raise NativeCompilationError(selected.status, "native loop-body selection: " + selected.reason)
    names = [
        selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
        for value in selected.graph.values
        if value.kind == "instruction"
    ]
    required_start = (
        ("atlas.load_mxu0_acc_bf16_tile",) if seed else ("atlas.mxu0_reset_tile", "atlas.mxu0_reset_acc_tile")
    )
    if not any(name in names for name in required_start) or (
        (seed is not None or plan.counts[2] > 1) and "atlas.mxu0_accumulate_tile" not in names
    ):
        raise NativeCompilationError(
            "unqualified_target", "selected loop body has no admitted reset/continuation lowering"
        )
    epilogue_selection = None
    if epilogue is not None:
        # The loop explicitly reads its completed accumulator into a BF16
        # register pair. Select the pointwise stage at that existing boundary;
        # no copy or change of numerical policy is implied by this composition.
        completed = SemanticNode(
            "completed",
            "input",
            (),
            TensorType((plan.tile_m, plan.tile_n), "bf16", plan.qualified_policy),
            effect="input",
        )
        final = replace(
            epilogue,
            inputs=(completed.id,),
            type=replace(epilogue.type, shape=completed.type.shape),
        )
        epilogue_request = KernelRequest(
            (completed, final),
            (final.id,),
            ("tensor_bf16",),
            ((completed.id, "tensor_bf16"),),
            request.target_identity,
            request.lowering_policy,
            request.source_identity,
        )
        picked = snapshot.select(epilogue_request, limits=limits)
        if picked.status != "selected" or not picked.graph or not picked.rules or not picked.allocation:
            raise NativeCompilationError(picked.status, "native epilogue selection: " + picked.reason)
        epi_names = [
            picked.rules.symbols[value.symbol].get("descriptor", {}).get("name")
            for value in picked.graph.values
            if value.kind == "instruction"
        ]
        if epi_names != ["atlas.vpu_relu_bf16_from_mxu0_tile"]:
            raise NativeCompilationError(
                "unqualified_target",
                "native epilogue selected an unadmitted realization",
            )
        epilogue_selection = {
            "engine": picked.engine,
            "request_digest": epilogue_request.digest(),
            "source_request_digest": request.digest(),
            "check_fingerprint": picked.check_fingerprint,
            "selection_scope": "completed_bf16_register_pair_to_terminal_epilogue",
            "body_request": epilogue_request.record(),
            "candidate_digest": picked.candidate.digest(),
            "selected_graph": asdict(picked.graph),
            "allocation": asdict(picked.allocation),
            "instructions": epi_names,
            "candidate_attempts": picked.candidate_attempts,
            "ordering_attempts": picked.ordering_attempts,
        }
    virtual, schedule = _virtual_schedule(
        plan, (ranges[0][0], ranges[1][0]), ranges[-1][0], epilogue,
        seed_address=ranges[2][0] if seed else None, k_unroll=unroll,
    )
    codegen.update(schedule)
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
    pipeline = _native_pipeline()
    if codegen["program_size_limit_bytes"] is not None:
        virtual = virtual.replace(
            "attributes {",
            f"attributes {{atlas.program_size_limit_bytes = {codegen['program_size_limit_bytes']} : i64, ",
            1,
        )
    from merlin.semantic_compiler.codegen import materialization_cache

    cache = materialization_cache()
    cache_key = (
        snapshot.profile.digest(),
        pipeline,
        virtual,
        hashlib.sha256((tools / "atlas-opt").read_bytes()).hexdigest(),
        hashlib.sha256((tools / "atlas-emit").read_bytes()).hexdigest(),
    )
    cached = cache.get(cache_key) if cache is not None else None
    codegen["materialization_cache_hit"] = cached is not None
    if cached is None:
        mandatory = _run([str(tools / "atlas-opt"), *NATIVE_PIPELINES["mandatory"], "-"], input_text=virtual)
        _run([str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"], input_text=mandatory)
        machine = (
            mandatory
            if pipeline == "mandatory"
            else _run([str(tools / "atlas-opt"), *NATIVE_PIPELINES[pipeline], "-"], input_text=virtual)
        )
    else:
        mandatory, machine = cached
    # Cache hits still replay the full physical/resource/timing checks.
    _run([str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"], input_text=machine)
    if cache is not None:
        cache[cache_key] = mandatory, machine
    program = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
    if program.get("selected_rtl_revision") != RTL_REVISION:
        raise NativeCompilationError("unqualified_target", "loop emitter uses a different RTL revision")
    binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"])
    artifacts = {
        "source-request.json": (json.dumps(request.record(), sort_keys=True, indent=2) + "\n").encode(),
        "atlas-virtual.mlir": virtual.encode(),
        "atlas-machine.mlir": machine.encode(),
        "program.bin": binary,
        "physical-program.json": (json.dumps(program, indent=2) + "\n").encode(),
        "looped-mnk-tiling.json": (json.dumps(full_plan.record(), sort_keys=True, indent=2) + "\n").encode(),
        "tile-body-selection.json": (
            json.dumps(
                {
                    "engine": selected.engine,
                    "request_digest": body.digest(),
                    "source_request_digest": request.digest(),
                    "check_fingerprint": selected.check_fingerprint,
                    "selection_scope": "representative_seeded_body"
                    if seed
                    else "representative_reset_and_continuation_body",
                    "body_request": body.record(),
                    "candidate_digest": selected.candidate.digest(),
                    "selected_graph": asdict(selected.graph),
                    "allocation": asdict(selected.allocation),
                    "instructions": names,
                    "candidate_attempts": selected.candidate_attempts,
                    "ordering_attempts": selected.ordering_attempts,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode(),
        "execution_plan.json": (
            json.dumps(
                {
                    "schema": "atlas.looped_mnk_execution_plan.v1",
                    "codegen": codegen,
                    "scope": "diagnostic_selected_standalone_core",
                    "shape": list(plan.extents[:2]),
                    "reduction_extent": plan.extents[2],
                    "input_addresses": [a for a, _ in ranges[:-1]],
                    "input_buffer_bytes": list(sizes[:-1]),
                    "output_addresses": [ranges[-1][0]],
                    "output_bytes": sizes[-1],
                    "output_physical_layouts": ["mn_tiles_bf16_pair_halves"],
                    "numerical_policy": plan.qualified_policy,
                    "tiling": plan.record(),
                    **(
                        {
                            "accumulator_seed": {
                                "input_index": 2,
                                "conversion": "none",
                                "physical_layout": "mn_tiles_bf16_pair_halves",
                            }
                        }
                        if seed
                        else {}
                    ),
                    **({"streaming": streaming.record()} if streaming else {}),
                    **({"epilogue": epilogue.record()} if epilogue is not None else {}),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode(),
    }
    if epilogue_selection is not None:
        artifacts["epilogue-selection.json"] = (
            json.dumps(epilogue_selection, sort_keys=True, indent=2) + "\n"
        ).encode()

    if pipeline == "optimized":
        artifacts["mandatory-qualification-machine.mlir"] = mandatory.encode()
        artifacts["mandatory-qualification-program.json"] = _run(
            [str(tools / "atlas-emit"), "--program-json", "-"], input_text=mandatory
        ).encode()

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    manifest = {
        "schema": "atlas.native_looped_mnk_compilation.v1",
        "artifact_kind": "program",
        "engine": "merlin_native",
        "compile_mode": "diagnostic",
        "qualification_scope": "diagnostic_only",
        **({"execution_requires": "streamed_host_dispatch"} if streaming else {}),
        "request_digest": request.digest(),
        "target_identity": request.target_identity,
        "lowering_pipeline": pipeline,
        "semantic_qualification_pipeline": "mandatory",
        "binary_sha256": sha(binary),
        "execution_plan_sha256": sha(artifacts["execution_plan.json"]),
        "stage_artifacts": {name: sha(data) for name, data in artifacts.items() if name != "program.bin"},
        "tile_body_check_fingerprint": selected.check_fingerprint,
        "native_snapshot_sha256": sha((snapshot.root / "manifest.json").read_bytes()),
        "io_memory_bytes": dram_slots() * TILE_BYTES,
        "binding_source_sha256": sha(Path(__file__).with_name("binding.py").read_bytes()),
        "loop_lowering_source_sha256": sha(Path(__file__).read_bytes()),
        **({"resident_loop_lowering_source_sha256": sha(Path(__file__).with_name("resident_looped_tiling.py").read_bytes())}
           if codegen["operand_reuse"] == "resident-a" else {}),
        "atlas_opt_sha256": sha((tools / "atlas-opt").read_bytes()),
        "atlas_emit_sha256": sha((tools / "atlas-emit").read_bytes()),
        "target_inventory_sha256": sha((target_source / "docs/selected-variant-inventory.json").read_bytes()),
    }
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in artifacts.items():
        (destination / name).write_bytes(data)
    (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest
