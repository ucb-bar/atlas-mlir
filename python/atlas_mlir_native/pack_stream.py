"""Replay admitted PACK and transpose command streams without changing their domains.

Compiler-side admission/replay retains its diagnostic scope. Independent reference
and evaluator authority remain separate; no hardware execution is established.
"""

from __future__ import annotations

from merlin.semantic_compiler.target_binding import NativeCompilationError

from .transfer import _check_atomic_dma_schedule, _check_mxu_dag_timing, _tile_effects


def _pack_roundtrip_effects(
    source: int, pair: int, destination: int, scale_code: int
) -> list[tuple]:
    """Retain PACK's exact conversion, then canonicalize its strip layout."""
    if (
        type(scale_code) is not int
        or scale_code not in range(122, 129)
        or type(pair) is not int
        or pair % 2
        or not 32 <= pair <= 60
        or type(destination) is not int
        or not 0 <= destination < 32
    ):
        raise NativeCompilationError(
            "unqualified_target", "pack roundtrip placement or scale changed"
        )
    return [
        ("seli", 3, scale_code),
        ("pack", "bf16_to_fp8", source, 62, 3),
        ("seli", 3, 127),
        ("pack", "fp8_to_bf16", 62, pair, 3),
        ("mxu_push", 0, 0, "acc_bf16", pair),
        ("seli", 3, 127),
        ("mxu_pop", 0, 0, "fp8", 3, destination),
    ]


def _check_pack_roundtrip_words(program: dict) -> None:
    """Bind conversion fields, scale loads and MXU copy/pop to actual words."""
    for row in program["instructions"]:
        op, fields = row["operation"], row["fields"]
        if op == "atlas.vpu_pack":
            direction = fields.get("direction")
            if direction not in {"bf16_to_fp8", "fp8_to_bf16"}:
                raise NativeCompilationError(
                    "unqualified_target", "pack conversion direction changed"
                )
            word = (
                (0x44 if direction == "bf16_to_fp8" else 0x45) << 25
                | fields["src"] << 19
                | fields["scale_reg"] << 13
                | fields["dst"] << 7
                | 0x57
            )
        elif op == "atlas.scalar_load" and fields.get("kind") == "seli":
            word = (
                fields["offset"] << 20
                | fields["base"] << 15
                | 7 << 12
                | fields["dst"] << 7
                | 3
            )
        elif op == "atlas.mxu_push" and fields.get("kind") == "acc_bf16":
            word = (
                (4 + fields["unit"]) << 25
                | fields["src"] << 13
                | fields["slot"] << 7
                | 0x77
            )
        elif op == "atlas.mxu_pop" and fields.get("format") == "fp8":
            word = (
                (6 + fields["unit"]) << 25
                | fields["slot"] << 19
                | fields["scale_reg"] << 13
                | fields["dst"] << 7
                | 0x77
            )
        else:
            continue
        if row.get("word_u32") != word:
            raise NativeCompilationError(
                "unqualified_target", "pack roundtrip word differs from its fields"
            )


def check_pack_stream(
    program: dict,
    *,
    input_address: int,
    output_address: int,
    source_pair: int = 32,
    relu_pair: tuple[int, int] | None = None,
    scale_code: int = 127,
) -> None:
    """Replay exact PACK/UNPACK, unit-scale MXU conversion and output DMA."""
    expected: list[tuple] = [
        ("config",),
        ("dma", "load", input_address, 0, 1024, 0),
        ("wait", 0),
        ("vload", relu_pair[0] if relu_pair else source_pair, 0),
        ("dma", "load", input_address + 1024, 256, 1024, 0),
        ("wait", 0),
        ("vload", (relu_pair[0] if relu_pair else source_pair) + 1, 256),
    ]
    pairs = (source_pair,) if relu_pair is None else relu_pair
    if any(
        type(pair) is not int or pair % 2 or not 32 <= pair <= 60 for pair in pairs
    ) or (
        relu_pair is not None
        and (relu_pair[0] == relu_pair[1] or source_pair != relu_pair[1])
    ):
        raise NativeCompilationError(
            "unqualified_target", "pack BF16 input placement changed"
        )
    if relu_pair is not None:
        expected.append(("relu", *relu_pair))
    effects = _tile_effects(program)
    unpacks = [effect for effect in effects if effect[:2] == ("pack", "fp8_to_bf16")]
    if len(unpacks) != 1:
        raise NativeCompilationError(
            "unqualified_target", "pack requires exactly one strip unpack"
        )
    expected.extend(_pack_roundtrip_effects(source_pair, unpacks[0][3], 0, scale_code))
    expected.extend(
        (
            ("vstore", 0, 131072),
            ("dma", "store", output_address, 131072, 1024, 1),
            ("wait", 1),
            ("complete",),
            ("trap",),
        )
    )
    if effects != expected:
        raise NativeCompilationError(
            "unqualified_target",
            "pack stream differs in layout, addresses, or transfer",
        )
    _check_pack_roundtrip_words(program)
    _check_mxu_dag_timing(program)


def check_transpose_stream(
    program: dict, *, input_address: int, output_address: int, source: int, result: int
) -> None:
    effects = _tile_effects(program)
    expected = [
        ("config",),
        ("dma", "load", input_address, 0, 1024, 0),
        ("wait", 0),
        ("vload", source, 0),
        ("transpose", source, result),
        ("vstore", result, 0x20000),
        ("dma", "store", output_address, 0x20000, 1024, 1),
        ("wait", 1),
        ("complete",),
        ("trap",),
    ]
    if effects != expected:
        raise NativeCompilationError(
            "unqualified_target",
            "emitted transpose addresses, scratch words, order, waits, or registers differ",
        )
    # Replay issue gaps from the final words, separately from the OOT delay
    # insertion pass. These are conservative diagnostic floors, not an RTL
    # latency qualification for an integrated core.
    issued: dict[str, int] = {}
    cycle = 0
    for row in program["instructions"]:
        op = row["operation"]
        if op in {"atlas.vload", "atlas.xlu_transpose", "atlas.vstore"} or (
            op == "atlas.dma" and row["fields"].get("direction") == "store"
        ):
            issued[op] = cycle
        if op == "atlas.delay":
            count = row["fields"].get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError(
                    "unqualified_target", "movement delay has invalid cycle count"
                )
            cycle += count
        cycle += 1
    for producer, consumer, minimum in (
        ("atlas.vload", "atlas.xlu_transpose", 34),
        ("atlas.xlu_transpose", "atlas.vstore", 65),
        ("atlas.vstore", "atlas.dma", 34),
    ):
        if issued[consumer] - issued[producer] < minimum:
            raise NativeCompilationError(
                "unqualified_target",
                f"emitted transpose has premature {consumer} after {producer}",
            )


def check_copy_transpose_stream(
    program: dict,
    *,
    input_address: int,
    output_addresses: tuple[int, int],
    source: int,
    result: int,
) -> None:
    effects = _tile_effects(program)
    expected = [
        ("config",),
        ("dma", "load", input_address, 0, 1024, 0),
        ("wait", 0),
        ("vload", source, 0),
        ("vstore", source, 0x20000),
        ("dma", "store", output_addresses[0], 0x20000, 1024, 1),
        ("wait", 1),
        ("transpose", source, result),
        ("vstore", result, 0x20000),
        ("dma", "store", output_addresses[1], 0x20000, 1024, 1),
        ("wait", 1),
        ("complete",),
        ("trap",),
    ]
    _check_atomic_dma_schedule(effects, expected, context="copy/transpose stream")
    issued: dict[str, list[int]] = {}
    cycle = 0
    for row in program["instructions"]:
        op = row["operation"]
        if op in {"atlas.vload", "atlas.vstore", "atlas.xlu_transpose"} or (
            op == "atlas.dma" and row["fields"].get("direction") == "store"
        ):
            issued.setdefault(op, []).append(cycle)
        if op == "atlas.delay":
            count = row["fields"].get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError(
                    "unqualified_target", "copy/transpose delay has invalid cycle count"
                )
            cycle += count
        cycle += 1
    gaps = (
        ("atlas.vload", 0, "atlas.vstore", 0, 34),
        ("atlas.vload", 0, "atlas.xlu_transpose", 0, 34),
        ("atlas.xlu_transpose", 0, "atlas.vstore", 1, 65),
        ("atlas.vstore", 0, "atlas.dma", 0, 34),
        ("atlas.vstore", 1, "atlas.dma", 1, 34),
    )
    if any(
        issued[end][end_index] - issued[start][start_index] < minimum
        for start, start_index, end, end_index, minimum in gaps
    ):
        raise NativeCompilationError(
            "unqualified_target", "emitted copy/transpose has premature use"
        )
