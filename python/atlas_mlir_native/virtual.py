"""Materialize the selected candidate as Atlas virtual device IR.

Compiler-side admission/replay retains its diagnostic scope. Independent reference
and evaluator authority remain separate; no hardware execution is established.
"""

from __future__ import annotations

from merlin.compiler.semantic.target_binding import NativeCompilationError

from .policy import (
    BF16_BINARY_MODES,
    BF16_ROW_REDUCTION_MODES,
    BF16_UNARY_MODES,
    DRAM_BASE,
    PACK_RULE_NAMES,
    STORE_RULE_NAMES,
    TILE_BYTES,
    TRANSPOSE_RULE_NAMES,
)


def _virtual_from_selected(
    graph,
    order: tuple[int, ...],
    names: list[str | None],
    source_ids: tuple[str, ...],
    input_slots: tuple[int, ...],
    output_slots: tuple[int, ...],
    *,
    pack_scale_code: int = 127,
    pack_scale_codes: dict[str, int] | None = None,
) -> str:
    """Materialize the selected instruction DAG through admitted Atlas virtual ops."""
    input_base = DRAM_BASE + input_slots[0] * TILE_BYTES
    output_base = DRAM_BASE + output_slots[0] * TILE_BYTES
    indexed_inputs = all(
        slot == input_slots[0] + index for index, slot in enumerate(input_slots)
    )
    indexed_outputs = all(
        slot == output_slots[0] + index for index, slot in enumerate(output_slots)
    )
    lines = [
        "module {",
        f"  func.func @native_tile() -> !atlas.virtual_state attributes "
        f"{{atlas.input_dram_base = {input_base} : i64, atlas.output_dram_base = {output_base} : i64}} {{",
        '    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state',
    ]
    values: dict[int, str] = {}
    state = 0
    fp8_store_size_declared = False
    for value_id in order:
        value = graph.value(value_id)
        name = names[value_id]
        if name in {
            "atlas.load_bf16_tile",
            "atlas.load_bf16_tile_mixed",
            "atlas.load_fp8_tile",
        }:
            if len(value.children) != 1 or graph.value(value.children[0]).kind not in {
                "input",
                "constant",
            }:
                raise NativeCompilationError(
                    "unqualified_target",
                    "selected tile load needs a DRAM boundary value",
                )
            source = graph.value(value.children[0]).source_node
            if source not in source_ids:
                raise NativeCompilationError(
                    "unqualified_target", "selected input is absent from the DRAM ABI"
                )
            source_index = source_ids.index(source)
            dtype = (
                "bf16"
                if name in {"atlas.load_bf16_tile", "atlas.load_bf16_tile_mixed"}
                else "fp8"
            )
            if indexed_inputs:
                lines.append(
                    f'    %s{state + 1}, %v{value_id} = "atlas.virtual_input_{dtype}"(%s{state}) '
                    f"{{index = {source_index} : i32}} : (!atlas.virtual_state) -> "
                    f"(!atlas.virtual_state, !atlas.virtual_{dtype})"
                )
                state += 1
            else:
                address = DRAM_BASE + input_slots[source_index] * TILE_BYTES - (1 << 32)
                size = 2048 if dtype == "bf16" else 1024
                lines.extend(
                    (
                        f"    %addr{value_id} = arith.constant {address} : i32",
                        f"    %bytes{value_id} = arith.constant {size} : i32",
                        (
                            f'    %s{state + 1}, %v{value_id} = "atlas.virtual_dma_load_{dtype}"'
                            f"(%s{state}, %addr{value_id}, %bytes{value_id}) : "
                            f"(!atlas.virtual_state, i32, i32) -> "
                            f"(!atlas.virtual_state, !atlas.virtual_{dtype})"
                        ),
                    )
                )
                state += 1
            values[value_id] = f"%v{value_id}"
            continue
        try:
            operands = tuple(values[child] for child in value.children)
        except KeyError as exc:
            raise NativeCompilationError(
                "unqualified_target", "selected instruction order omits a producer"
            ) from exc
        if name in {"atlas.xlu_transpose_fp8", *TRANSPOSE_RULE_NAMES}:
            lines.append(
                f'    %v{value_id} = "atlas.virtual_xlu_transpose"({operands[0]}) '
                ": (!atlas.virtual_fp8) -> !atlas.virtual_fp8"
            )
            values[value_id] = f"%v{value_id}"
        elif name in {"atlas.vpu_relu_bf16_tile", "atlas.vpu_relu_bf16_from_mxu0_tile"}:
            lines.append(
                f'    %v{value_id} = "atlas.virtual_vpu_unary"({operands[0]}) '
                '{kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'
            )
            values[value_id] = f"%v{value_id}"
        elif name in BF16_UNARY_MODES:
            kind = BF16_UNARY_MODES[name]
            lines.append(
                f'    %v{value_id} = "atlas.virtual_vpu_unary"({operands[0]}) '
                f'{{kind = "{kind}"}} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'
            )
            values[value_id] = f"%v{value_id}"
        elif name in BF16_ROW_REDUCTION_MODES:
            kind = BF16_ROW_REDUCTION_MODES[name]
            lines.append(
                f'    %v{value_id} = "atlas.virtual_vpu_reduce"({operands[0]}) '
                f'{{kind = "{kind}"}} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'
            )
            values[value_id] = f"%v{value_id}"
        elif name in BF16_BINARY_MODES:
            kind = BF16_BINARY_MODES[name]
            lines.append(
                f'    %v{value_id} = "atlas.virtual_vpu_binary"({operands[0]}, {operands[1]}) '
                f'{{kind = "{kind}"}} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16'
            )
            values[value_id] = f"%v{value_id}"
        elif name in PACK_RULE_NAMES:
            selected_code = (pack_scale_codes or {}).get(
                value.source_node, pack_scale_code
            )
            lines.append(
                f'    %v{value_id} = "atlas.virtual_pack_fp8"({operands[0]}) '
                f"{{scale_code = {selected_code} : i32}} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8"
            )
            values[value_id] = f"%v{value_id}"
        elif name in {
            "atlas.mxu0_reset_tile",
            "atlas.mxu1_reset_tile",
            "atlas.mxu1_finite_reset_tile",
        }:
            unit = 0 if name == "atlas.mxu0_reset_tile" else 1
            lines.extend(
                (
                    f'    %s{state + 1}, %weight{value_id} = "atlas.virtual_mxu_load_weight"(%s{state}, {operands[1]}) '
                    + f"{{unit = {unit} : i32}} : (!atlas.virtual_state, !atlas.virtual_fp8) -> "
                    + f"(!atlas.virtual_state, !atlas.virtual_mxu_weight<{unit}>)",
                    f'    %s{state + 2}, %acc{value_id} = "atlas.virtual_mxu_reset"'
                    f"(%s{state + 1}, {operands[0]}, %weight{value_id}) "
                    + f": (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<{unit}>) -> "
                    + f"(!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>)",
                    f'    %s{state + 3}, %v{value_id} = "atlas.virtual_mxu_readout_bf16"'
                    f'(%s{state + 2}, %acc{value_id}) '
                    + f": (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>) -> "
                    + "(!atlas.virtual_state, !atlas.virtual_bf16)",
                )
            )
            state += 3
            values[value_id] = f"%v{value_id}"
        elif name == "atlas.mxu0_reset_acc_tile":
            lines.extend(
                (
                    f'    %s{state + 1}, %weight{value_id} = "atlas.virtual_mxu_load_weight"(%s{state}, {operands[1]}) '
                    + "{unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> "
                    + "(!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)",
                    f'    %s{state + 2}, %acc{value_id} = "atlas.virtual_mxu_reset"'
                    f"(%s{state + 1}, {operands[0]}, %weight{value_id}) "
                    + ": (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> "
                    + "(!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)",
                )
            )
            state += 2
            values[value_id] = f"%acc{value_id}"
        elif name == "atlas.mxu0_accumulate_tile":
            lines.extend(
                (
                    f'    %s{state + 1}, %weight{value_id} = "atlas.virtual_mxu_load_weight"(%s{state}, {operands[1]}) '
                    + "{unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> "
                    + "(!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)",
                    f'    %s{state + 2}, %acc{value_id} = "atlas.virtual_mxu_accumulate"'
                    + f"(%s{state + 1}, {operands[0]}, %weight{value_id}, {operands[2]}) "
                    + ": (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>, "
                    + "!atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)",
                )
            )
            state += 2
            values[value_id] = f"%acc{value_id}"
        elif name in {
            "atlas.store_bf16_tile",
            "atlas.store_bf16_mxu0_tile",
            "atlas.store_bf16_mxu1_tile",
            "atlas.store_bf16_mxu1_finite_tile",
            "atlas.store_bf16_mxu0_acc_tile",
        }:
            if value_id not in graph.outputs:
                raise NativeCompilationError(
                    "unqualified_target", "selected BF16 store is not an output root"
                )
            if name == "atlas.store_bf16_mxu0_acc_tile":
                lines.append(
                    f'    %s{state + 1}, %readout{value_id} = "atlas.virtual_mxu_readout_bf16"'
                    f"(%s{state}, {operands[0]}) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) "
                    "-> (!atlas.virtual_state, !atlas.virtual_bf16)"
                )
                state += 1
                operands = (f"%readout{value_id}",)
            output_index = graph.outputs.index(value_id)
            if indexed_outputs:
                lines.append(
                    f'    %s{state + 1} = "atlas.virtual_output_bf16"(%s{state}, {operands[0]}) '
                    f"{{index = {output_index} : i32}} : "
                    "(!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state"
                )
                state += 1
            else:
                address = (
                    DRAM_BASE + output_slots[output_index] * TILE_BYTES - (1 << 32)
                )
                lines.extend(
                    (
                        f"    %out{value_id} = arith.constant {address} : i32",
                        f"    %outbytes{value_id} = arith.constant 2048 : i32",
                        (
                            f'    %s{state + 1} = "atlas.virtual_dma_store_bf16"'
                            f"(%s{state}, {operands[0]}, %out{value_id}, %outbytes{value_id}) : "
                            f"(!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state"
                        ),
                    )
                )
                state += 1
            values[value_id] = operands[0]
        elif name in {"atlas.store_fp8_tile", *STORE_RULE_NAMES}:
            if value_id not in graph.outputs:
                raise NativeCompilationError(
                    "unqualified_target", "selected FP8 store is not an output root"
                )
            output_index = graph.outputs.index(value_id)
            address = DRAM_BASE + output_slots[output_index] * TILE_BYTES - (1 << 32)
            lines.append(f"    %out{value_id} = arith.constant {address} : i32")
            if not fp8_store_size_declared:
                lines.append("    %fp8_store_bytes = arith.constant 1024 : i32")
                fp8_store_size_declared = True
            lines.extend(
                (
                    f'    %s{state + 1} = "atlas.virtual_dma_store_fp8"'
                    f"(%s{state}, {operands[0]}, %out{value_id}, %fp8_store_bytes) "
                    + ": (!atlas.virtual_state, !atlas.virtual_fp8, i32, i32) -> "
                    + "!atlas.virtual_state",
                )
            )
            state += 1
            values[value_id] = operands[0]
        else:
            raise NativeCompilationError(
                "unsupported_semantics",
                f"no Atlas virtual emission for selected {name}",
            )
    lines.extend((f"    return %s{state} : !atlas.virtual_state", "  }", "}"))
    return "\n".join(lines) + "\n"
