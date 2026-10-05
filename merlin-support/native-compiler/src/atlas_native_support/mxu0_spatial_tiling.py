"""Complete M/N/K tile partition for the selected ordered MXU0 contraction.

Only physical boundary packing is performed on the host. Every contraction
and transfer in the lowered graph still goes through Merlin-native selection.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType

from .mxu0 import Mxu0Contract, mxu0_profile
from .mxu0_tiling import OrderedKTilePlan, lower_ordered_k_contraction

# Outer tiling policy: avoid one extraction product over many independent roots.
MAX_COMBINED_SPATIAL_NODES = 24


@dataclass(frozen=True)
class SpatialTilePlan:
    source: KernelRequest
    lowered: KernelRequest
    panels: tuple[tuple[str, str, int, int], ...]
    output_tiles: tuple[tuple[str, int, int], ...]
    m_size: int
    n_size: int
    k_size: int
    tile_rows: int
    tile_cols: int
    tile_k: int

    @property
    def host_activity(self) -> str:
        return "row_major_fp8_to_zero_padded_mnk_panels_and_bf16_pair_tiles_copy"

    def record(self) -> dict[str, object]:
        return {
            "schema": "atlas.spatial_tile_plan.v1",
            "source_request_digest": self.source.digest(),
            "lowered_request_digest": self.lowered.digest(),
            "source_layout": "row_major_fp8",
            "panel_layout": "row_major_fp8_k_panel",
            "result_layout": "column_halves_row_major_bf16_pair_tiles",
            "m_size": self.m_size,
            "n_size": self.n_size,
            "k_size": self.k_size,
            "tile_rows": self.tile_rows,
            "tile_cols": self.tile_cols,
            "tile_k": self.tile_k,
            "padding_encoding": "fp8_e4m3_positive_zero",
            "panels": [
                {
                    "source": source,
                    "panel": panel,
                    "row_start": row_start,
                    "row_stop": min(
                        row_start + (self.tile_rows if source == self.source.nodes[0].id
                                     else self.tile_cols),
                        self.m_size if source == self.source.nodes[0].id else self.n_size,
                    ),
                    "k_start": k_start,
                    "k_stop": min(k_start + self.tile_k, self.k_size),
                }
                for source, panel, row_start, k_start in self.panels
            ],
            "output_tiles": [
                {"source": node, "m_start": m_start, "n_start": n_start}
                for node, m_start, n_start in self.output_tiles
            ],
        }

    def fixed_panel_inputs(self, fixed_inputs: dict[str, int]) -> dict[str, int]:
        sources = {source for source, _, _, _ in self.panels}
        if set(fixed_inputs) != sources or any(
            type(slot) is not int or slot < 0 for slot in fixed_inputs.values()
        ):
            raise ValueError("spatial source ABI needs one nonnegative slot per input")
        k_tiles = (self.k_size + self.tile_k - 1) // self.tile_k
        result = {
            panel: fixed_inputs[source] + row_start // (
                self.tile_rows if source == self.source.nodes[0].id else self.tile_cols
            ) * k_tiles + k_start // self.tile_k
            for source, panel, row_start, k_start in self.panels
        }
        if len(set(result.values())) != len(result):
            raise ValueError("spatial source ABI overlaps physical input panels")
        return result

    def fixed_tile_outputs(self, fixed_outputs: tuple[int, ...]) -> tuple[int, ...]:
        if (len(fixed_outputs) != 1 or type(fixed_outputs[0]) is not int
                or fixed_outputs[0] < 0 or fixed_outputs[0] % 2):
            raise ValueError("spatial source ABI needs one aligned BF16 pair base")
        return tuple(fixed_outputs[0] + 2 * index
                     for index in range(len(self.output_tiles)))

    def partition_outputs(
        self, fixed_inputs: dict[str, int], fixed_outputs: tuple[int, ...],
    ) -> tuple[tuple[KernelRequest, dict[str, int], tuple[int, ...]], ...]:
        """Keep each output's complete ordered K chain in one native request."""
        panel_addresses = self.fixed_panel_inputs(fixed_inputs)
        output_addresses = self.fixed_tile_outputs(fixed_outputs)
        input_slots = set(panel_addresses.values())
        output_slots = {slot + offset for slot in output_addresses for offset in (0, 1)}
        if input_slots & output_slots:
            raise ValueError("spatial output pair overlaps a live input panel")
        nodes_by_id = {node.id: node for node in self.lowered.nodes}
        input_storages = dict(self.lowered.input_storages)
        all_outputs = {node_id for node_id, _, _ in self.output_tiles}
        segments: list[tuple[KernelRequest, dict[str, int], tuple[int, ...]]] = []
        def visit(node_id: str, needed: set[str]) -> None:
            if node_id in needed:
                return
            node = nodes_by_id[node_id]
            for child in node.inputs:
                visit(child, needed)
            needed.add(node_id)

        for index, (output_id, _, _) in enumerate(self.output_tiles):
            needed: set[str] = set()
            visit(output_id, needed)
            if needed & (all_outputs - {output_id}):
                raise ValueError("one spatial output depends on another output tile")
            nodes = tuple(node for node in self.lowered.nodes if node.id in needed)
            boundaries = tuple((node.id, input_storages[node.id]) for node in nodes
                               if node.op == "input")
            segment = KernelRequest(
                nodes, (output_id,), ("external_bf16",), boundaries,
                self.lowered.target_identity, lowering_policy="strict-native",
                source_identity=f"atlas-spatial-output-v1:{self.source.digest()}:{index}",
            )
            segments.append((
                segment,
                {node_id: panel_addresses[node_id] for node_id, _ in boundaries},
                (output_addresses[index],),
            ))
        return tuple(segments)

    def panelize_inputs(self, runtime_inputs: dict[str, bytes]) -> dict[str, bytes]:
        sources = {source for source, _, _, _ in self.panels}
        if set(runtime_inputs) != sources:
            raise ValueError("spatial runtime inputs differ from the source signature")
        source_rows = {
            self.source.nodes[0].id: self.m_size,
            self.source.nodes[1].id: self.n_size,
        }
        for source in sources:
            payload = runtime_inputs[source]
            if type(payload) is not bytes or len(payload) != source_rows[source] * self.k_size:
                raise ValueError(f"spatial runtime input {source} has wrong byte length")
        result: dict[str, bytes] = {}
        for source, panel, row_start, k_start in self.panels:
            row_count = self.tile_rows if source == self.source.nodes[0].id else self.tile_cols
            payload = runtime_inputs[source]
            panel_rows = []
            for row in range(row_start, row_start + row_count):
                if row >= source_rows[source]:
                    panel_rows.append(bytes(self.tile_k))
                    continue
                first = row * self.k_size + k_start
                stop = row * self.k_size + min(k_start + self.tile_k, self.k_size)
                panel_rows.append(payload[first:stop].ljust(self.tile_k, b"\x00"))
            result[panel] = b"".join(panel_rows)
        return result

    def assemble_output(self, physical_tiles: dict[str, bytes]) -> bytes:
        """Copy checked BF16 pair tiles into one row-major source output."""
        if set(physical_tiles) != {node for node, _, _ in self.output_tiles}:
            raise ValueError("spatial output tiles differ from the compiled signature")
        output = bytearray(self.m_size * self.n_size * 2)
        half_columns = self.tile_cols // 2
        for node, m_start, n_start in self.output_tiles:
            payload = physical_tiles[node]
            if type(payload) is not bytes or len(payload) != self.tile_rows * self.tile_cols * 2:
                raise ValueError("spatial BF16 output tile has wrong byte length")
            for row in range(self.tile_rows):
                if m_start + row >= self.m_size:
                    continue
                for column in range(self.tile_cols):
                    if n_start + column >= self.n_size:
                        continue
                    pair_index = ((column // half_columns) * self.tile_rows * half_columns
                                  + row * half_columns + column % half_columns)
                    logical_index = ((m_start + row) * self.n_size + n_start + column)
                    output[2 * logical_index:2 * logical_index + 2] = (
                        payload[2 * pair_index:2 * pair_index + 2]
                    )
        return bytes(output)


def reserve_external_panel_slots(
    m_size: int, k_size: int, n_size: int, contract: Mxu0Contract,
) -> tuple[dict[str, int], tuple[int, ...]]:
    """Reserve disjoint, aligned external windows for one tiled contraction.

    Each FP8 panel occupies one physical matrix-register-sized block. A BF16
    output tile occupies an aligned pair. The returned bases are consumed by
    the native compiler's fixed external placement, not by an example kernel.
    """
    geometry = contract.record["geometry"]
    rows, cols = geometry["array_rows"], geometry["array_cols"]
    external_slots = (
        contract.movement.record["dram_window_bytes"]
        // geometry["matrix_register_bytes"]
    )
    if min(m_size, k_size, n_size, rows, cols, external_slots) <= 0:
        raise ValueError("region geometry and external storage must be positive")
    k_tiles = (k_size + rows - 1) // rows
    activation_panels = ((m_size + rows - 1) // rows) * k_tiles
    weight_panels = ((n_size + cols - 1) // cols) * k_tiles
    output_base = max(8, activation_panels + weight_panels)
    output_base += output_base % 2
    output_registers = 2 * ((m_size + rows - 1) // rows) * ((n_size + cols - 1) // cols)
    if output_base + output_registers > external_slots:
        raise ValueError("region panels and BF16 outputs exceed selected external storage")
    return {"activation": 0, "weight": activation_panels}, (output_base,)


def lower_spatial_contraction(
    request: KernelRequest, contract: Mxu0Contract
) -> SpatialTilePlan:
    """Tile one source contraction and an optional BF16 ReLU over M, N and K."""
    geometry = contract.record["geometry"]
    policy = contract.record["numerical_policy"]
    tile_rows, tile_cols = geometry["array_rows"], geometry["array_cols"]
    tile_k = tile_rows
    if request.target_identity != contract.identity or request.lowering_policy != "strict-native":
        raise ValueError("spatial source differs from selected strict-native contract")
    if len(request.nodes) not in (3, 4) or len(request.outputs) != 1 or request.constants:
        raise ValueError("spatial source needs two inputs, one contraction, and optional ReLU")
    activation, weight, result = request.nodes[:3]
    post = request.nodes[3] if len(request.nodes) == 4 else None
    if (activation.op, activation.effect, weight.op, weight.effect,
            result.op, result.effect) != ("input", "input", "input", "input", "contraction", "pure"):
        raise ValueError("spatial source has unadmitted operation or effect")
    expected_output = post.id if post is not None else result.id
    if (result.inputs != (activation.id, weight.id) or request.outputs != (expected_output,)
            or request.output_storages != ("external_bf16",)
            or dict(request.input_storages) != {
                activation.id: "external_fp8_row_major",
                weight.id: "external_fp8_row_major",
            }):
        raise ValueError("spatial source has unsupported graph or boundary representation")
    if post is not None and (
        post.op != "relu" or post.effect != "pure" or post.inputs != (result.id,)
        or post.type != result.type or post.attrs or post.index_maps
    ):
        raise ValueError("spatial post operation is outside selected BF16 ReLU scope")
    if len(activation.type.shape) != 2 or len(weight.type.shape) != 2:
        raise ValueError("spatial source operands must be rank two")
    m_size, k_size = activation.type.shape
    n_size, weight_k = weight.type.shape
    if (
        k_size != weight_k or min(m_size, n_size, k_size) <= 0
        or (m_size == tile_rows and n_size == tile_cols and k_size % tile_k == 0)
        or activation.type != TensorType((m_size, k_size), policy["operand_dtype"],
                                         policy["operand_policy"])
        or weight.type != TensorType((n_size, k_size), policy["operand_dtype"],
                                     policy["operand_policy"])
        or result.type != TensorType((m_size, n_size), policy["result_dtype"],
                                     policy["result_policy"])
        or result.attrs != (("reduction_order", "k_ascending"),)
    ):
        raise ValueError("spatial source shape or policy is outside complete-tile scope")
    reset = next(
        descriptor for descriptor in mxu0_profile(contract).descriptors
        if descriptor.name == "mxu0_matmul_reset"
    )
    if result.index_maps != reset.index_maps:
        raise ValueError("spatial source index maps differ from selected weight orientation")

    nodes: list[SemanticNode] = []
    panels: list[tuple[str, str, int, int]] = []
    input_storages: list[tuple[str, str]] = []
    for source, dimension, label, tile_extent in (
        (activation, m_size, "m", tile_rows),
        (weight, n_size, "n", tile_cols),
    ):
        for row_index, row_start in enumerate(range(0, dimension, tile_extent)):
            for k_index, k_start in enumerate(range(0, k_size, tile_k)):
                panel = f"{source.id}__{label}{row_index}k{k_index}"
                panels.append((source.id, panel, row_start, k_start))
                nodes.append(SemanticNode(
                    panel, "input", (),
                    TensorType((tile_extent, tile_k), source.type.dtype,
                               source.type.numerical_policy),
                    effect="input",
                ))
                input_storages.append((panel, "external_fp8"))
    output_tiles: list[tuple[str, int, int]] = []
    for m_index, m_start in enumerate(range(0, m_size, tile_rows)):
        for n_index, n_start in enumerate(range(0, n_size, tile_cols)):
            previous: str | None = None
            final_node = f"{result.id}__m{m_index}n{n_index}"
            for k_index in range((k_size + tile_k - 1) // tile_k):
                final = k_index == (k_size + tile_k - 1) // tile_k - 1
                node_id = final_node if final else f"{final_node}k{k_index}"
                operands = (f"{activation.id}__m{m_index}k{k_index}",
                            f"{weight.id}__n{n_index}k{k_index}")
                if previous is not None:
                    operands += (previous,)
                nodes.append(SemanticNode(
                    node_id, "contraction" if previous is None else "contraction_accumulate",
                    operands, TensorType((tile_rows, tile_cols), result.type.dtype,
                                         result.type.numerical_policy),
                    attrs=result.attrs,
                    index_maps=reset.index_maps if previous is None else (
                        *reset.index_maps[:2], reset.index_maps[2], reset.index_maps[2]
                    ),
                ))
                previous = node_id
            if post is not None:
                post_node = f"{post.id}__m{m_index}n{n_index}"
                nodes.append(SemanticNode(
                    post_node, "relu", (final_node,),
                    TensorType((tile_rows, tile_cols), post.type.dtype,
                               post.type.numerical_policy),
                ))
                final_node = post_node
            output_tiles.append((final_node, m_start, n_start))
    node_ids = [node.id for node in nodes]
    if len(set(node_ids)) != len(node_ids) or set(node_ids) & {node.id for node in request.nodes}:
        raise ValueError("spatial source names collide with generated tiles")
    lowered = KernelRequest(
        tuple(nodes), tuple(item[0] for item in output_tiles),
        ("external_bf16",) * len(output_tiles), tuple(input_storages),
        contract.identity, lowering_policy="strict-native",
        source_identity=f"atlas-spatial-tiles-v1:{request.digest()}",
    )
    return SpatialTilePlan(
        request, lowered, tuple(panels), tuple(output_tiles),
        m_size, n_size, k_size, tile_rows, tile_cols, tile_k,
    )


def lower_tiled_contraction(
    request: KernelRequest, contract: Mxu0Contract
) -> OrderedKTilePlan | SpatialTilePlan:
    """Select one checked Atlas tile partition by source dimensions."""
    if len(request.nodes) not in (3, 4) or len(request.nodes[2].type.shape) != 2:
        raise ValueError("tiled source needs one rank-two contraction and optional ReLU")
    m_size, n_size = request.nodes[2].type.shape
    geometry = contract.record["geometry"]
    if (len(request.nodes) == 4 or m_size != geometry["array_rows"]
            or n_size != geometry["array_cols"]
            or request.nodes[0].type.shape[1] % geometry["array_rows"]):
        return lower_spatial_contraction(request, contract)
    return lower_ordered_k_contraction(request, contract)
