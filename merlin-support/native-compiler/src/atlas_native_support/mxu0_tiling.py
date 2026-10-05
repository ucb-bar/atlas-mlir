"""Checked ordered-K partition for the selected MXU0 tile geometry.

This is an Atlas boundary/layout policy. Merlin still selects every lowered
machine instruction and allocates its physical storage from the target profile.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType

from .mxu0 import Mxu0Contract, mxu0_profile


@dataclass(frozen=True)
class OrderedKTilePlan:
    source: KernelRequest
    lowered: KernelRequest
    panels: tuple[tuple[str, str, int, int], ...]
    reduction_size: int
    tile_size: int

    @property
    def host_activity(self) -> str:
        return "row_major_fp8_to_k_panels_copy"

    def record(self) -> dict[str, object]:
        return {
            "schema": "atlas.ordered_k_tile_plan.v1",
            "source_request_digest": self.source.digest(),
            "lowered_request_digest": self.lowered.digest(),
            "reduction_size": self.reduction_size,
            "tile_size": self.tile_size,
            "source_layout": "row_major_fp8",
            "panel_layout": "row_major_fp8_k_panel",
            "panels": [
                {"source": source, "panel": panel, "k_start": start, "k_stop": stop}
                for source, panel, start, stop in self.panels
            ],
        }

    def fixed_panel_inputs(self, fixed_inputs: dict[str, int]) -> dict[str, int]:
        sources = {source for source, _, _, _ in self.panels}
        if set(fixed_inputs) != sources or any(type(slot) is not int or slot < 0 for slot in fixed_inputs.values()):
            raise ValueError("ordered-K source ABI needs one nonnegative slot per input")
        result = {
            panel: fixed_inputs[source] + start // self.tile_size
            for source, panel, start, _ in self.panels
        }
        if len(set(result.values())) != len(result):
            raise ValueError("ordered-K source ABI overlaps physical input panels")
        return result

    def panelize_inputs(self, runtime_inputs: dict[str, bytes]) -> dict[str, bytes]:
        sources = {source for source, _, _, _ in self.panels}
        if set(runtime_inputs) != sources:
            raise ValueError("ordered-K runtime inputs differ from the source signature")
        rows = {node.id: node.type.shape[0] for node in self.source.nodes if node.op == "input"}
        result: dict[str, bytes] = {}
        for source, panel, start, stop in self.panels:
            payload = runtime_inputs[source]
            if type(payload) is not bytes or len(payload) != rows[source] * self.reduction_size:
                raise ValueError(f"ordered-K runtime input {source} has wrong type or byte length")
            result[panel] = b"".join(
                payload[row * self.reduction_size + start:row * self.reduction_size + stop]
                for row in range(rows[source])
            )
        return result


def lower_ordered_k_contraction(
    request: KernelRequest, contract: Mxu0Contract
) -> OrderedKTilePlan:
    """Partition one source contraction without changing product order.

    The source boundary is row-major FP8. Each panel is materialized by a
    checked host launch copy; this is explicitly non-computational packaging.
    Tails, initial accumulators, and other source operations are refused.
    """
    geometry = contract.record["geometry"]
    policy = contract.record["numerical_policy"]
    rows, cols = geometry["array_rows"], geometry["array_cols"]
    tile_size = geometry["array_rows"]
    if request.target_identity != contract.identity or request.lowering_policy != "strict-native":
        raise ValueError("ordered-K source differs from selected strict-native contract")
    if len(request.nodes) != 3 or len(request.outputs) != 1 or request.constants:
        raise ValueError("ordered-K source needs two inputs and one contraction")
    activation, weight, result = request.nodes
    if (activation.op, activation.effect, weight.op, weight.effect,
            result.op, result.effect) != ("input", "input", "input", "input", "contraction", "pure"):
        raise ValueError("ordered-K source has unadmitted operation or effect")
    if result.inputs != (activation.id, weight.id) or request.outputs != (result.id,) or (
        request.output_storages != ("external_bf16",)
    ) or dict(request.input_storages) != {
        activation.id: "external_fp8_row_major", weight.id: "external_fp8_row_major",
    }:
        raise ValueError("ordered-K source has unsupported graph or boundary representation")
    if len(activation.type.shape) != 2 or len(weight.type.shape) != 2:
        raise ValueError("ordered-K source operands must be rank two")
    reduction_size = activation.type.shape[1]
    if (
        activation.type != TensorType((rows, reduction_size), policy["operand_dtype"],
                                      policy["operand_policy"])
        or weight.type != TensorType((cols, reduction_size), policy["operand_dtype"],
                                    policy["operand_policy"])
        or result.type != TensorType((rows, cols), policy["result_dtype"],
                                    policy["result_policy"])
        or reduction_size <= tile_size or reduction_size % tile_size
        or result.attrs != (("reduction_order", "k_ascending"),)
    ):
        raise ValueError("ordered-K source shape or numerical policy is outside complete-tile scope")
    reset = next(
        descriptor for descriptor in mxu0_profile(contract).descriptors
        if descriptor.name == "mxu0_matmul_reset"
    )
    if result.index_maps != reset.index_maps:
        raise ValueError("ordered-K source index maps differ from selected weight orientation")
    panel_nodes: list[SemanticNode] = []
    panels: list[tuple[str, str, int, int]] = []
    input_storages: list[tuple[str, str]] = []
    for tile_index, start in enumerate(range(0, reduction_size, tile_size)):
        for source in (activation, weight):
            panel = f"{source.id}__k{tile_index}"
            panels.append((source.id, panel, start, start + tile_size))
            panel_nodes.append(SemanticNode(
                panel, "input", (),
                TensorType((source.type.shape[0], tile_size), source.type.dtype,
                           source.type.numerical_policy),
                effect="input",
            ))
            input_storages.append((panel, "external_fp8"))
    if len({node.id for node in panel_nodes}) != len(panel_nodes) or (
        result.id in {node.id for node in panel_nodes}
    ):
        raise ValueError("ordered-K source names collide with generated panels")
    previous: str | None = None
    for tile_index in range(reduction_size // tile_size):
        final = tile_index == reduction_size // tile_size - 1
        node_id = result.id if final else f"{result.id}__k{tile_index}"
        if node_id in {node.id for node in panel_nodes}:
            raise ValueError("ordered-K result name collides with generated panel")
        operands = (f"{activation.id}__k{tile_index}", f"{weight.id}__k{tile_index}")
        if previous is not None:
            operands += (previous,)
        panel_nodes.append(SemanticNode(
            node_id, "contraction" if previous is None else "contraction_accumulate",
            operands, result.type, attrs=result.attrs,
            index_maps=reset.index_maps if previous is None else (
                *reset.index_maps[:2], reset.index_maps[2], reset.index_maps[2]
            ),
        ))
        previous = node_id
    lowered = KernelRequest(
        tuple(panel_nodes), (result.id,), ("external_bf16",),
        tuple(input_storages), contract.identity,
        lowering_policy="strict-native",
        source_identity=f"atlas-ordered-k-v1:{request.digest()}",
    )
    return OrderedKTilePlan(request, lowered, tuple(panels), reduction_size, tile_size)
