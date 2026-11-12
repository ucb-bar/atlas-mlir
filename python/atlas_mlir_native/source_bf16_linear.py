"""Admit a closed BF16 widening/f32 contraction/final BF16 rounding region.

The original typed source stays intact. The explicit policy selects ascending-K
binary32 accumulation because the source reduction order is unresolved.
"""

from dataclasses import dataclass
from math import prod

from merlin.semantic_compiler.model import IndexMap, TensorType, pointwise_identity_maps


POLICY = "bf16-exact-product-f32-ltr-rne-bf16-rne-v1"


def _attrs(node):
    return {key: value for key, value in node.attrs if not key.startswith("prov.")}


@dataclass(frozen=True)
class SourceBF16Linear:
    computations: tuple[str, ...]
    lhs: str
    rhs: str
    seed: str
    matmul: str
    output: str

    def receipt(self, source, *, original_source_sha256, application_source_digest):
        return {
            "schema": "atlas.source_bf16_f32acc_admission.v1",
            "original_source_sha256": original_source_sha256,
            "application_source_request_digest": application_source_digest,
            "source_region_request_digest": source.digest(),
            "source_form": "bf16-extf-f32-positive-zero-matmul-truncf-bf16",
            "source_computations": [source.node(key).record() for key in self.computations],
            "source_boundaries": [source.node(key).record() for key in (self.lhs, self.rhs, self.seed)],
            "source_reduction_order": "unresolved",
            "selected_policy": POLICY,
            "selected_reduction_order": "ascending K, left to right",
            "selected_arithmetic": {
                "operand_widening": "exact BF16 to binary32",
                "product": "binary32 RNE multiply; no BF16 product rounding",
                "accumulation": "binary32 RNE add after each product; no contraction",
                "initializer": "original positive-zero binary32 source seed",
                "output": "one final BF16 round to nearest ties to even",
            },
            "widened_intermediates_escape": False,
            "source_types_changed": False,
            "weight_codes_or_exponents_changed": False,
        }


def recognize_source_bf16_linear(source, output_id):
    """Prove exact shapes/maps/casts and all uses in the complete source DAG."""
    def require(condition, message):
        if not condition:
            raise ValueError("source BF16 f32 accumulation: " + message)

    require(not source.physical_extents and not source.boundary_layouts and not source.output_layouts,
            "requires dense logical source tensors")
    output = source.node(output_id)
    require(output.effect == "pure" and output.op == "f32_to_bf16" and len(output.inputs) == 1
            and _attrs(output) == {"numeric_semantics": "f32-to-bf16-rne-v1"}
            and output.index_maps == pointwise_identity_maps(2, 1), "final cast or indexing differs")
    matmul = source.node(output.inputs[0])
    attrs = _attrs(matmul)
    add_order = attrs.pop("scalar_add_order", "accumulator-product")
    require(matmul.effect == "pure" and matmul.op == "source_f32_matmul_unqualified"
            and len(matmul.inputs) == 3 and add_order in {"accumulator-product", "product-accumulator"}
            and attrs == {"reduction_order": "unresolved", "numeric_semantics": "source-f32-mul-add-unqualified-v1"},
            "source contraction arithmetic differs")
    left, right, seed = (source.node(key) for key in matmul.inputs)
    require(left.id != right.id, "requires two distinct source widenings")
    for widening in (left, right):
        require(widening.effect == "pure" and widening.op == "bf16_to_f32" and len(widening.inputs) == 1
                and _attrs(widening) == {"numeric_semantics": "bf16-to-f32-exact-v1"}
                and widening.index_maps == pointwise_identity_maps(2, 1), "operand widening or indexing differs")
        operand = source.node(widening.inputs[0])
        require(operand.type == TensorType(widening.type.shape, "bf16", "raw-bit-pattern")
                and widening.type == TensorType(widening.type.shape, "f32", "raw-bit-pattern")
                and len(operand.type.shape) == 2, "widening changes source shape or type")
    m, k = left.type.shape
    require(right.type.shape[0] == k, "contraction K dimensions differ")
    n = right.type.shape[1]
    require(matmul.type == TensorType((m, n), "f32", "raw-bit-pattern")
            and seed.type == matmul.type and output.type == TensorType((m, n), "bf16", "raw-bit-pattern"),
            "contraction seed or output type differs")
    expected_maps = tuple(IndexMap(3, rows, (0, 0)) for rows in (
        ((1, 0, 0), (0, 0, 1)), ((0, 0, 1), (0, 1, 0)),
        ((1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 1, 0))))
    require(matmul.index_maps == expected_maps, "contraction indexing differs")
    binding = next((item for item in source.constants if item.node_id == seed.id), None)
    require(seed.op == "constant" and seed.effect == "constant" and not seed.inputs
            and not seed.index_maps and not _attrs(seed) and binding is not None
            and binding.encoding == "f32-le-raw" and not any(bytes.fromhex(binding.data_hex))
            and len(binding.data_hex) == 8 * prod(seed.type.shape),
            "requires the original positive-zero binary32 constant seed")
    private = {left.id, right.id, matmul.id}
    computations = {*private, output.id}
    require(not private.intersection(source.outputs)
            and all(node.id in computations or not private.intersection(node.inputs) for node in source.nodes),
            "widened or accumulated binary32 intermediates escape")
    # A widening belongs only to this contraction, and the f32 result belongs
    # only to its final truncation. This also rejects extra internal uses.
    require(all({node.id for node in source.nodes if key in node.inputs} == {consumer}
                for key, consumer in ((left.id, matmul.id), (right.id, matmul.id), (matmul.id, output.id))),
            "widened or accumulated binary32 intermediates have additional uses")
    return SourceBF16Linear(tuple(node.id for node in source.nodes if node.id in computations),
                            left.inputs[0], right.inputs[0], seed.id, matmul.id, output.id)
