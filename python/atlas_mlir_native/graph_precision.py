"""Diagnostic lowering of parsed f32 tensor graphs to the selected Atlas tile.

This is an explicit precision change, not an equality rewrite. The input and
output conversions and every replacement are recorded for later quality checks.
Only the bounded, static operation subset below is admitted.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, replace

from merlin.semantic_compiler.linalg_bridge import (
    LinalgBridgeError,
    translate_linalg_text,
)
from merlin.semantic_compiler.model import (
    ConstantBinding,
    IndexMap,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.precision_values import (
    FloatBoundarySpec,
    contraction_restore_exponent,
)
from merlin.semantic_compiler.softmax_source import RowSoftmaxSource

from .binding import (
    BF16_ADD_ATTRS,
    BF16_EXP_ATTRS,
    BF16_MUL_ATTRS,
    BF16_RECIP_ATTRS,
    BF16_RELU_ATTRS,
    BF16_ROW_MAX_ATTRS,
    BF16_ROW_SUM_ATTRS,
    BF16_SUB_ATTRS,
    BF16_ZERO_BINARY_ATTRS,
    MXU0_INPUT_DOMAIN,
    MXU0_POLICY,
    ROUNDED_PACK_DOMAIN,
    ROUNDED_PACK_POLICY,
    SCALED_PACK_DOMAIN,
    SCALED_PACK_POLICY,
    SCALED_PACK_RANGE,
    TARGET_IDENTITY,
    _mxu0_maps,
    _row_reduce_maps,
    _transpose_maps,
    diagnostic_mxu0_f32_profile,
)

_SOURCE_MATMUL_MAPS = (
    IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
    IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
    IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
)
_SOURCE_ROW_MAPS = (
    IndexMap(2, ((1, 0), (0, 1)), (0, 0)),
    IndexMap(2, ((1, 0),), (0,)),
)
_SOURCE_BROADCAST_MAPS = (
    IndexMap(2, ((1, 0), (0, 1)), (0, 0)),
    IndexMap(2, ((1, 0), (0, 0)), (0, 0)),
    IndexMap(2, ((1, 0), (0, 1)), (0, 0)),
)
_TILE = (32, 32)
_BF16 = TensorType(_TILE, "bf16", "raw-bit-pattern")
_FP8 = TensorType(_TILE, "fp8_e4m3", "raw-bit-pattern")
_PACKED = TensorType(_TILE, "fp8_e4m3", SCALED_PACK_POLICY)
_ROUNDED_PACKED = TensorType(_TILE, "fp8_e4m3", ROUNDED_PACK_POLICY)
_MXU = TensorType(_TILE, "bf16", MXU0_POLICY)
# These older profiles use the same graph arithmetic and selected RTL.
# Keep their public precision documents and historical artifacts unchanged.
_LEGACY_GRAPH_TARGET = "atlas-rtl-0079c0541111197741a231c002e3843fa6f545b2:movement-mxu01-vpu-pack-tile-v21"
_LEGACY_GRAPH_TARGET_V22 = _LEGACY_GRAPH_TARGET.removesuffix("v21") + "v22"
_LEGACY_GRAPH_TARGET_V23 = _LEGACY_GRAPH_TARGET.removesuffix("v21") + "v23"


@dataclass(frozen=True)
class DiagnosticGraphPrecision:
    request: KernelRequest
    source_sha256: str
    policy_sha256: str
    source_to_target: tuple[tuple[str, str | None], ...]
    replacements: tuple[tuple[str, str], ...]
    boundary_conversions: tuple[dict[str, object], ...]
    source_operations: tuple[str, ...]
    source_region: RowSoftmaxSource | None = None
    source_kind: str | None = None

    def record(self) -> dict[str, object]:
        record = {
            "schema": "atlas.diagnostic_graph_precision_result.v1",
            "request_digest": self.request.digest(),
            "source_sha256": self.source_sha256,
            "policy_sha256": self.policy_sha256,
            "source_to_target": dict(self.source_to_target),
            "replacements": dict(self.replacements),
            "boundary_conversions": list(self.boundary_conversions),
            "source_operations": list(self.source_operations),
            "qualification": "diagnostic_precision_change; original_model_quality_unmeasured",
        }
        if self.source_region is not None:
            record["source_region"] = self.source_region.record()
        if self.source_kind is not None:
            record["source_kind"] = self.source_kind
        return record


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise LinalgBridgeError(reason)


def _bf16_rne(bits: int) -> int:
    exponent = (bits >> 23) & 0xff
    _require(exponent not in (0, 255), "diagnostic BF16 constant needs finite normal f32 elements")
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
    _require((rounded & 0x7f80) not in (0, 0x7f80), "diagnostic BF16 constant overflows its normal range")
    return rounded & 0xffff


def diagnostic_graph_input_spec(policy: dict[str, object]) -> FloatBoundarySpec:
    """Use the selected tile's executable f32-to-FP8 boundary for graph inputs."""
    base = diagnostic_mxu0_f32_profile().conversion_specs
    conversion = policy.get("input_conversion")
    if base is None or conversion not in {
        "f32-to-e4m3-rne-finite-v1", "f32-to-e4m3-rne-finite-positive-zero-v1",
    }:
        raise LinalgBridgeError("selected graph input conversion is unsupported")
    exponent = policy.get("input_scale_exponent", 0)
    if type(exponent) is not int or not -32 <= exponent <= 32:
        raise LinalgBridgeError("graph input scale exponent must be an integer in [-32, 32]")
    if exponent and policy.get("schema") != "atlas.diagnostic_graph_precision.v3":
        raise LinalgBridgeError("graph input scaling requires the scaled precision policy")
    return replace(base[0], policy=str(conversion), pre_encode_scale_exponent=exponent,
                   canonicalize_negative_zero=conversion == "f32-to-e4m3-rne-finite-positive-zero-v1")


def lower_diagnostic_graph(source: str, *, entry: str, policy: dict[str, object]) -> DiagnosticGraphPrecision:
    """Lower a fully parsed, static f32 graph using one declared Atlas policy.

    The policy is deliberately independent of model names and source node IDs.
    A caller must qualify input domains and original-model quality separately.
    """
    scale_code = policy.get("pack_scale_code") if type(policy) is dict else None
    schema = policy.get("schema") if type(policy) is dict else None
    _require(isinstance(policy, dict) and policy.get("target_identity") in {
        TARGET_IDENTITY, _LEGACY_GRAPH_TARGET, _LEGACY_GRAPH_TARGET_V22, _LEGACY_GRAPH_TARGET_V23,
    },
             "graph precision policy selects an unqualified target identity")
    expected = {
        "schema": schema,
        "target_identity": policy.get("target_identity") if isinstance(policy, dict) else None,
        "input_conversion": policy.get("input_conversion") if type(policy) is dict else None,
        "output_conversion": "bf16-to-f32-exact-v1",
        "constant_conversion": "f32-to-bf16-rne-v1",
        "interlayer_pack": "scaled-e8m0-exact-normal-v1",
        "pack_scale_code": scale_code,
    }
    pack_policy = policy.get("interlayer_pack") if type(policy) is dict else None
    if schema in {"atlas.diagnostic_graph_precision.v2", "atlas.diagnostic_graph_precision.v3"}:
        expected["arithmetic"] = "finite-normal-or-signed-zero-v1"
    if schema == "atlas.diagnostic_graph_precision.v3":
        expected["input_scale_exponent"] = policy.get("input_scale_exponent")
        expected["contraction_rescale"] = "restore-operand-scales-bf16-v1"
        if "pack_pre_scale_exponent" in policy:
            expected["pack_pre_scale_exponent"] = policy["pack_pre_scale_exponent"]
    pre_scale = policy.get("pack_pre_scale_exponent", 0) if type(policy) is dict else 0
    _require(type(pre_scale) is int and -16 <= pre_scale <= 16,
             "pack pre-scale exponent must be an integer in [-16, 16]")
    _require(
        schema in {"atlas.diagnostic_graph_precision.v1", "atlas.diagnostic_graph_precision.v2",
                   "atlas.diagnostic_graph_precision.v3"}
        and type(scale_code) is int and scale_code in range(SCALED_PACK_RANGE[0], SCALED_PACK_RANGE[1] + 1)
        and type(pack_policy) is str
        and pack_policy in {"scaled-e8m0-exact-normal-v1", "scaled-e8m0-rtl-rne-v1"}
        and policy == {**expected, "interlayer_pack": pack_policy},
        "selected graph precision policy is unsupported",
    )
    input_spec = diagnostic_graph_input_spec(policy)
    restore_scales = schema == "atlas.diagnostic_graph_precision.v3"
    zero_arithmetic = schema != "atlas.diagnostic_graph_precision.v1"

    def binary(kind: str):
        op = kind + "_bf16" + ("_zero" if zero_arithmetic else "")
        attrs = BF16_ZERO_BINARY_ATTRS[op] if zero_arithmetic else {
            "add": BF16_ADD_ATTRS, "sub": BF16_SUB_ATTRS, "mul": BF16_MUL_ATTRS,
        }[kind]
        return op, attrs

    multiply, mul_attrs = binary("mul")
    packed_type, pack_domain = (
        (_PACKED, SCALED_PACK_DOMAIN)
        if pack_policy == "scaled-e8m0-exact-normal-v1"
        else (_ROUNDED_PACKED, ROUNDED_PACK_DOMAIN)
    )
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    policy_bytes = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    translated = translate_linalg_text(
        source, entry=entry, target_identity=TARGET_IDENTITY, lowering_policy="diagnostic",
    )
    source_nodes = {node.id: node for node in translated.request.nodes}
    source_constants = {binding.node_id: binding for binding in translated.request.constants}
    consumers: dict[str, list[tuple[SemanticNode, int]]] = {name: [] for name in source_nodes}
    for consumer in translated.request.nodes:
        for index, operand in enumerate(consumer.inputs):
            consumers[operand].append((consumer, index))
    target_nodes: list[SemanticNode] = []
    target_constants: list[ConstantBinding] = []
    mapping: dict[str, str | None] = {}
    replacements: list[tuple[str, str]] = []
    packs: dict[str, str] = {}
    used_constants: set[str] = set()
    encoded_exponents: dict[str, int] = {}
    correction_constants: dict[tuple[int, tuple[int, ...]], str] = {}

    def mapped(source_id: str) -> str:
        value = mapping.get(source_id)
        _require(value is not None, f"source value {source_id} has no Atlas representation")
        return value

    def append(node: SemanticNode) -> str:
        target_nodes.append(node)
        return node.id

    def scale_constant(exponent: int, shape: tuple[int, ...] = _TILE) -> str:
        from math import prod

        key = (exponent, shape)
        if key not in correction_constants:
            name = f"precision_scale_{'neg' if exponent < 0 else 'pos'}_{abs(exponent)}"
            if shape != _TILE:
                name += "_" + "x".join(map(str, shape))
            bits = struct.unpack("<I", struct.pack("<f", 2.0 ** exponent))[0]
            data = struct.pack("<H", _bf16_rne(bits)) * prod(shape)
            correction_constants[key] = append(SemanticNode(
                name, "constant", (), replace(_BF16, shape=shape), attrs=(mul_attrs[0],), effect="constant",
            ))
            target_constants.append(ConstantBinding(name, "bf16-le-raw", data.hex()))
        return correction_constants[key]

    def pack(source_id: str) -> str:
        value = mapped(source_id)
        producer = next(node for node in target_nodes if node.id == value)
        if producer.type == replace(_FP8, shape=producer.type.shape) or producer.type == replace(packed_type, shape=producer.type.shape):
            return value
        _require(producer.type in (replace(_BF16, shape=producer.type.shape), replace(_MXU, shape=producer.type.shape)),
                 "matmul operand needs FP8 or admitted BF16 producer")
        if value not in packs:
            name = f"{value}__fp8"
            operand = value
            if pre_scale:
                constant = scale_constant(pre_scale, producer.type.shape)
                operand = append(SemanticNode(
                    f"{value}__pack_scaled", multiply, (value, constant), replace(_BF16, shape=producer.type.shape),
                    attrs=mul_attrs, index_maps=pointwise_identity_maps(2, 2),
                ))
                replacements.append((operand, f"BF16 pre-pack scale 2**({pre_scale})"))
            append(SemanticNode(
                name, "pack_fp8", (operand,), replace(packed_type, shape=producer.type.shape),
                attrs=(("input_domain", pack_domain), ("scale_code", scale_code)),
                index_maps=pointwise_identity_maps(2, 1),
            ))
            packs[value] = name
            encoded_exponents[name] = 127 - scale_code + pre_scale
        return packs[value]

    def check_zero(source_id: str, shape: tuple[int, ...]) -> None:
        node = source_nodes[source_id]
        binding = source_constants[source_id]
        _require(node.op == "constant" and node.type.shape == shape
                 and binding.encoding == "f32-le-raw" and set(bytes.fromhex(binding.data_hex)) == {0},
                 "source reduction or contraction needs a positive-zero initializer")
        used_constants.add(source_id)
        replacements.append((source_id, "positive-zero initializer folded into selected operation"))

    for node in translated.request.nodes:
        _require(node.type.dtype == "f32", f"source value {node.id} has an unsupported dtype")
        if node.op == "input":
            _require(len(node.type.shape) == 2, "diagnostic graph needs static matrix inputs")
            mapping[node.id] = append(SemanticNode(
                node.id, "input", (), replace(_FP8, shape=node.type.shape),
                attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input",
            ))
            encoded_exponents[node.id] = input_spec.pre_encode_scale_exponent
        elif node.op == "constant":
            mapping[node.id] = None  # Initializers and literals are handled at their use.
        elif node.op == "transpose":
            _require(len(node.type.shape) == 2 and len(node.inputs) == 1
                     and node.type.shape == tuple(reversed(source_nodes[node.inputs[0]].type.shape))
                     and node.index_maps == _transpose_maps()
                     and node.id not in translated.request.outputs
                     and all(consumer.op == "source_f32_matmul_unqualified" and index == 1
                             for consumer, index in consumers[node.id]),
                     "source transpose needs an exclusively MXU-weight use")
            mapping[node.id] = mapped(node.inputs[0])
            replacements.append((node.id, "physical MXU weight orientation"))
        elif node.op == "source_f32_matmul_unqualified":
            _require(len(node.type.shape) == 2 and len(node.inputs) == 3
                     and node.index_maps == (*_SOURCE_MATMUL_MAPS, _SOURCE_MATMUL_MAPS[-1])
                     and dict(node.attrs) == {
                         "reduction_order": "unresolved",
                         "numeric_semantics": "source-f32-mul-add-unqualified-v1",
                     }, "unsupported source matmul semantics or maps")
            check_zero(node.inputs[2], node.type.shape)
            lhs_source, rhs_source = node.inputs[:2]
            _require(source_nodes[lhs_source].op != "transpose", "transposed lhs needs a new layout route")
            lhs = pack(lhs_source)
            rhs_node = source_nodes[rhs_source]
            if rhs_node.op == "transpose":
                rhs = pack(rhs_node.inputs[0])
            else:
                rhs = pack(rhs_source)
                _require(rhs != mapped(rhs_source), "raw FP8 rhs transpose needs a physical route")
                rhs = append(SemanticNode(
                    f"{rhs}__transpose", "transpose", (rhs,), packed_type,
                    index_maps=_transpose_maps(),
                ))
                encoded_exponents[rhs] = encoded_exponents[pack(rhs_source)]
            exponent = contraction_restore_exponent(encoded_exponents[lhs], encoded_exponents[rhs])
            physical = f"{node.id}__encoded_dot" if restore_scales and exponent else node.id
            mapping[node.id] = append(SemanticNode(
                physical, "matmul", (lhs, rhs), replace(_MXU, shape=node.type.shape), index_maps=_mxu0_maps(),
            ))
            if restore_scales and exponent:
                # A constant power-of-two multiply restores the logical BF16
                # units before any consumer, including softmax or an output.
                constant = scale_constant(exponent, node.type.shape)
                mapping[node.id] = append(SemanticNode(
                    node.id, multiply, (physical, constant), replace(_BF16, shape=node.type.shape),
                    attrs=mul_attrs, index_maps=pointwise_identity_maps(2, 2),
                ))
                replacements.append((physical, f"restore dot scale with BF16 2**({exponent}) multiply"))
            mapping[node.inputs[2]] = None
            replacements.append((node.id, "f32 matmul -> ordered FP8/BF16 MXU0 reset"))
        elif node.op == "f32_mul":
            _require(node.type.shape == _TILE and node.index_maps == pointwise_identity_maps(2, 2)
                     and dict(node.attrs) == {"numeric_semantics": "ieee754-f32-mul-rne-v1"},
                     "unsupported source multiply")
            operands: list[str] = []
            for source_id in node.inputs:
                if source_nodes[source_id].op != "constant":
                    operands.append(mapped(source_id))
                    continue
                binding = source_constants[source_id]
                _require(binding.encoding == "f32-le-raw" and source_nodes[source_id].type.shape == _TILE,
                         "multiply literal must be a declared f32 tile")
                bits = struct.unpack(f"<{len(binding.data_hex) // 8}I", bytes.fromhex(binding.data_hex))
                data = b"".join(struct.pack("<H", _bf16_rne(bit)) for bit in bits)
                mapping[source_id] = append(SemanticNode(
                    source_id, "constant", (), _BF16, attrs=(mul_attrs[0],), effect="constant",
                ))
                target_constants.append(ConstantBinding(source_id, "bf16-le-raw", data.hex()))
                used_constants.add(source_id)
                operands.append(source_id)
                replacements.append((source_id, "f32 literal -> BF16 RNE constant tile"))
            _require(len(operands) == 2 and operands[0] != operands[1], "multiply needs two distinct operands")
            mapping[node.id] = append(SemanticNode(
                node.id, multiply, tuple(operands), _BF16,
                attrs=mul_attrs, index_maps=pointwise_identity_maps(2, 2),
            ))
            replacements.append((node.id, "f32 multiply -> BF16 multiply"))
        elif node.op == "f32_add":
            _require(node.type.shape == _TILE and len(node.inputs) == 2
                     and node.inputs[0] != node.inputs[1]
                     and node.index_maps == pointwise_identity_maps(2, 2)
                     and dict(node.attrs) == {"numeric_semantics": "ieee754-f32-add-rne-v1"},
                     "unsupported source pointwise add")
            operands = tuple(mapped(source_id) for source_id in node.inputs)
            types = {value.id: value.type for value in target_nodes}
            _require(all(types.get(source) in {_BF16, _MXU} for source in operands),
                     "source add needs admitted BF16 producers")
            addition, add_attrs = binary("add")
            mapping[node.id] = append(SemanticNode(
                node.id, addition, operands, _BF16,
                attrs=add_attrs, index_maps=pointwise_identity_maps(2, 2),
            ))
            replacements.append((node.id, "f32 add -> BF16 add under exact-FP32-sum domain"))
        elif node.op in {"source_f32_reduce_max_unqualified", "source_f32_reduce_add_unqualified"}:
            _require(len(node.inputs) == 2 and node.type.shape == (32,)
                     and node.index_maps == (*_SOURCE_ROW_MAPS, _SOURCE_ROW_MAPS[-1])
                     and dict(node.attrs).get("axis") == 1,
                     "unsupported source row reduction")
            initializer = node.inputs[1]
            initial_node = source_nodes[initializer]
            binding = source_constants.get(initializer)
            _require(initial_node.op == "constant" and binding is not None
                     and initial_node.type.shape == (32,) and binding.encoding == "f32-le-raw",
                     "row reduction needs a declared f32 initializer")
            if node.op == "source_f32_reduce_max_unqualified":
                _require({bytes.fromhex(binding.data_hex)[i:i+4]
                          for i in range(0, 128, 4)} == {bytes.fromhex("000080ff")},
                         "row maximum needs negative-infinity initializer")
                used_constants.add(initializer)
                replacements.append((initializer, "negative-infinity row-max initializer folded"))
                operation, attrs = "row_max_bf16", BF16_ROW_MAX_ATTRS
                expected_semantics = "source-f32-maximum-unqualified-v1"
            else:
                check_zero(initializer, (32,))
                operation, attrs = "row_sum_bf16", BF16_ROW_SUM_ATTRS
                expected_semantics = "source-f32-add-unqualified-v1"
            _require(dict(node.attrs).get("numeric_semantics") == expected_semantics
                     and dict(node.attrs).get("reduction_order") == "unresolved"
                     and dict(node.attrs).get("scalar_operand_order",
                                              dict(node.attrs).get("scalar_add_order"))
                     == "input-accumulator",
                     "unsupported source reduction arithmetic")
            mapping[node.id] = append(SemanticNode(
                node.id, operation, (mapped(node.inputs[0]),), _BF16,
                attrs=attrs, index_maps=_row_reduce_maps(),
            ))
            replacements.append((node.id, f"f32 row reduction -> {operation} broadcast tile"))
        elif node.op == "reshape":
            _require(len(node.inputs) == 1 and node.type.shape == (32, 1)
                     and source_nodes[node.inputs[0]].type.shape == (32,)
                     and source_nodes[node.inputs[0]].op in {
                         "source_f32_reduce_max_unqualified", "source_f32_reduce_add_unqualified",
                     } and node.id not in translated.request.outputs
                     and all(consumer.op in {"f32_sub", "f32_div"} and index == 1
                             for consumer, index in consumers[node.id]),
                     "unsupported source reshape")
            mapping[node.id] = mapped(node.inputs[0])
            replacements.append((node.id, "row vector reshape -> broadcast tile view"))
        elif node.op == "f32_sub":
            _require(node.type.shape == _TILE and node.index_maps == _SOURCE_BROADCAST_MAPS
                     and source_nodes[node.inputs[1]].op == "reshape"
                     and dict(node.attrs) == {"numeric_semantics": "ieee754-f32-sub-rne-v1"},
                     "unsupported source broadcast subtraction")
            subtraction, sub_attrs = binary("sub")
            mapping[node.id] = append(SemanticNode(
                node.id, subtraction, tuple(mapped(source_id) for source_id in node.inputs), _BF16,
                attrs=sub_attrs, index_maps=pointwise_identity_maps(2, 2),
            ))
            replacements.append((node.id, "f32 subtraction -> BF16 subtraction"))
        elif node.op == "source_f32_exp_unqualified":
            _require(node.type.shape == _TILE and len(node.inputs) == 1
                     and node.index_maps == pointwise_identity_maps(2, 1)
                     and dict(node.attrs) == {
                         "numeric_semantics": "source-f32-exp-unqualified-v1",
                     },
                     "unsupported source exponential maps")
            mapping[node.id] = append(SemanticNode(
                node.id, "exp_bf16", (mapped(node.inputs[0]),), _BF16,
                attrs=BF16_EXP_ATTRS, index_maps=pointwise_identity_maps(2, 1),
            ))
            replacements.append((node.id, "f32 exp -> selected BF16 LUT exp"))
        elif node.op == "f32_div":
            _require(node.type.shape == _TILE and node.index_maps == _SOURCE_BROADCAST_MAPS
                     and source_nodes[node.inputs[1]].op == "reshape"
                     and dict(node.attrs) == {"numeric_semantics": "ieee754-f32-div-rne-v1"},
                     "unsupported source broadcast division")
            reciprocal = append(SemanticNode(
                f"{node.id}__reciprocal", "recip_bf16", (mapped(node.inputs[1]),), _BF16,
                attrs=BF16_RECIP_ATTRS, index_maps=pointwise_identity_maps(2, 1),
            ))
            mapping[node.id] = append(SemanticNode(
                node.id, multiply, (mapped(node.inputs[0]), reciprocal), _BF16,
                attrs=mul_attrs, index_maps=pointwise_identity_maps(2, 2),
            ))
            replacements.append((node.id, "f32 division -> BF16 reciprocal LUT and multiply"))
        elif node.op == "f32_relu_ordered_select":
            _require(len(node.type.shape) == 2 and len(node.inputs) == 1
                     and node.index_maps == pointwise_identity_maps(2, 1)
                     and dict(node.attrs) == {
                         "numeric_semantics": "f32-ordered-lt-positive-zero-select-v1",
                     }, "unsupported source ordered ReLU")
            mapping[node.id] = append(SemanticNode(
                node.id, "relu", (mapped(node.inputs[0]),), replace(_BF16, shape=node.type.shape),
                attrs=BF16_RELU_ATTRS, index_maps=pointwise_identity_maps(2, 1),
            ))
            # This is part of the selected precision transformation. The
            # source preserves -0; hardware sign clamp canonicalizes it.
            replacements.append((node.id, "f32 ordered zero select -> BF16 sign clamp; negative zero canonicalized"))
        elif node.op == "source_f32_composite_unqualified":
            _require(len(node.type.shape) == 2 and node.index_maps == pointwise_identity_maps(2, 1)
                     and dict(node.attrs).get("operation") == "clamp_lower_literal"
                     and dict(node.attrs).get("literal_bits") == "0x00000000"
                     and dict(node.attrs).get("numeric_semantics")
                     == "source-f32-composite-unqualified-v1",
                     "unsupported source composite")
            mapping[node.id] = append(SemanticNode(
                node.id, "relu", (mapped(node.inputs[0]),), replace(_BF16, shape=node.type.shape),
                attrs=BF16_RELU_ATTRS, index_maps=pointwise_identity_maps(2, 1),
            ))
            replacements.append((node.id, "f32 zero clamp -> BF16 sign clamp"))
        else:
            raise LinalgBridgeError(f"source node {node.id} has unsupported operation {node.op}")

    _require(set(mapping) == set(source_nodes), "precision lowering omitted a parsed source node")
    _require(used_constants == set(source_constants), "precision lowering discarded a source constant")
    outputs = tuple(mapped(value) for value in translated.request.outputs)
    _require(len(set(outputs)) == len(outputs), "precision lowering collapsed ordered outputs")
    request = KernelRequest(
        tuple(target_nodes), outputs, ("external",) * len(outputs),
        tuple((node.id, "external") for node in target_nodes if node.op == "input"),
        TARGET_IDENTITY, lowering_policy="diagnostic",
        source_identity=f"{source_sha}:{policy_sha}", constants=tuple(target_constants),
    )
    input_spec = diagnostic_graph_input_spec(policy)
    conversions = tuple(
        {"source": node.id, "source_dtype": "f32", "target_dtype": "fp8_e4m3",
         "policy": str(policy["input_conversion"]), "conversion_spec": input_spec.record()}
        for node in translated.request.nodes if node.op == "input"
    ) + tuple(
        {"source": source_id, "source_dtype": "f32", "target_dtype": "bf16",
         "policy": str(policy["output_conversion"])}
        for source_id in translated.request.outputs
    )
    return DiagnosticGraphPrecision(
        request, source_sha, policy_sha, tuple(mapping.items()), tuple(replacements),
        conversions, translated.source_operations,
    )
