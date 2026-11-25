"""Preserve authored FP8 code and signed-exponent tensors at source intake.

Providers register their source operation names explicitly. Translation preserves
authored bytes and dtypes; closed diagnostic intake additionally validates known
Atlas values through the generic decoder. Nothing encodes or calibrates weights;
device arithmetic and packing remain provider responsibilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np

from merlin.compiler.semantic.linalg.linalg_types import _type
from merlin.compiler.semantic.model import KernelRequest, SemanticNode
from merlin.compiler.semantic.linalg.registered_source import operation_name
from merlin.compiler.quantization.contraction.scaled_fp8 import ContractionNumerics, require_same_numerics

from .fp8_contract import OPS, admit_rank2_nt, atlas_numerics

_CONTRACTION = "prequantized_contraction"
_QUANTIZER = "static_activation_quantizer"
NUMERICS_ATTRIBUTE = "numerical.contraction"


def contraction_numerics_attribute(op_name, output_dtype, *, bias=False):
    """Diagnostic candidate retained alongside original source metadata.

    Normalization preserves this declaration for later policy admission; it
    does not establish source arithmetic equivalence or select compilation.
    """
    dtype = {"bfloat16": "bf16", "float32": "f32"}.get(output_dtype)
    return json.dumps(atlas_numerics(op_name, output_dtype=dtype, bias=bias).record(),
                      sort_keys=True, separators=(",", ":"))


def _attributes(op) -> tuple:
    from xdsl.dialects.builtin import IntegerAttr, StringAttr

    if op.properties or op.regions or op.successors:
        raise ValueError("prequantized source operation must have only scalar attributes")
    result = []
    for key, value in op.attributes.items():
        if key == "op_name__":  # xDSL's unregistered-operation spelling, not semantics.
            continue
        if isinstance(value, StringAttr):
            scalar = value.data
        elif isinstance(value, IntegerAttr):
            scalar = value.value.data
        elif key.startswith("prov."):
            # SemanticNode attributes are JSON scalars. Keep the complete MLIR
            # spelling of structured correspondence rather than dropping it.
            scalar = str(value)
        else:
            raise ValueError(f"prequantized source attribute {key!r} is not a string or integer")
        if key == "semantic_kind":
            raise ValueError("source cannot override the registered semantic kind")
        result.append((key, scalar))
    return tuple(sorted(result))


def translate_prequantized_contraction(op, operand_ids, result_ids):
    """Source translator callback for an explicitly registered NT contraction."""
    if len(operand_ids) not in {4, 5} or len(result_ids) != 1:
        raise ValueError("prequantized contraction needs code/exponent pairs and an optional bias")
    name = operation_name(op)
    attrs = dict(_attributes(op))
    if name in OPS:
        declared = contraction_numerics_attribute(name, attrs.get("output_dtype"), bias=len(operand_ids) == 5)
        if attrs.get(NUMERICS_ATTRIBUTE, declared) != declared:
            raise ValueError("Atlas diagnostic route changes declared contraction numerical semantics")
        attrs[NUMERICS_ATTRIBUTE] = declared
    nodes = (SemanticNode(result_ids[0], name, operand_ids, _type(op.results[0].type),
                          attrs=(*sorted(attrs.items()), ("semantic_kind", _CONTRACTION))),)
    _check_contraction(nodes[0], tuple(_type(value.type) for value in op.operands))
    return nodes


def translate_activation_quantizer(op, operand_ids, result_ids):
    """Preserve authored static or dynamic quantizers without calculating scales."""
    if len(operand_ids) != 1 or len(result_ids) != 2:
        raise ValueError("activation quantizer needs one source and ordered code/exponent results")
    source_attrs = _attributes(op)
    a = dict(source_attrs)
    allowed = {"mode", "reduce", "rule", "static_exp", "code_dtype", "rows", "tag",
               "exponent_dtype", "weight_format", "dequantization"}
    if (any(key not in allowed and not key.startswith("prov.") for key in a)
            or a.get("exponent_dtype", "int32") != "int32"
            or a.get("dequantization", "e4m3(code)*2^exponent") != "e4m3(code)*2^exponent"):
        raise ValueError("activation quantizer has unknown or changed numeric metadata")
    mode, reduce = a.get("mode"), a.get("reduce")
    if (mode not in {"static", "dynamic"} or a.get("code_dtype") != "uint8"
            or a.get("rule") not in {"ref", "nc"}
            or reduce not in {"tensor", "token", "head", "block", "head_block"}
            or type(a.get("rows")) is not int or a["rows"] <= 0):
        raise ValueError("activation quantizer requires declared policy and logical uint8 codes")
    if mode == "static":
        if type(a.get("static_exp")) is not int or not -(1 << 31) <= a["static_exp"] < (1 << 31):
            raise ValueError("static quantizer requires fixed signed-i32 exponent")
        exponent_shape = ()
    else:
        if "static_exp" in a:
            raise ValueError("dynamic quantizer must preserve runtime exponent production")
        exponent_shape = None
    source, codes, exponent = (_type(value.type) for value in (*op.operands, *op.results))
    if not source.shape:
        raise ValueError("activation quantizer requires a non-scalar activation")
    if exponent_shape is None:
        if reduce == "tensor":
            exponent_shape = ()
        elif reduce == "head":
            if len(source.shape) < 2:
                raise ValueError("head quantizer requires at least two activation dimensions")
            exponent_shape = (*source.shape[:-2], 1, 1)
        else:
            exponent_shape = (*source.shape[:-1], 1)
    if (source.dtype not in {"bf16", "f32"} or codes.dtype != "i8" or source.shape != codes.shape
            or exponent.dtype != "i32" or exponent.shape != exponent_shape):
        raise ValueError("activation quantizer source, code, or signed exponent shape differs")
    kind = _QUANTIZER if mode == "static" else "dynamic_activation_quantizer"
    attrs = (*source_attrs, ("semantic_kind", kind))
    return tuple(SemanticNode(key, operation_name(op) + suffix, operand_ids, type_, attrs=attrs)
                 for key, suffix, type_ in zip(result_ids, (".codes", ".exponent"), (codes, exponent), strict=True))


# Existing provider registrations retain their source spelling and callback API.
translate_static_quantizer = translate_activation_quantizer


def _check_exponent(exp, code):
    if exp.dtype != "i32" or len(exp.shape) > len(code.shape):
        raise ValueError("prequantized exponent must be a signed-i32 broadcast tensor")
    aligned = (1,) * (len(code.shape) - len(exp.shape)) + exp.shape
    if any(a not in {1, b} for a, b in zip(aligned, code.shape)) or aligned[-1] != 1:
        raise ValueError("prequantized exponent cannot vary along the contraction K dimension")


def _check_contraction(node, types):
    a = dict(node.attrs)
    required = {
        "semantic_kind": _CONTRACTION, "code_dtype": "uint8", "exponent_dtype": "int32",
        "dequantization": "e4m3(code)*2^exponent", "rhs_transposed": "true",
        "output_rescale": "2^(lhs_exp+rhs_exp)", "bias_order": "after_output_cast",
    }
    allowed = {*required, "weight_format", "output_dtype", "pop_shift"}
    if node.op in OPS:
        allowed.add(NUMERICS_ATTRIBUTE)
        declared = contraction_numerics_attribute(node.op, a.get("output_dtype"), bias=len(types) == 5)
        if a.get(NUMERICS_ATTRIBUTE) != declared:
            raise ValueError("Atlas diagnostic route changes declared contraction numerical semantics")
    if "pop_shift" in a and (type(a["pop_shift"]) is not int or a["pop_shift"] != 0):
        raise ValueError("already-quantized intake requires unfolded zero pop_shift")
    if any(a.get(key) != value for key, value in required.items()) or any(
        key not in allowed and not key.startswith("prov.") for key in a
    ) or not isinstance(a.get("weight_format"), str) or not a["weight_format"]:
        raise ValueError("prequantized contraction metadata is missing, unknown, or changed")
    expected_dtype = {"bfloat16": "bf16", "float32": "f32"}.get(a.get("output_dtype"))
    if expected_dtype != node.type.dtype or node.type.numerical_policy != "raw-bit-pattern":
        raise ValueError("prequantized contraction output dtype differs from its metadata")
    lhs, lhs_exp, rhs, rhs_exp = types[:4]
    if lhs.dtype != "i8" or rhs.dtype != "i8" or len(lhs.shape) < 2 or len(rhs.shape) < 2 or lhs.shape[-1] != rhs.shape[-1]:
        raise ValueError("prequantized contraction needs raw byte NT operands with matching K")
    _check_exponent(lhs_exp, lhs)
    _check_exponent(rhs_exp, rhs)
    lp, rp = lhs.shape[:-2], rhs.shape[:-2]
    rank = max(len(lp), len(rp))
    lp, rp = (1,) * (rank - len(lp)) + lp, (1,) * (rank - len(rp)) + rp
    if any(x != y and x != 1 and y != 1 for x, y in zip(lp, rp)):
        raise ValueError("prequantized contraction batch dimensions do not broadcast")
    batch = tuple(max(x, y) for x, y in zip(lp, rp))
    m, n, k = lhs.shape[-2], rhs.shape[-2], lhs.shape[-1]
    if node.type.shape != (*batch, m, n):
        raise ValueError("prequantized contraction result shape differs from NT operands")
    if len(types) == 5 and (types[4].shape != (n,) or types[4].dtype != node.type.dtype):
        raise ValueError("prequantized contraction bias must match the output column and dtype")
    return m, n, k, batch


@dataclass(frozen=True)
class AlreadyQuantizedContraction:
    source: KernelRequest
    region_id: str
    kind: str
    operand_ids: tuple[str, str]
    exponent_ids: tuple[str, str]
    output_id: str
    attrs: tuple
    m: int
    n: int
    k: int
    batch_shape: tuple[int, ...]
    input_ids: tuple[str, ...]
    source_digest: str
    numerics: ContractionNumerics | None = None
    value_admission: str | None = None

    def record(self):
        result = {
            "schema": "merlin.already_quantized_contraction.v1",
            "source_request_digest": self.source_digest, "region_id": self.region_id,
            "kind": self.kind, "operand_ids": list(self.operand_ids), "exponent_ids": list(self.exponent_ids),
            "output_id": self.output_id, "attributes": dict(self.attrs), "input_ids": list(self.input_ids),
            "shape": {"m": self.m, "n": self.n, "k": self.k, "batch": list(self.batch_shape)},
            "constant_bindings": [{"node_id": item.node_id, "encoding": item.encoding,
                                   "sha256": sha256(bytes.fromhex(item.data_hex)).hexdigest()}
                                  for item in self.source.constants],
            "conversion": "none; preserve logical uint8 codes and signed int32 exponent data",
        }
        if self.numerics is not None:
            result["selected_numerics"] = self.numerics.record()
            result["numerical_authorization"] = (
                "explicit diagnostic lowering policy only; source arithmetic equivalence unproven")
            result["value_admission"] = self.value_admission
            result["runtime_obligations"] = [
                "validate runtime operand codes, signed exponents and exponent sums",
                "check finite normal or zero BF16 readout and restored result",
                "check device status before atomic output publication",
            ]
        return result

    def verify_source(self, source):
        if source.digest() != self.source_digest:
            raise ValueError("already-quantized intake source identity or constant bytes changed")


def lower_already_quantized_contraction(source: KernelRequest, *, candidate_numerics=None) -> AlreadyQuantizedContraction:
    """Check a closed source, admitting Atlas constants without rewriting bytes.

    Runtime inputs retain explicit validation obligations. This is the existing
    provider intake, not a compiler or an MXU arithmetic evaluator.
    """
    pure = tuple(node for node in source.nodes if node.effect == "pure")
    if len(pure) != 1 or source.lowering_policy != "diagnostic" or source.physical_extents or source.boundary_layouts:
        raise ValueError("already-quantized intake requires one diagnostic source contraction")
    node = pure[0]
    if (dict(node.attrs).get("semantic_kind") != _CONTRACTION or len(node.inputs) not in {4, 5}
            or source.outputs != (node.id,) or node.index_maps
            or {item.id for item in source.nodes} != {*node.inputs, node.id}):
        raise ValueError("already-quantized source computation or complete operand accounting differs")
    operands = tuple(source.node(key) for key in node.inputs)
    if any(item.effect not in {"input", "constant"} for item in operands):
        raise ValueError("already-quantized operands must be declared inputs or immutable constants")
    m, n, k, batch = _check_contraction(node, tuple(item.type for item in operands))
    numerics, value_admission = None, None
    if node.op in OPS:
        if len(node.inputs) != 4:
            raise ValueError("Atlas numerical value admission requires unbiased source operands")
        if operands[0].type.shape[:-2] != batch or operands[2].type.shape[:-2] not in (batch, ()):
            raise ValueError("Atlas numerical value admission requires identical batch prefixes or one shared weight")
        numerics = atlas_numerics(node.op, output_dtype=node.type.dtype)
        require_same_numerics(numerics, numerics if candidate_numerics is None else candidate_numerics)
        value_admission = _admit_atlas_constants(source, node, operands)
    elif candidate_numerics is not None:
        raise ValueError("Atlas numerical admission requires an Atlas source operation")
    return AlreadyQuantizedContraction(source, dict(node.attrs).get("prov.region_id", node.id), node.op,
                                       (node.inputs[0], node.inputs[2]), (node.inputs[1], node.inputs[3]),
                                       node.id, node.attrs, m, n, k, batch,
                                       tuple(item.id for item in source.nodes if item.effect in {"input", "constant"}),
                                       source.digest(), numerics, value_admission)


def _admit_atlas_constants(source, node, operands):
    """Validate every available constant, including partially bound exports."""
    values = {}
    bindings = {item.node_id: item for item in source.constants}
    for index, operand in enumerate(operands):
        binding = bindings.get(operand.id)
        if binding is None:
            continue
        dtype = np.int8 if index % 2 == 0 else np.dtype("<i4")
        value = np.frombuffer(bytes.fromhex(binding.data_hex), dtype=dtype).reshape(operand.type.shape)
        if index % 2 == 0:
            raw = value.view(np.uint8)
            if np.any(~((raw == 0) | ((raw >= 8) & (raw <= 126)) | ((raw >= 136) & (raw <= 254)))):
                raise ValueError("Atlas constant requires finite normal cap448 codes or positive zero")
        elif np.any((value < -127) | (value > 127)):
            raise ValueError("Atlas constant exponent leaves [-127,127]")
        values[index] = value
    # Each batch slice is one rank-two NT contraction; a shared weight serves every slice.
    # The narrow helper accepts these source layouts and makes their logical
    # exponent axes explicit before invoking Merlin's generic decoder.
    batch = node.type.shape[:-2]
    for exp, code in ((operands[1], operands[0]), (operands[3], operands[2])):
        if exp.type.shape not in ((), (*code.type.shape[:-1], 1), (*code.type.shape[:-2], 1, 1)):
            raise ValueError("Atlas numerical value admission requires scalar, per-row, or per-batch exponents")

    def per_slice(index, rows, columns):
        return np.broadcast_to(values[index], (*batch, rows, columns))

    m, n, k = operands[0].type.shape[-2], operands[2].type.shape[-2], operands[0].type.shape[-1]
    if 1 in values and 3 in values:
        lhs = per_slice(1, m, 1)[..., 0].astype(np.int64)
        rhs = per_slice(3, n, 1)[..., 0].astype(np.int64)
        sums = lhs[..., :, None] + rhs[..., None, :]
        if np.any((sums < -133) | (sums > 127)):
            raise ValueError("Atlas readout exponent sum leaves [-133,127]")
    if len(values) == 4:
        lhs_codes, lhs_exp = per_slice(0, m, k), per_slice(1, m, 1)
        rhs_codes, rhs_exp = per_slice(2, n, k), per_slice(3, n, 1)
        for index in np.ndindex(*batch):
            admit_rank2_nt(node.op, lhs_codes[index], lhs_exp[index], rhs_codes[index], rhs_exp[index])
        return "all-constant-operands-checked"
    return "available-constants-checked; runtime-input-validation-required"
