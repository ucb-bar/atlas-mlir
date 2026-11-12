"""Validate adapter correspondence and replace generic calls with Atlas semantics.

The manifest joins stamped source identities, never call positions or symbol
spellings. No tensor data is read, converted, or synthesized by this pass.
"""
from __future__ import annotations

import hashlib
import json
from math import prod

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, TensorType, i32
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.ir import BlockArgument, OpResult
from xdsl.rewriter import Rewriter

from .source_ops import PVMatmulOp, QKMatmulOp, QLinearOp, QuantizeActOp
from .source_translation import translate_activation_quantizer, translate_prequantized_contraction


class AtlasNormalizeError(ValueError):
    """An opaque call differs from its validated native storage contract."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _ids(attrs):
    result = set()
    for key in ("prov.source_node_ids", "prov.origin_node_ids"):
        value = attrs.get(key)
        if value is not None:
            if not isinstance(value, ArrayAttr) or any(not isinstance(v, StringAttr) for v in value):
                raise AtlasNormalizeError(f"malformed {key}")
            result.update(v.data for v in value)
    return result


def _abi(value, record):
    typ = value.type
    shape = record.get("shape")
    dtype = {"uint8": "i8", "int32": "i32", "bfloat16": "bf16",
             "float32": "f32", "float16": "f16", "float64": "f64"}.get(record.get("dtype"))
    if (not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(typ, TensorType) or dtype is None or str(typ.element_type) != dtype
            or list(typ.get_shape()) != shape):
        raise AtlasNormalizeError("ordered tensor shape or dtype differs from manifest")


def _operand_attrs(value):
    if isinstance(value, OpResult):
        return value.owner.attributes, value.index
    if isinstance(value, BlockArgument):
        owner = value.owner.parent_op()
        if not isinstance(owner, FuncOp) or owner.arg_attrs is None:
            raise AtlasNormalizeError("function argument has no source identity metadata")
        return owner.arg_attrs.data[value.index].data, 0
    raise AtlasNormalizeError("operand has no explicit source binding")


def _check_buffers(extension):
    buffers, source = extension.get("buffers"), extension.get("source_tensors")
    if not isinstance(buffers, dict) or not isinstance(source, dict):
        raise AtlasNormalizeError("Atlas manifest requires immutable buffer and source inventories")
    sizes = {"uint8": 1, "int32": 4, "bfloat16": 2, "float32": 4, "float16": 2, "float64": 8,
             "int64": 8, "bool": 1}
    for target, binding in buffers.items():
        if not isinstance(target, str) or not isinstance(binding, dict):
            raise AtlasNormalizeError("invalid lifted buffer inventory")
        authored = source.get(binding.get("source_name"))
        if not isinstance(authored, dict) or any(binding.get(k) != authored.get(k) for k in ("shape", "dtype", "bytes", "sha256")):
            raise AtlasNormalizeError("lifted buffer differs from immutable source inventory")
        shape, dtype, digest = binding.get("shape"), binding.get("dtype"), binding.get("sha256")
        if (not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape)
                or dtype not in sizes or type(binding.get("bytes")) is not int
                or binding.get("bytes") != prod(shape) * sizes[dtype]
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise AtlasNormalizeError("lifted buffer has invalid immutable byte metadata")
    return buffers


def _check_operands(op, call, buffers):
    records = call.get("operands")
    if not isinstance(records, list) or len(records) != len(op.operands):
        raise AtlasNormalizeError("ordered operand count differs from manifest")
    for value, record in zip(op.operands, records, strict=True):
        _abi(value, record)
        attrs, index = _operand_attrs(value)
        if type(record.get("result_index")) is not int:
            raise AtlasNormalizeError("ordered operand result index must be an integer")
        bound = record.get("node_id") in _ids(attrs) and record.get("result_index") == index
        aliases = attrs.get("prov.value_aliases")
        if aliases is not None:
            try:
                if not isinstance(aliases, StringAttr):
                    raise ValueError("expected JSON StringAttr")
                aliases = json.loads(aliases.data)
                if not isinstance(aliases, list):
                    raise ValueError("expected alias list")
                matches = []
                for alias in aliases:
                    if (not isinstance(alias, dict) or not isinstance(alias.get("node_ids"), list)
                            or any(not isinstance(key, str) for key in alias["node_ids"])
                            or type(alias.get("result_index")) is not int
                            or type(alias.get("alias_result_index")) is not int):
                        raise ValueError("invalid value alias")
                    if record.get("node_id") in alias["node_ids"]:
                        matches.append((alias["result_index"], alias["alias_result_index"]))
                if len(set(matches)) > 1:
                    raise ValueError("ambiguous value alias")
                bound |= (index, record.get("result_index")) in matches
            except (ValueError, TypeError) as exc:
                raise AtlasNormalizeError("invalid or ambiguous operand value alias metadata") from exc
        if not bound:
            raise AtlasNormalizeError(f"ordered operand source identity or result index differs for {record.get('role')}: "
                                      f"expected {record.get('node_id')}#{record.get('result_index')}, "
                                      f"observed {sorted(_ids(attrs))}#{index}")
        actual = attrs.get("prov.get_attr")
        buffer = record.get("buffer")
        if (actual.data if isinstance(actual, StringAttr) else None) != buffer:
            raise AtlasNormalizeError("lifted buffer binding differs from manifest")
        if buffer is not None:
            binding = buffers.get(buffer)
            if not isinstance(binding, dict) or any(binding.get(k) != record.get(k) for k in ("shape", "dtype")):
                raise AtlasNormalizeError("lifted buffer inventory differs from operand ABI")
            digest = binding.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise AtlasNormalizeError("lifted buffer has no immutable byte digest")


def _typed(op, call, digest, buffers):
    if any(not key.startswith(("prov.", "m2m.literal.")) and key != "m2m.schema" for key in op.attributes):
        raise AtlasNormalizeError("opaque call contains unrecognized semantic metadata")
    target = call.get("target")
    classes = {"atlas.quantize_act.default": QuantizeActOp, "atlas.qlinear.default": QLinearOp,
               "atlas.qk_matmul.default": QKMatmulOp, "atlas.pv_matmul.default": PVMatmulOp}
    if target == "atlas.qconv2d.default":
        raise AtlasNormalizeError("atlas.qconv2d is unsupported: convolution lowering is not qualified")
    if target not in classes:
        raise AtlasNormalizeError(f"unsupported native Atlas operation: {target}")
    schema = op.attributes.get("m2m.schema")
    kind = target.split(".")[1]
    if (not isinstance(schema, StringAttr) or schema.data != call.get("schema")
            or not schema.data.startswith(f"atlas::{kind}(")):
        raise AtlasNormalizeError("opaque custom operation schema differs from manifest")
    literals = {}
    for key, value in op.attributes.items():
        if key.startswith("m2m.literal."):
            if not isinstance(value, StringAttr):
                raise AtlasNormalizeError("opaque literal must use canonical JSON")
            try:
                literals[key.removeprefix("m2m.literal.")] = json.loads(value.data)
            except ValueError as exc:
                raise AtlasNormalizeError("opaque literal is invalid JSON") from exc
    if _json(literals) != _json(call.get("literals")):
        raise AtlasNormalizeError("schema-named literals differ from manifest")
    _check_operands(op, call, buffers)
    results = call.get("results")
    if not isinstance(results, list) or len(results) != len(op.results):
        raise AtlasNormalizeError("ordered result count differs from manifest")
    for index, (value, record) in enumerate(zip(op.results, results, strict=True)):
        if type(record.get("index")) is not int or record.get("index") != index:
            raise AtlasNormalizeError("ordered result index differs from manifest")
        _abi(value, record)
    attrs = {}
    attrs.update({k: StringAttr(v) for k, v in {
        "code_dtype": "uint8", "exponent_dtype": "int32", "weight_format": "atlas_e4m3",
        "dequantization": "e4m3(code)*2^exponent"}.items()})
    roles = [record.get("role") for record in call["operands"]]
    operands = list(op.operands)
    if kind == "quantize_act":
        if roles != ["x"] or literals.get("code_dtype") != "uint8":
            raise AtlasNormalizeError("quantize_act requires x and logical uint8 codes")
        expected = {"mode", "reduce", "rule", "static_exp", "code_dtype", "tag", "rows"}
        if set(literals) != expected:
            raise AtlasNormalizeError("quantize_act literal roles differ")
        for key in ("mode", "reduce", "rule", "tag"):
            if not isinstance(literals[key], str):
                raise AtlasNormalizeError(f"quantize_act {key} must be a string")
            attrs[key] = StringAttr(literals[key])
        for key in ("rows", "static_exp"):
            if key == "static_exp" and literals[key] is None:
                continue
            if type(literals[key]) is not int or not -(1 << 31) <= literals[key] < (1 << 31):
                raise AtlasNormalizeError(f"quantize_act {key} must be signed i32")
            attrs[key] = IntegerAttr(literals[key], i32)
    else:
        names = {"qlinear": ["x_codes", "x_exp", "w_codes", "w_exp"],
                 "qk_matmul": ["q_codes", "q_exp", "k_codes", "k_exp"],
                 "pv_matmul": ["p_codes", "p_exp", "vt_codes", "v_exp"]}[kind]
        expected = {"out_dtype", "weight_format"} if kind == "qlinear" else ({"pop_shift"} if kind == "pv_matmul" else set())
        if kind == "qlinear" and literals.get("bias") is None and "bias" in literals:
            expected.add("bias")
        if set(literals) != expected or roles not in (names, names + ["bias"] if kind == "qlinear" else names):
            raise AtlasNormalizeError("contraction ordered operand or literal roles differ")
        if kind == "pv_matmul" and (type(literals.get("pop_shift")) is not int or literals["pop_shift"] != 0):
            raise AtlasNormalizeError("atlas.pv_matmul pop_shift must be zero for unfolded import")
        if kind == "qlinear":
            if literals.get("weight_format") != "atlas_e4m3" or (len(operands) == 4) != (literals.get("bias", False) is None):
                raise AtlasNormalizeError("qlinear requires authored weight format and explicit optional bias")
            operands = [*operands[:4], [] if len(operands) == 4 else [operands[4]]]
        attrs.update({k: StringAttr(v) for k, v in {
            "rhs_transposed": "true", "output_rescale": "2^(lhs_exp+rhs_exp)",
            "output_dtype": literals.get("out_dtype", "float32"), "bias_order": "after_output_cast"}.items()})
    # Bindings and schema literals are consumed above. Emit the same typed
    # source semantics as the original native importer, keeping the manifest
    # commitment once at module scope. Opaque dispatch markers and alias
    # certificates are intake metadata, not executable source attributes.
    hint = "quantize" if kind == "quantize_act" else "matmul"
    attrs["prov._pattern_hint"] = StringAttr(hint)
    attrs["prov.op"] = StringAttr(hint)
    if kind != "quantize_act":
        attrs["prov.family"] = StringAttr("contraction")
    for key in ("prov.source_node_ids", "prov.origin_node_ids", "prov.trace_role"):
        if key in op.attributes:
            attrs[key] = op.attributes[key]
    attrs["prov.aten"] = StringAttr(target)
    for key, value in op.attributes.items():
        if key.startswith("prov.") and key not in {
            "prov.source_node_ids", "prov.origin_node_ids", "prov.trace_role",
            "prov.region_id", "prov.dispatch_id", "prov.value_aliases",
            "prov._pattern_hint", "prov.op", "prov.family", "prov.aten",
        }:
            attrs[key] = value
    result = classes[target].build(operands=operands, result_types=[v.type for v in op.results], attributes=attrs)
    try:
        result.verify()
        translator = translate_activation_quantizer if kind == "quantize_act" else translate_prequantized_contraction
        translator(result, tuple(f"arg{i}" for i in range(len(op.operands))), tuple(f"result{i}" for i in range(len(op.results))))
    except Exception:
        result.drop_all_references()
        raise
    return result


def normalize(module, manifest):
    """Normalize exactly the native calls declared by an Atlas v2 manifest."""
    if (not isinstance(manifest, dict) or manifest.get("schema") != "m2m.quantization_manifest.v2"
            or manifest.get("adapter_id") != "atlas_fp8"):
        raise AtlasNormalizeError("Atlas normalization requires the selected v2 adapter manifest")
    extension = manifest.get("atlas")
    if (not isinstance(extension, dict) or extension.get("schema") != "atlas.m2m_adapter.v1"
            or extension.get("storage") != "uint8"):
        raise AtlasNormalizeError("Atlas manifest requires immutable uint8 storage metadata")
    records = extension.get("calls")
    if not isinstance(records, list):
        raise AtlasNormalizeError("Atlas manifest requires a native call census")
    if not records and extension.get("saved_native_program", {}).get("no_native_calls") is not True:
        raise AtlasNormalizeError("empty native census requires an authenticated saved boundary with no native calls")
    calls = {}
    sites = {site.get("site_id"): site for site in manifest.get("sites", [])}
    for record in records:
        key = record.get("node_id")
        if not isinstance(key, str) or not key or key in calls:
            raise AtlasNormalizeError("missing or duplicate manifest call source identity")
        site = sites.get(record.get("site_id"), {})
        if site.get("status") != "quantized" or site.get("execution_route") != "atlas.native_fp8":
            raise AtlasNormalizeError("native call requires an explicitly quantized native site")
        calls[key] = record
    digest = hashlib.sha256(_json(manifest).encode()).hexdigest()
    buffers = _check_buffers(extension)
    prior = module.attributes.get("prov.quantization_manifest_sha256")
    if prior is not None and prior != StringAttr(digest):
        raise AtlasNormalizeError("module manifest digest differs from selected manifest")
    replacements, consumed = [], set()
    try:
        for op in module.walk():
            if not isinstance(op, CallOp):
                continue
            matches = _ids(op.attributes) & calls.keys()
            if len(matches) != 1:
                raise AtlasNormalizeError("opaque call has missing or ambiguous manifest source identity")
            key = next(iter(matches))
            if key in consumed:
                raise AtlasNormalizeError("manifest source identity matches duplicate opaque calls")
            consumed.add(key)
            replacements.append((op, _typed(op, calls[key], digest, buffers)))
        if consumed != calls.keys():
            raise AtlasNormalizeError("manifest contains unmatched native calls")
    except Exception:
        for _, new in replacements:
            new.drop_all_references()
        raise
    scheme = module.attributes.get("prov.quantization")
    if scheme is not None and scheme != StringAttr("external:atlas_fp8"):
        for _, new in replacements:
            new.drop_all_references()
        raise AtlasNormalizeError("module quantization scheme differs from selected adapter")
    # Validate the entire correspondence before mutating the caller's module.
    for old, new in replacements:
        Rewriter.replace_op(old, new)
    used = {op.callee.root_reference.data for op in module.walk() if isinstance(op, CallOp)}
    for op in list(module.body.block.ops):
        if isinstance(op, FuncOp) and op.is_declaration and op.sym_name.data not in used:
            Rewriter.erase_op(op)
    # These certificates have served their exact source joins. Their persistent
    # receipt remains in capture-trace.json and the digest-bound manifest.
    for op in module.walk():
        op.attributes.pop("prov.value_aliases", None)
        if isinstance(op, FuncOp):
            op.properties.pop("arg_attrs", None)
    module.attributes["prov.quantization_manifest_sha256"] = StringAttr(digest)
    module.verify()
    return module
