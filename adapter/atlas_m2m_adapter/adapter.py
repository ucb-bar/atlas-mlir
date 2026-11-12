"""Capture authenticated, unfolded raw replay without transforming weight data.

The target loader owns architecture construction and attaches ``_atlas_export_dir``.
Optional ``_atlas_export_prefix`` is the exact wrapper prefix to remove from model
state names; ``_atlas_source_prefix`` prepends a source subtree for a layer boundary.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import json
import operator
from pathlib import Path

import torch

SCHEMA = "atlas.m2m_adapter.v1"
NATIVE_OPS = {"quantize_act", "qlinear", "qconv2d", "qk_matmul", "pv_matmul"}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(value):
    path = Path(value).absolute()
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError("export paths must not use symlinks")
    return path


def tensor_record(value):
    if value.device.type != "cpu":
        raise ValueError("immutable export intake requires CPU payloads")
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"dtype": str(value.dtype).removeprefix("torch."), "shape": list(value.shape),
            "bytes": len(raw), "sha256": _sha(raw)}


def _type(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError("native operator ABI must contain only tensor results")
    return {"dtype": str(value.dtype).removeprefix("torch."),
            "shape": [int(dim) for dim in value.shape]}


def _value_abi(value):
    """Compare claimed tensor metadata without materializing a fake tensor."""
    if isinstance(value, torch.Tensor):
        return ("tensor", str(value.dtype), tuple(str(dim) for dim in value.shape))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_value_abi(item) for item in value))
    return (type(value).__name__, str(value))


def _source_name(name, prefix, source_prefix):
    if prefix and not name.startswith(prefix):
        raise ValueError(f"state lies outside explicit wrapper prefix: {name}")
    return source_prefix + name[len(prefix):]


def _validate_policy(policy, contract, tensors):
    if policy.get("schema") != "atlas_torchao.execution_policy.v1":
        raise ValueError("unsupported export policy schema")
    if contract.get("schema") != "atlas_torchao.export_manifest.v1":
        raise ValueError("selected contract must be the original export manifest")
    if contract.get("format_version") != 1 or policy.get("format_version", 1) != 1:
        raise ValueError("unsupported export format version")
    for document in (policy, contract):
        if any(key in document for key in ("pack_fold", "pack_fold_summary", "pack_fold_applied")):
            raise ValueError("folded exports are not supported")
    sites = policy.get("sites")
    if not isinstance(sites, dict) or not sites:
        raise ValueError("export has no site coverage")
    declared = contract.get("sites")
    expected = [{"site_id": name, "status": site["status"], "kind": site["kind"]}
                for name, site in sites.items()]
    if not isinstance(declared, list) or sorted(declared, key=lambda row: row["site_id"]) != sorted(
            expected, key=lambda row: row["site_id"]):
        raise ValueError("export manifest and policy site coverage differ")
    if policy.get("tensors", {}).get("count") != len(tensors):
        raise ValueError("export tensor inventory count differs")
    for name, site in sites.items():
        if site.get("status") not in {"quantized", "host", "skipped"}:
            raise ValueError(f"unsupported policy status: {name}")
        if any(key in site for key in ("pack_fold", "pack_folds")) or site.get("pv_pop_shift", 0):
            raise ValueError(f"folded site is unsupported: {name}")
        rules = [site.get("activation", {})] + list(site.get("operands", {}).values())
        if any(rule.get("mode") == "static_per_step" for rule in rules):
            raise ValueError(f"mutable activation policy is unsupported: {name}")
        for rule in rules:
            if rule.get("mode") == "static":
                if type(rule.get("exp")) is not int:
                    raise ValueError(f"static activation exponent must be an integer: {name}")
                if "tensor" in rule:
                    stored = tensors[rule["tensor"]]
                    if stored.dtype != torch.int32 or stored.numel() != 1 or stored.item() != rule["exp"]:
                        raise ValueError(f"static activation tensor differs from policy: {name}")
        if site["status"] != "quantized" or site["kind"] not in {"linear", "conv2d"}:
            continue
        if site.get("residual_passes") or site["weight"].get("block", 0):
            raise ValueError(f"unsupported residual or blocked weight site: {name}")
        weight = site["weight"]
        codes, exponent = tensors[weight["codes"]], tensors[weight["scale_exp"]]
        shape = list(site["weight_shape"])
        code_shape = shape if site["kind"] == "linear" else [shape[0], codes.numel() // shape[0]]
        if codes.dtype != torch.uint8 or exponent.dtype != torch.int32 or list(codes.shape) != code_shape:
            raise ValueError(f"export code/exponent dtype or code shape differs: {name}")
        if tuple(exponent.shape) not in ((), (1,), (shape[0], 1)):
            raise ValueError(f"unsupported exponent shape: {name}")


def _state_receipt(items, tensors, policy, prefix, source_prefix):
    receipt = {}
    for name, value in items:
        source = _source_name(name, prefix, source_prefix)
        if value.device.type == "meta":
            site = policy["sites"].get(source.removesuffix(".weight"))
            if not source.endswith(".weight") or not site or site["status"] != "quantized":
                raise ValueError(f"unexpected meta state: {name}")
            if list(value.shape) != site["weight_shape"] or str(value.dtype).removeprefix("torch.") != site["weight_dtype"]:
                raise ValueError(f"meta placeholder dtype or shape differs: {name}")
            continue
        record = tensor_record(value)
        if source not in tensors or record != tensor_record(tensors[source]):
            raise ValueError(f"state differs from immutable export: {name}")
        receipt[name] = {"source_name": source, **record}
    return receipt


def _bound_arguments(node):
    schema = node.target._schema
    result = []
    for index, spec in enumerate(schema.arguments):
        if index < len(node.args):
            value = node.args[index]
        elif spec.name in node.kwargs:
            value = node.kwargs[spec.name]
        elif spec.has_default_value():
            value = spec.default_value
        else:
            raise ValueError(f"missing native schema argument: {spec.name}")
        result.append((spec.name, value))
    return result


def _tuple_source(value):
    if value.op == "call_function" and value.target is operator.getitem:
        parent, index = value.args
        if not isinstance(parent, torch.fx.Node) or type(index) is not int:
            raise ValueError("ambiguous native tuple result")
        if isinstance(parent.meta.get("val"), (tuple, list)):
            return parent, index
    return value, 0


def _native_name(node):
    target = str(node.target)
    if node.op == "call_function" and target.startswith("atlas."):
        name = target.split(".")[1]
        if name not in NATIVE_OPS:
            raise ValueError(f"unsupported native operator: {target}")
        return name
    return None


def _emit_like(graph, node, target, operands):
    """Insert ``target(*operands)`` before ``node`` with fake-tensor metadata and node provenance."""
    from torch._guards import detect_fake_mode
    from torch.fx.passes.shape_prop import _extract_tensor_metadata

    values = [item.meta["val"] if isinstance(item, torch.fx.Node) else item for item in operands]
    with detect_fake_mode([node.meta["val"]]):
        value = target(*values)
    with graph.inserting_before(node):
        result = graph.call_function(target, tuple(operands))
    result.meta.update({key: item for key, item in node.meta.items() if key not in {"val", "tensor_meta"}},
                       val=value, tensor_meta=_extract_tensor_metadata(value))
    return result


def _patch_im2col(graph, node):
    """Non-overlapping patch im2col (stride = kernel, no padding or dilation) is a pure layout.

    unfold(x)[b, (c, i, j), (p, q)] = x[b, c, p*kh + i, q*kw + j], which is
    reshape(b, c, oh, kh, ow, kw), permute(0, 1, 3, 5, 2, 4), reshape(b, c*kh*kw, oh*ow).
    """
    x, kernel, dilation, padding, stride = node.args
    value = x.meta["val"]
    if (node.kwargs or list(dilation) != [1, 1] or list(padding) != [0, 0] or list(stride) != list(kernel)
            or value.dim() != 4 or value.shape[2] % kernel[0] or value.shape[3] % kernel[1]):
        return []
    batch, channels, height, width = value.shape
    kh, kw = kernel
    oh, ow = height // kh, width // kw
    current = _emit_like(graph, node, torch.ops.aten.reshape.default, [x, [batch, channels, oh, kh, ow, kw]])
    current = _emit_like(graph, node, torch.ops.aten.permute.default, [current, [0, 1, 3, 5, 2, 4]])
    current = _emit_like(graph, node, torch.ops.aten.reshape.default, [current, [batch, channels * kh * kw, oh * ow]])
    if current.meta["val"].shape != node.meta["val"].shape or current.meta["val"].dtype != node.meta["val"].dtype:
        raise ValueError(f"patch im2col layout changed the result ABI: {node.name}")
    node.replace_all_uses_with(current)
    graph.erase_node(node)
    return [{"node": node.name, "operator": "im2col", "layout": "non-overlapping patches"}]


def _split_native_tails(program):
    """Rewrite bias and convolution tails of native contractions into their ordinary source nodes.

    ``atlas::qlinear`` with a bias is the bias-free contraction followed by ``y + bias`` in the
    layer dtype, and ``atlas::qconv2d`` is the same contraction on im2col codes followed by
    ``RefQuantConv2d.forward``'s reshape, permute, reshape, and bias addition (atlas_fx_ops
    defines both bodies this way). Codes, exponents, and bias tensors are passed unchanged.
    """
    from torch._guards import detect_fake_mode
    from torch.fx.passes.shape_prop import _extract_tensor_metadata

    graph, rewrites = program.graph, []
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target is torch.ops.aten.im2col.default:
            rewrites.extend(_patch_im2col(graph, node))
            continue
        name = _native_name(node)
        if name not in {"qlinear", "qconv2d"}:
            continue
        args = dict(_bound_arguments(node))
        bias = args["bias"]
        if name == "qlinear" and bias is None:
            continue
        dtype = getattr(torch, args["out_dtype"])
        if bias is not None and bias.meta["val"].dtype != dtype:
            raise ValueError(f"native bias dtype differs from its layer dtype: {node.name}")
        fake = detect_fake_mode([node.meta["val"]])
        meta = {key: value for key, value in node.meta.items() if key not in {"val", "tensor_meta"}}

        def emit(target, operands):
            values = [item.meta["val"] if isinstance(item, torch.fx.Node) else item for item in operands]
            with fake:
                value = target(*values)
            result = graph.call_function(target, tuple(operands))
            result.meta.update(meta, val=value, tensor_meta=_extract_tensor_metadata(value))
            return result

        with graph.inserting_before(node):
            current = emit(torch.ops.atlas.qlinear.default,
                           [args["x_codes"], args["x_exp"], args["w_codes"], args["w_exp"], None,
                            args["out_dtype"], args["weight_format"]])
            if name == "qconv2d":
                batch, height, width = args["batch"], args["out_h"], args["out_w"]
                channels = args["w_codes"].meta["val"].shape[0]
                current = emit(torch.ops.aten.reshape.default, [current, [batch, height * width, channels]])
                current = emit(torch.ops.aten.permute.default, [current, [0, 2, 1]])
                current = emit(torch.ops.aten.reshape.default, [current, [batch, channels, height, width]])
                if bias is not None:
                    bias = emit(torch.ops.aten.view.default, [bias, [1, -1, 1, 1]])
            if bias is not None:
                current = emit(torch.ops.aten.add.Tensor, [current, bias])
        if current.meta["val"].shape != node.meta["val"].shape or current.meta["val"].dtype != node.meta["val"].dtype:
            raise ValueError(f"native tail rewrite changed the result ABI: {node.name}")
        node.replace_all_uses_with(current)
        graph.erase_node(node)
        rewrites.append({"node": node.name, "operator": name, "bias": bias is not None})
    if rewrites:
        graph.lint()
        program.graph_module.recompile()
    return rewrites


def _calls(program, policy, buffers, *, allow_no_native_calls=False):
    from m2m.capture.trace import _literal, snapshot_exported_program

    snapshot_exported_program(program, stage="original")
    inputs = {spec.arg.name: (index, str(spec.target)) for index, spec in
              enumerate(program.graph_signature.input_specs) if hasattr(spec.arg, "name")}
    nodes = list(program.graph.nodes)
    native = {node: _native_name(node) for node in nodes if _native_name(node)}
    weights = {site["weight"]["codes"]: name for name, site in policy["sites"].items()
               if site["status"] == "quantized" and site["kind"] in {"linear", "conv2d"}}
    sites = {}
    for node, name in native.items():
        args = dict(_bound_arguments(node))
        if name in {"qlinear", "qconv2d"}:
            code = args["w_codes"]
            target = inputs.get(code.name, (None, None))[1]
            if target not in buffers or buffers[target]["source_name"] not in weights:
                raise ValueError("native contraction lacks authenticated lifted weight binding")
            site = weights[buffers[target]["source_name"]]
            exponent_target = inputs.get(args["w_exp"].name, (None, None))[1]
            if exponent_target not in buffers or buffers[exponent_target]["source_name"] != policy["sites"][site]["weight"]["scale_exp"]:
                raise ValueError("native contraction has the wrong exponent buffer")
            sites[node] = site
            quantizer, result = _tuple_source(args["x_codes"])
            if native.get(quantizer) != "quantize_act" or result != 0:
                raise ValueError("native contraction activation is not a codes/exponent pair")
            exp_quantizer, exp_result = _tuple_source(args["x_exp"])
            if exp_quantizer is not quantizer or exp_result != 1:
                raise ValueError("native activation exponent does not match codes")
            if quantizer in sites and sites[quantizer] != site:
                raise ValueError("activation quantizer ambiguously serves different sites")
            sites[quantizer] = site
        elif name == "quantize_act" and args.get("tag"):
            site, separator, role = args["tag"].rpartition(":")
            if not separator or site not in policy["sites"] or role not in {"q", "k", "p", "v"}:
                raise ValueError("native attention tag has no policy site")
            sites[node] = site
    for node, name in native.items():
        if name in {"qk_matmul", "pv_matmul"}:
            first = next(iter(dict(_bound_arguments(node)).values()))
            quantizer, index = _tuple_source(first)
            if index or quantizer not in sites:
                raise ValueError("native attention contraction lacks a policy-bound quantizer")
            sites[node] = sites[quantizer]
    counts, records = Counter(), []
    for node, name in native.items():
        if node not in sites:
            raise ValueError(f"native call has no unique policy site: {node.name}")
        site = sites[node]
        operands, literals = [], {}
        for role, value in _bound_arguments(node):
            if isinstance(value, torch.fx.Node):
                producer, result = _tuple_source(value)
                input_index, target = inputs.get(producer.name, (None, None))
                operands.append({"role": role, "node_id": producer.meta["_m2m_node_id"],
                                 "result_index": result, "buffer": target if target in buffers else None,
                                 "input_index": input_index, **_type(value.meta.get("val"))})
            else:
                literals[role] = _literal(value)
        if name == "pv_matmul" and literals.get("pop_shift", 0):
            raise ValueError("nonzero PV pop_shift is unsupported")
        values = node.meta.get("val")
        values = list(values) if isinstance(values, (tuple, list)) else [values]
        key = (site, name)
        records.append({"node_id": node.meta["_m2m_node_id"], "site_id": site,
                        "invocation": counts[key], "target": str(node.target),
                        "schema": str(node.target._schema), "operands": operands, "literals": literals,
                        "results": [{"index": index, **_type(value)} for index, value in enumerate(values)]})
        counts[key] += 1
    if not records and not allow_no_native_calls:
        raise ValueError("capture contains no native Atlas calls")
    return records


def atlas_fp8_adapter(model, example_inputs, *, contract_bytes, policy_bytes,
                      original_frontend_snapshot=None):
    """Implement ``m2m.quantization_adapters`` without modifying export tensors."""
    saved = isinstance(model, torch.export.ExportedProgram)
    if not saved and (model.training or any(module.training for module in model.modules())):
        raise ValueError("immutable capture requires eval mode")
    if not example_inputs:
        raise ValueError("capture requires example inputs")
    root = _path(model._atlas_export_dir)
    if (root / "manifest.json").read_bytes() != contract_bytes or (root / "policy.json").read_bytes() != policy_bytes:
        raise ValueError("selected contract/policy bytes differ from export files")
    contract, policy = json.loads(contract_bytes), json.loads(policy_bytes)
    if set(contract.get("files", {})) != {"model.safetensors", "policy.json"}:
        raise ValueError("export manifest has an unexpected file roster")
    for name, record in contract["files"].items():
        if _path(root / name).stat().st_size != record.get("bytes"):
            raise ValueError(f"export file byte count differs: {name}")
    # Import the original custom-op definitions before any exported-program load.
    importlib.import_module("atlas_fx_ops")
    load_quantized = importlib.import_module("load_quantized")
    fx_export = importlib.import_module("fx_export")
    tensors, read_policy, read_manifest = load_quantized.read_export(root, verify=True)
    if read_manifest != contract or read_policy != policy or contract.get("policy_sha256") != _sha(policy_bytes):
        raise ValueError("export changed or its selected policy digest differs")
    _validate_policy(policy, contract, tensors)
    importlib.import_module("tile_native").verify_export(policy, tensors)
    frozen = {name: tensor_record(value) for name, value in sorted(tensors.items())}
    prefix = getattr(model, "_atlas_export_prefix", "")
    source_prefix = getattr(model, "_atlas_source_prefix", "")
    if saved:
        path = _path(model._atlas_saved_program_path)
        if _sha(path.read_bytes()) != model._atlas_saved_program_sha256:
            raise ValueError("saved native program differs from selected source bytes")
        selected = load_program(path)
        if (str(selected.graph) != str(model.graph)
                or selected.graph_signature != model.graph_signature
                or [_value_abi(n.meta.get("val")) for n in selected.graph.nodes]
                   != [_value_abi(n.meta.get("val")) for n in model.graph.nodes]
                or [n.meta.get("nn_module_stack") for n in selected.graph.nodes]
                   != [n.meta.get("nn_module_stack") for n in model.graph.nodes]):
            raise ValueError("saved native graph differs from selected source program")
        # Replay only a loaded native graph. The selected path and tensor inventory
        # bind it to the source capsule; every lifted state tensor is authenticated
        # below against read_export, without re-exporting or touching its dtype.
        program = model
        before = _state_receipt(program.state_dict.items(), tensors, policy, prefix, source_prefix)
    else:
        state = list(model.named_parameters()) + list(model.named_buffers())
        before = _state_receipt(state, tensors, policy, prefix, source_prefix)
        for path, layer in model.named_modules():
            if fx_export._is_quant_layer(layer):
                if layer.layer_name != _source_name(path, prefix, source_prefix):
                    raise ValueError(f"quantized module policy identity differs: {path}")
                if "forward" in layer.__dict__:
                    raise ValueError(f"quantized instance forward overrides are unsupported: {path}")
        with torch.no_grad(), fx_export._native_sites(model, policy, storage="uint8") as plan:
            unsupported = [path for path, layer in model.named_modules()
                           if fx_export._is_quant_layer(layer) and path not in plan["linear"]
                           and path not in plan["conv2d"]]
            if unsupported:
                raise ValueError(f"quantized boundary sites have no native route: {unsupported}")
            if plan["attention"] is None and any(isinstance(getattr(layer, "_atlas_ref_attn", None), dict)
                                                 for layer in model.modules()):
                raise ValueError("attention boundary has no native policy route")
            program = torch.export.export(model, tuple(example_inputs), strict=True)
    # State and buffers include nonpersistent lifted constants, with their original names.
    lifted = dict(program.state_dict)
    lifted.update(dict(program.named_buffers()))
    graph_constants = {}
    for spec in program.graph_signature.input_specs:
        if str(spec.kind).endswith("CONSTANT_TENSOR"):
            target = str(spec.target)
            value = program.constants[target]
            if target in before:
                lifted[target] = value
            else:
                # Exported tensor literals are not original model state. Keep their
                # explicit InputKind binding separate from the source tensor roster.
                graph_constants[target] = {"input_kind": "CONSTANT_TENSOR", **tensor_record(value)}
                lifted.pop(target, None)
    buffers = _state_receipt(lifted.items(), tensors, policy, prefix, source_prefix)
    after_items = program.state_dict.items() if saved else list(model.named_parameters()) + list(model.named_buffers())
    after = _state_receipt(after_items, tensors, policy, prefix, source_prefix)
    if before != after or frozen != {name: tensor_record(value) for name, value in sorted(tensors.items())}:
        raise ValueError("capture changed immutable source state")
    # Every declared raw code/exponent buffer in this boundary must survive lifting.
    required = {name for name, record in before.items() if record["source_name"].endswith((".weight_codes", ".weight_scale_exp"))}
    if not required <= set(buffers):
        raise ValueError("capture dropped declared raw codes or exponents")
    tails = _split_native_tails(program)
    calls = _calls(program, policy, buffers, allow_no_native_calls=saved)
    native_sites = {call["site_id"] for call in calls}
    site_records = []
    for name, site in policy["sites"].items():
        status = site["status"]
        record = {"site_id": name, "status": status, "kind": site["kind"]}
        if name in native_sites:
            record.update(status="quantized", execution_route="atlas.native_fp8")
        elif status == "quantized":
            record.update(status="skipped", reason="outside captured native boundary")
        site_records.append(record)
    manifest = {"schema": "m2m.quantization_manifest.v2", "adapter_id": "atlas_fp8",
                "contract_sha256": _sha(contract_bytes), "policy_sha256": _sha(policy_bytes),
                "sites": site_records,
                "atlas": {"schema": SCHEMA, "storage": "uint8", "calls": calls,
                          "buffers": buffers, "source_tensors": frozen,
                          "graph_constants": graph_constants,
                          "bf16_opt_out": policy.get("bf16_opt_out", []),
                          "source_provenance": contract.get("provenance", {}),
                          "source_recipe": contract.get("recipe", {}),
                          "native_tail_rewrites": tails,
                          "weight_conversion_performed": False, "pack_fold_performed": False}}
    if saved:
        manifest["atlas"]["saved_native_program"] = {
            "sha256": model._atlas_saved_program_sha256,
            "no_native_calls": not calls,
            "original_trace_role": "already_native_source_graph",
        }
    json.dumps(manifest, sort_keys=True, allow_nan=False)
    return program, manifest


def load_program(path):
    """Register the pinned source custom ops before loading a saved native export."""
    importlib.import_module("atlas_fx_ops")
    return torch.export.load(_path(path))
