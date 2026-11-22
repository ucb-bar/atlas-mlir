"""Authenticate selected placement metadata and join it to source provenance.

Only frontend source MLIR provenance and selected-plan metadata are inspected.
The parser is used without translation, lowering, or numerical execution.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from .source_io import checked_path, digest
from .stages import CONTRACTS, build, composite_silu_route


def _source_provenance(text, *, bindings=False):
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    module = parse_mlir_text(text)
    functions = [op for op in module.walk() if op.name == "func.func" and len(op.body.blocks)]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("source provenance requires one single-block function")
    result = {}
    values = {arg: f"arg{i}" for i, arg in enumerate(functions[0].body.block.args)}
    source_bindings, groups, literals, splats = {}, defaultdict(list), {}, {}
    for ordinal, op in enumerate(functions[0].body.block.ops):
        operands = [values.get(arg) for arg in op.operands]
        results = [f"op{ordinal}r{i}" for i, _ in enumerate(op.results)]
        values.update(zip(op.results, results))
        lowered = _lowered(op)
        if "f32" in lowered:
            literals[results[0]] = lowered["f32"]
        elif op.name == "tensor.splat" and operands and operands[0] in literals:
            # Equal splats anywhere in the graph are interchangeable inputs once providers share constants.
            splats[results[0]] = [literals[operands[0]], str(op.results[0].type)]
        origins = op.attributes.get("prov.origin_node_ids")
        if origins is None:
            continue
        if not isinstance(origins, ArrayAttr) or any(not isinstance(x, StringAttr) for x in origins):
            raise ValueError("invalid frontend origin provenance")
        # The source evaluator consumes the quantized FX graph, not the prepared graph.
        result[ordinal] = sorted({x.data for x in origins
                                  if x.data.startswith("g:quantized:")})
        for node_id in result[ordinal]:
            target = op.attributes.get("prov.aten")
            bias_order = op.attributes.get("bias_order")
            record = {"operands": operands, "results": results,
                      "target": target.data if isinstance(target, StringAttr) else None,
                      "bias_order": bias_order.data if isinstance(bias_order, StringAttr) else None}
            groups[node_id].append({**record, **lowered})
            if node_id in source_bindings:
                source_bindings[node_id] = None  # many source operations need a separate adapter
            else:
                source_bindings[node_id] = record
    # A source operation lowered to several prepared operations binds as their group, in source order.
    for node_id, ops in groups.items():
        if len(ops) > 1:
            targets = {op["target"] for op in ops}
            source_bindings[node_id] = {"target": targets.pop() if len(targets) == 1 else None, "ops": ops,
                                        "splats": splats}
    return source_bindings if bindings else result


def _lowered(op):
    """What a stage contract may check about one prepared operation: its name, literal, reduced axes, maps, and body."""
    from xdsl.dialects.builtin import DenseArrayBase, FloatAttr

    record = {"name": op.name, "operand_types": [str(x.type) for x in op.operands],
              "result_types": [str(x.type) for x in op.results]}
    value = op.properties.get("value")
    if op.name == "arith.constant" and isinstance(value, FloatAttr) and str(value.type) == "f32":
        record["f32"] = value.value.data
    dimensions = op.properties.get("dimensions")
    if op.name == "linalg.reduce" and isinstance(dimensions, DenseArrayBase):
        record["dimensions"] = list(dimensions.get_values())
    maps = op.properties.get("indexing_maps")
    if maps is not None:
        record["indexing_maps"] = [str(m) for m in maps]
        record["identity_maps"] = len(maps) == len(op.operands) and all(
            re.fullmatch(r"affine_map<\((.*)\) -> \(\1\)>", str(m)) for m in maps)
        iterators = op.properties.get("iterator_types", ())
        record["parallel_iterators"] = all(str(x) == "#linalg.iterator_type<parallel>" for x in iterators)
        record["iterator_count"] = len(iterators)
        segments = op.properties.get("operandSegmentSizes")
        if isinstance(segments, DenseArrayBase):
            record["operand_segments"] = list(segments.get_values())
    if op.regions and op.regions[0].blocks:
        if len(op.regions[0].blocks) != 1:
            return record
        block = op.regions[0].block
        values = {arg: i for i, arg in enumerate(block.args)}
        body, types = [], [str(arg.type) for arg in block.args]
        for ordinal, inner in enumerate(block.ops):
            body.append([inner.name, *(values.get(x) for x in inner.operands)])
            values.update((value, f"body{ordinal}r{i}") for i, value in enumerate(inner.results))
            types.extend(str(value.type) for value in inner.results)
        record["body"], record["body_types"] = body, types
        record["strict_float"] = all(inner.properties.get("fastmath") is None or
                                     str(inner.properties["fastmath"]) == "#arith.fastmath<none>" for inner in block.ops)
    return record


def _join_rows(rows, provenance, members=None):
    """members maps a row to the semantic nodes it computes, from the authenticated request."""
    by_node = defaultdict(list)
    decisions = []
    for row in rows:
        decision = {key: row[key] for key in ("id", "placement", "implementation")}
        if "kind" in row.get("metadata", {}):
            decision["kind"] = row["metadata"]["kind"]
        decision["inputs"] = [item["buffer"] for item in row.get("inputs", [])]
        decision["buffers"] = [out["buffer"] for out in row.get("outputs", [])]
        decisions.append(decision)
        ids = set()
        for buffer in decision["buffers"]:
            match = re.match(r"^op(\d+)r\d+(?:[_.]|$)", buffer)
            if match:
                ids.update(provenance.get(int(match[1]), []))
        for node_id in sorted(ids):
            by_node[node_id].append(decision)
    # A source operation that no selected output names (one inside a fused
    # region) belongs to the one region whose member nodes compute it.
    named, inner = set(by_node), defaultdict(list)
    for decision in decisions:
        ids = set()
        for name in (members or {}).get(decision["id"], ()):
            match = re.match(r"^op(\d+)r\d+(?:[_.]|$)", name)
            if match:
                ids.update(provenance.get(int(match[1]), []))
        for node_id in sorted(ids - named):
            inner[node_id].append(decision)
    ambiguous = {node_id for node_id, rows in inner.items() if len(rows) > 1}
    by_node.update((node_id, rows) for node_id, rows in inner.items() if node_id not in ambiguous)
    # Walk provider edges, stopping at other source results. Intermediate names
    # have no semantics and need not contain the source operation ordinal.
    producers = defaultdict(list)
    for row in decisions:
        for output in row["buffers"]:
            producers[output].append(row)
    owners = defaultdict(set)
    for node_id, seeds in by_node.items():
        for row in seeds:
            owners[row["id"]].add(node_id)
    for node_id, seeds in by_node.items():
        owned = {r["id"] for r in seeds}
        pending = list(seeds)
        while pending:
            for value in pending.pop()["inputs"]:
                for producer in producers[value]:
                    if owners[producer["id"]] and node_id not in owners[producer["id"]]:
                        continue
                    if producer["id"] not in owned:
                        owned.add(producer["id"])
                        pending.append(producer)
        by_node[node_id] = [r for r in decisions if r["id"] in owned]
    # None marks a source operation that several fused regions compute; it refuses when evaluated.
    return {**by_node, **dict.fromkeys(ambiguous)}



class SelectedDecisions:
    def __init__(self, application_dir, manifest_sha256, capture_root, capture_receipt_sha256):
        app, capture = checked_path(application_dir), checked_path(capture_root)
        self._identities = {}

        def authenticate(path, expected):
            path = checked_path(path)
            if not isinstance(expected, str) or digest(path) != expected:
                raise ValueError("decision identity differs: " + str(path))
            self._identities[str(path)] = expected
            return path

        manifest_path = authenticate(app / "manifest.json", manifest_sha256)
        manifest = json.loads(manifest_path.read_text())
        plan_path = authenticate(app / "selected-plan.json", manifest.get("stage_artifacts", {}).get("selected-plan.json"))
        plan = json.loads(plan_path.read_text())
        if plan.get("schema") != "global_plan_v1":
            raise ValueError("unsupported selected-plan schema")
        receipt_path = authenticate(capture / "capture_receipt.json", capture_receipt_sha256)
        receipt = json.loads(receipt_path.read_text())
        model_record = receipt["artifacts"]["model.mlir"]
        model_path = authenticate(capture / "model.mlir", model_record["sha256"])
        if model_path.stat().st_size != model_record["bytes"]:
            raise ValueError("source MLIR size differs")
        members = {}
        if "source-request.json" in manifest.get("stage_artifacts", {}):
            request = json.loads(authenticate(app / "source-request.json",
                                              manifest["stage_artifacts"]["source-request.json"]).read_text())
            pure = [node["id"] for node in request["nodes"] if node["effect"] == "pure"]
            for row in plan["selected"]:
                indices = row.get("nodes", [])
                if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < len(pure) for i in indices):
                    raise ValueError("selected region nodes leave the authenticated source request")
                members[row["id"]] = [pure[index] for index in indices]
        self._by_node = _join_rows(plan["selected"], _source_provenance(model_path.read_text()), members)
        self._bindings = _source_provenance(model_path.read_text(), bindings=True)
        self._app, self._authenticate, self._softmax = app, authenticate, {}
        self._artifacts = manifest.get("stage_artifacts", {})
        self._regions = {region.get("id"): region for region in manifest.get("regions", [])}
        self.record = {"schema": "atlas.source_selected_decisions.v1", "identities": self._identities.copy(),
                       "plan_digest": plan.get("digest"), "mapping": "frontend top-level operation ordinal to selected output buffer, else to the region computing it; quantized origin provenance",
                       "compiler_numerical_nodes_read": False,
                       "selected_fields_read": ["schema", "digest", "selected.id", "selected.placement",
                                                "selected.implementation", "selected.inputs.buffer", "selected.outputs.buffer", "selected.metadata.kind",
                                                "selected.nodes", "source_request.nodes.id", "source_request.nodes.effect",
                                                "softmax regions: manifest.regions.id, description.native_export.package_name and source_interface.additive_mask.finite_exclusion_bits; segment source-request nodes.op, nodes.attrs.input_domain, and nodes.type.numerical_policy"],
                       "applied": {}}

    def node_rules(self, node, base_rules):
        from .evaluator import _HOST_OPS
        node_id = str(node["id"])
        rows = self._by_node.get(node_id, [])
        if rows is None:
            raise ValueError("source node is computed inside more than one fused region: " + node_id)
        # Dense buffer views alias storage without arithmetic, like source layouts.
        arithmetic = [r for r in rows if not (r["implementation"].startswith("source_layout")
                                              or r["implementation"] in ("source_constant_only", "merlin.dense_buffer_view"))]
        if not arithmetic:
            arithmetic = [r for r in rows if r["implementation"] == "source_constant_only"]
        if not arithmetic:
            if node.get("op") != "call_function" or node.get("target") in _HOST_OPS:
                return dict(base_rules)
            raise ValueError("no authenticated arithmetic placement for source node " + node_id)
        placements = {r["placement"] for r in arithmetic}
        entry = {"selected": arithmetic, "placements": sorted(placements)}
        self.record["applied"][node_id] = entry
        split = None
        binding = self._bindings.get(node_id)
        if len(placements) > 1 or (len(rows) > 1 and binding and "ops" not in binding and
                binding.get("target") in CONTRACTS):
            split = build(rows, self._bindings.get(node_id), node.get("target"))
        if (len(placements) != 1 and split is None) or not placements <= {"host", "accelerator"}:
            raise ValueError("conflicting arithmetic placements for source node " + node_id)
        rules = {**base_rules, "execution": "device" if "accelerator" in placements else "host", "authenticated_placement": True}
        if split is not None:
            rules["numerical_stages"] = split
            entry["numerical_stages"] = split
        routes = sorted({r["implementation"] for r in arithmetic})
        rules["selected_routes"] = routes
        entry["numerical_routes"] = routes
        composite_silu = composite_silu_route(node.get("target"), routes)
        if "source_exact_float_device" in routes and len(routes) != 1 and not composite_silu and split is None:
            raise ValueError("conflicting exact-float numerical routes for source node " + node_id)
        softmax = [r["id"] for r in arithmetic if r["implementation"] in ("source_softmax", "source_masked_softmax")]
        if softmax:
            if len(softmax) != 1:
                raise ValueError("source node is computed by more than one selected softmax region: " + node_id)
            rules["softmax"] = entry["softmax"] = self._softmax_declaration(softmax[0])
        if any(r["implementation"] == "source_already_quantized" for r in arithmetic):
            from .target_rules import select_mxu
            rules["selected_unit"] = select_mxu(0, route="already-quantized")
            entry["unit"] = {"selected_unit": rules["selected_unit"], "explicit_plan_unit_absent": True,
                             "basis": "shared declared already-quantized route in target_rules.select_mxu"}
        return rules

    def _softmax_declaration(self, region_id):
        """The selected softmax region's own policy, score domain, and mask encoding, as its producer declared them."""
        if region_id not in self._softmax:
            export = self._regions.get(region_id, {}).get("description", {}).get("native_export", {})
            name = f"segments/{export.get('package_name')}/source-request.json"
            if not isinstance(export.get("package_name"), str) or name not in self._artifacts:
                raise ValueError("selected softmax region declares no authenticated row-softmax request: " + region_id)
            path = self._authenticate(self._app / name, self._artifacts[name])
            self.record["identities"][str(path)] = self._artifacts[name]
            nodes = json.loads(path.read_text()).get("nodes", [])
            if (len(nodes) < 2 or nodes[0].get("op") != "input" or nodes[-1].get("op") not in ("softmax", "masked_softmax")
                    or not isinstance(nodes[0].get("attrs"), dict)):
                raise ValueError("selected softmax request has no declared scores input and softmax: " + region_id)
            mask = export.get("source_interface", {}).get("additive_mask") or {}
            self._softmax[region_id] = {"region": region_id, "policy": nodes[-1].get("type", {}).get("numerical_policy"),
                                        "score_domain": nodes[0]["attrs"].get("input_domain"),
                                        "exclusion_bits": mask.get("finite_exclusion_bits")}
        return dict(self._softmax[region_id])

    def verify(self):
        for path, expected in self._identities.items():
            if digest(path) != expected:
                raise ValueError("decision identity changed during evaluation: " + path)
