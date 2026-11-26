"""Reference coverage: the rule that evaluates each source node, and every refusal at once.

``audit`` runs the reference's own ``evaluate`` with every check it applies in the reference stage, with two
differences. A node the reference refuses is recorded, then bound to a stand-in of its declared dtype and shape (ones
for floating values, zeros for integers, true for booleans), so the walk reaches every later node. And the device
contraction kernel, the K32 MXU emulation that is nearly all of the stage's cost, returns a stand-in readout after
its argument and rule checks. No reference rule is copied or relaxed here.

Placement, handler, stage, and conversion refusals depend only on source metadata and the plan. A rule refusal at a
node that reads a stand-in, directly or through any earlier node derived from one, is marked ``value_dependent``: its
reason may come from the stand-in data rather than a missing rule. Every other refusal is a missing rule. A refused
output is recorded like any other refusal. Values are not compared, and device numerics are not evaluated: a covered node can still refuse
on its real values in the reference stage, which stays the final word.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .evaluator import _STORAGE, Tensor, _call, _dtype, _out_of_place

SCHEMA = "atlas.reference_coverage.v1"
_PLACEMENT = "no authenticated arithmetic placement for source node"


def _kind(reason):
    if reason.startswith(_PLACEMENT):
        return "placement"
    if reason == "no independent source handler":
        return "handler"
    if reason.startswith("invalid numerical stage graph"):
        return "stage"
    if reason.startswith("unsupported source dtype conversion"):
        return "conversion"
    return "rule"


def _general(reason):
    """A reason with node ids removed, so one missing rule groups all its nodes."""
    return re.sub(r"g:[A-Za-z0-9_]+:[A-Za-z0-9_]+:n\d+", "<node>", reason)


def _form(node, values):
    """Operand dtypes, the result dtype, and whether a Python scalar or an in-place destination is involved."""
    tensors, scalars = [], 0
    for value in values:
        if isinstance(value, Tensor):
            tensors.append(value.dtype)
        elif isinstance(value, (bool, int, float)):
            scalars += 1
    result = [spec.get("dtype") for spec in node.get("results", []) if spec.get("dtype") is not None]
    form = "in-place" if _out_of_place(str(node.get("target", ""))) else "out-of-place"
    return {"operands": tensors, "scalars": scalars, "results": [str(_dtype(d)) for d in result], "form": form}


def _flat(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _flat(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flat(item)
    else:
        yield value


def _references(value):
    if isinstance(value, dict):
        if "node_id" in value or "value_id" in value:
            yield value.get("node_id", value.get("value_id"))
            return
        for item in value.values():
            yield from _references(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _references(item)


def _stand_in(spec):
    dtype = _dtype(spec["dtype"])
    shape = tuple(spec["shape"])
    fill = {"bf16": 0x3F80, "f32": 0x3F800000, "bool": True}.get(dtype, 0)
    return Tensor(np.full(shape, fill, _STORAGE[dtype]), dtype)


class Audit:
    """Passed to ``evaluate(..., audit=)``; collects one row per source node."""

    def __init__(self):
        self.rows, self.stand_in, self._node, self._args, self._owners = [], set(), None, None, {}

    def _reads(self, node):
        """Source nodes this node reads, through node-id and value-id references."""
        refs = _references([node.get("args", []), node.get("kwargs", {})])
        return {self._owners.get(ref, ref) for ref in refs}

    def begin(self, node):
        self._node, self._args = node, None
        node_id = str(node.get("id"))
        self._owners.update((str(spec["id"]), node_id) for spec in node.get("results", []) if "id" in spec)
        self.rows.append({"id": node_id, "target": str(node.get("target", node.get("op"))), "rules": []})
        # A value derived from a stand-in, through views, aliases, or arithmetic, is a stand-in too.
        if self._reads(node) & self.stand_in:
            self.stand_in.add(node_id)

    def route(self, name):
        self.rows[-1]["rules"].append(name)

    def resolved(self, args, kwargs, rules):
        self._args = (args, kwargs, rules)

    def contraction(self, shape):
        self.stand_in.add(self.rows[-1]["id"])
        self.rows[-1]["device_numerics"] = "stand-in"
        return np.full(shape, 0x3F80, np.uint16)

    def refused(self, node, reason):
        row = self.rows[-1]
        reads = self._reads(node)
        kind = _kind(reason)
        row.update(status="refused", reason=reason, kind=kind, value_dependent=kind == "rule" and bool(reads & self.stand_in))
        if self._args is not None:
            args, kwargs, rules = self._args
            row["signature"] = _form(node, list(_flat([args, kwargs])))
            if reason.startswith(_PLACEMENT):
                # Behind a missing placement there can be a missing host rule as well; try the host rule alone.
                try:
                    _call(node["target"], args, kwargs, {**rules, "execution": "host"})
                except (ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
                    row["host_rule"] = str(error)
                else:
                    row["host_rule"] = "evaluates"
        self.stand_in.add(row["id"])
        results = node.get("results", [])
        if any(spec.get("dtype") is None for spec in results):
            return None
        values = [_stand_in(spec) for spec in results]
        return values[0] if len(values) == 1 else values

    def report(self, decisions=None):
        applied = (decisions.record.get("applied", {}) if decisions is not None else {})
        for row in self.rows:
            entry = applied.get(row["id"])
            if entry is not None:
                row["decision"] = ("compile_time_constant" if "compile_time_constant" in entry
                                   else {"placements": entry.get("placements"), "routes": entry.get("numerical_routes"),
                                         **({"stages": True} if "numerical_stages" in entry else {})})
            row.setdefault("status", "covered")
        refused = [row for row in self.rows if row["status"] == "refused"]
        groups = {}
        for row in refused:
            signature = row.get("signature", {})
            key = (row["kind"], row["target"], _general(row["reason"]), tuple(signature.get("operands", ())),
                   signature.get("scalars", 0), tuple(signature.get("results", ())), signature.get("form"),
                   row["value_dependent"], row.get("host_rule"))
            group = groups.setdefault(key, {"kind": row["kind"], "target": row["target"], "reason": _general(row["reason"]),
                                            "signature": signature, "value_dependent": row["value_dependent"],
                                            **({"host_rule": _general(row["host_rule"])} if "host_rule" in row else {}),
                                            "count": 0, "nodes": []})
            group["count"] += 1
            group["nodes"].append(row["id"])
        return {"schema": SCHEMA, "nodes": len(self.rows), "covered": len(self.rows) - len(refused),
                "refused": len(refused), "missing_rules": sum(1 for row in refused if not row["value_dependent"]),
                "rules": dict(Counter("+".join(row["rules"]) or "none" for row in self.rows if row["status"] == "covered")),
                "groups": sorted(groups.values(), key=lambda g: (g["value_dependent"], g["kind"], g["target"])),
                "per_node": self.rows}


def audit(nodes, constants, inputs, target_rules=None, *, decisions=None):
    """Run the coverage audit; returns its report."""
    from .evaluator import evaluate

    recorder = Audit()
    evaluate(nodes, constants, inputs, target_rules, decisions=decisions, audit=recorder)
    return recorder.report(decisions)
