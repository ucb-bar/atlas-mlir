"""Results shaped like a proposed Merlin ``op_timing`` facts block (pending review)."""
import hashlib
import json
from pathlib import Path

SCHEMA = "atlas.op_timing.v0-proposal"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assumptions(spec):
    latencies = {name: e["response"]["latency"] for name, e in spec["events"].items() if "response" in e}
    latency = next(iter(set(latencies.values()))) if len(set(latencies.values())) == 1 else latencies or None
    return {"scratchpad_read_latency": latency, "reset_cycles": spec["reset_cycles"],
            "flush_cycles": spec["flush_cycles"], "limit": spec["limit"]}


def record(spec, operation, variant=None, result=None, reason=None, evidence=None):
    """One record; ``result`` comes from simulation, otherwise the record is unresolved with ``reason``."""
    name = f"{spec['engine']}.{operation}" + (f"/{variant}" if variant else "")
    item = {"name": name, "engine": spec["engine"], "operation": operation, "variant": variant,
            "module": spec["module"], "source": "control_simulation", "assumptions": assumptions(spec),
            "events": None, "first_free_age": None}
    if any("next_issue" in op for op in spec["operations"].values()):
        item["next_issue_age"] = None
    if result is None:
        return {**item, "status": "unresolved", "reason": reason, "evidence": evidence or reason}
    return {**item, **result, "status": "computed", "evidence": evidence}


def unresolve(item, reason):
    item.update(events=None, first_free_age=None, status="unresolved", reason=reason, evidence=reason)
    if "next_issue_age" in item:
        item["next_issue_age"] = None


def document(target, hw_ir, exporter, records):
    return {"schema": SCHEMA, "target": target,
            "hw_ir": {"path": str(Path(hw_ir).resolve()), "sha256": sha256(hw_ir)},
            "exporter": {"path": str(Path(exporter).resolve())}, "records": records}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
