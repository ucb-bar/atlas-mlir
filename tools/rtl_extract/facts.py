"""Extractor records and the Merlin ``op_timing`` document built from them.

Records carry an internal ``status``/``reason`` while checks run; ``block`` drops them. Unknown is
``None``, never a guess, and ``evidence`` is the one prose field (the derivation, or why a block is unresolved).
"""
import hashlib
import json
from pathlib import Path

SCHEMA = "merlin.op_timing.v1"
SOURCE = "control_simulation"


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
            "module": spec["module"], "source": SOURCE, "assumptions": assumptions(spec),
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


def block(item):
    """The named ``op_timing`` block for one record; an unresolved record keeps null values and its reason as evidence."""
    out = {key: item[key] for key in ("name", "module", "engine", "operation") if key in item}
    if item["variant"]:
        out["variant"] = item["variant"]
    out.update(source=item["source"], evidence=item["evidence"], assumptions=item["assumptions"],
               events=item["events"], first_free_age=item["first_free_age"])
    if "next_issue_age" in item:
        out["next_issue_age"] = item["next_issue_age"]
    return out


def document(hw_ir, records):
    """The extractor output: IR identity (path and SHA-256) and the ``op_timing`` named-block list."""
    return {"schema": SCHEMA, "hw_ir": {"path": str(Path(hw_ir).resolve()), "sha256": sha256(hw_ir)},
            "op_timing": [block(item) for item in records]}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
