#!/usr/bin/env python3
"""Check the hand OOT evidence ledger against pinned selected source files.

This is a census/drift check. It does not turn a BitPat or decoder row into a
legality, numerical, timing, or hardware-execution qualification.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/selected-variant-inventory.json"


def load_inventory(path: pathlib.Path = INVENTORY) -> dict:
    return json.loads(path.read_text())


def source_rows(instruction_text: str, decode_text: str) -> tuple[dict[str, str], dict[str, str]]:
    patterns = re.findall(r'def\s+(\w+)\s*=\s*BitPat\("b([01?_]+)"\)', instruction_text)
    decode = re.findall(r'^\s*(\w+)\s*->\s*List\(([^)]*)\)', decode_text, re.M)
    if len(patterns) != len(dict(patterns)) or len(decode) != len(dict(decode)):
        raise ValueError("duplicate RTL BitPat or decoder row")
    bits = {name: pattern.replace("_", "") for name, pattern in patterns}
    controls = {name: ",".join(field.strip() for field in fields.split(","))
                for name, fields in decode}
    if any(len(pattern) != 32 for pattern in bits.values()):
        raise ValueError("RTL BitPat is not 32 bits")
    return bits, controls


def validate_domains(inventory: dict, rtl_root: pathlib.Path | None = None) -> None:
    """Check authored finite domains without promoting them to legal hardware modes."""
    domains = inventory["parameter_domains"]
    if not isinstance(domains, dict) or not domains:
        raise ValueError("parameter domains must be a nonempty mapping")
    for name, domain in domains.items():
        if not isinstance(name, str) or not name or not isinstance(domain, dict):
            raise ValueError(f"unstructured parameter domain: {name}")
        if not isinstance(domain.get("unit"), str) or not domain["unit"]:
            raise ValueError(f"missing parameter unit: {name}")
        if type(domain.get("reviewed")) is not bool:
            raise ValueError(f"missing review status: {name}")
        sources = domain.get("evidence_sources")
        if (not isinstance(sources, list) or not sources or
                any(not isinstance(source, str) for source in sources) or
                len(sources) != len(set(sources))):
            raise ValueError(f"missing parameter evidence: {name}")
        for source in sources:
            if not source or pathlib.Path(source).is_absolute() or ".." in pathlib.Path(source).parts:
                raise ValueError(f"invalid parameter evidence path: {name}")
            if rtl_root is not None and not (rtl_root / source).is_file():
                raise ValueError(f"parameter evidence missing from RTL: {name}: {source}")
        kind = domain.get("kind")
        if kind == "integer":
            if set(domain) != {"kind", "unit", "intervals", "reviewed", "evidence_sources"}:
                raise ValueError(f"malformed integer parameter domain: {name}")
            intervals = domain["intervals"]
            if not isinstance(intervals, list) or not intervals:
                raise ValueError(f"empty integer parameter domain: {name}")
            previous_max = None
            for interval in intervals:
                if not isinstance(interval, dict) or set(interval) != {"min", "max", "step"}:
                    raise ValueError(f"malformed interval: {name}")
                low, high, step = (interval[field] for field in ("min", "max", "step"))
                if (any(type(value) is not int for value in (low, high, step))
                        or step <= 0 or low > high or (high - low) % step
                        or previous_max is not None and low <= previous_max):
                    raise ValueError(f"invalid interval: {name}")
                previous_max = high
        elif kind == "enum":
            if set(domain) != {"kind", "unit", "values", "reviewed", "evidence_sources"}:
                raise ValueError(f"malformed enum parameter domain: {name}")
            values = domain["values"]
            if (not isinstance(values, list) or not values
                    or any(type(value) is not int for value in values)
                    or len(values) != len(set(values))):
                raise ValueError(f"invalid enum parameter domain: {name}")
        else:
            raise ValueError(f"unknown parameter domain kind: {name}")


def validate_rows(inventory: dict, instruction_text: str, decode_text: str) -> None:
    validate_domains(inventory)
    rows = inventory["variants"]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate declared variant")
    if len(ids) != 99:
        raise ValueError(f"expected 99 frozen selected modes, got {len(ids)}")
    bits, controls = source_rows(instruction_text, decode_text)
    if set(ids) != set(bits) or set(ids) != set(controls):
        raise ValueError(f"required variant drift: ledger={set(ids)^set(bits)}, "
                         f"decoder={set(ids)^set(controls)}")
    domains = set(inventory["parameter_domains"])
    for row in rows:
        name = row["id"]
        if row["rtl_bitpat"] != bits[name]:
            raise ValueError(f"BitPat drift: {name}")
        if row["rtl_decode_controls"] != controls[name]:
            raise ValueError(f"decoder control drift: {name}")
        if not row["required"] or not row["represented"] or not row["word_emitted"]:
            raise ValueError(f"declared required/representation/emission drift: {name}")
        if not row["llvm_word_emitted"]:
            raise ValueError(f"declared LLVM word route drift: {name}")
        if row["software_admitted"]:
            raise ValueError(f"unreviewed full-domain admission: {name}")
        if row["standalone_core_executed"] != row["independent_semantic_test"]:
            raise ValueError(f"unmatched bounded evidence fields: {name}")
        if row["independent_semantic_test"] != bool(row["bounded_evidence"]):
            raise ValueError(f"missing bounded evidence identity: {name}")
        if not row["blocked"]:
            raise ValueError(f"missing full-qualification blocker: {name}")
        if not row["parameter_domains"] or set(row["parameter_domains"]) - domains:
            raise ValueError(f"unknown parameter domain: {name}")


def _git_revision(root: pathlib.Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip()


def _require_clean_source(root: pathlib.Path, paths: set[str]) -> None:
    for path in sorted(paths):
        result = subprocess.run(["git", "-C", str(root), "diff", "--quiet",
                                 "HEAD", "--", path], check=False)
        if result.returncode:
            raise ValueError(f"selected source has local changes: {path}")


def validate_sources(inventory: dict, rtl_root: pathlib.Path,
                     model_root: pathlib.Path) -> None:
    source = inventory["selected_sources"]
    if _git_revision(rtl_root) != source["rtl_revision"]:
        raise ValueError("selected RTL revision changed")
    if _git_revision(model_root) != source["model_revision"]:
        raise ValueError("inspected model revision changed")
    validate_domains(inventory, rtl_root)
    _require_clean_source(
        rtl_root, {source["rtl_patterns"], source["rtl_decode"]} |
        {path for row in inventory["variants"]
         for path in row["architecture_sources"]} |
        {path for domain in inventory["parameter_domains"].values()
         for path in domain["evidence_sources"]})
    _require_clean_source(model_root, {source["model_classes"]})
    instructions = (rtl_root / source["rtl_patterns"]).read_text()
    decode = (rtl_root / source["rtl_decode"]).read_text()
    validate_rows(inventory, instructions, decode)
    model_text = (model_root / source["model_classes"]).read_text()
    model_classes = {node.name for node in ast.parse(model_text).body
                     if isinstance(node, ast.ClassDef)}
    declared_ops = {
        f"atlas.{name}" for name in re.findall(
            r'Atlas_MachineOp<"(\w+)">',
            (ROOT / "include/Atlas/AtlasOps.td").read_text())
    }
    test_methods: dict[str, set[str]] = {}
    for row in inventory["variants"]:
        name = row["id"]
        if row["dialect_op"] not in declared_ops:
            raise ValueError(f"dialect operation missing for {name}")
        for path in row["architecture_sources"]:
            if not (rtl_root / path).is_file():
                raise ValueError(f"architecture source missing for {name}: {path}")
        if not row["model_classes"] or set(row["model_classes"]) - model_classes:
            raise ValueError(f"model identity missing for {name}")
        if row["bounded_evidence"]:
            filename, method = row["bounded_evidence"].split("::", 1)
            if filename not in test_methods:
                tree = ast.parse((ROOT / "test" / filename).read_text())
                test_methods[filename] = {node.name for node in ast.walk(tree)
                                          if isinstance(node, ast.FunctionDef)}
            if method not in test_methods[filename]:
                raise ValueError(f"bounded test missing for {name}")


def counts(inventory: dict) -> dict[str, int]:
    fields = ("required", "software_admitted", "represented", "word_emitted",
              "llvm_word_emitted", "independent_semantic_test", "standalone_core_executed")
    result = {field: sum(bool(row[field]) for row in inventory["variants"])
              for field in fields}
    result["blocked"] = sum(bool(row["blocked"]) for row in inventory["variants"])
    result["denominator"] = len(inventory["variants"])
    return result


def family_counts(inventory: dict) -> dict[str, dict[str, int]]:
    families = sorted({row["family"] for row in inventory["variants"]})
    return {family: counts({"variants": [row for row in inventory["variants"]
                                          if row["family"] == family]})
            for family in families}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-root", type=pathlib.Path, required=True)
    parser.add_argument("--model-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        inventory = load_inventory()
        validate_sources(inventory, args.rtl_root, args.model_root)
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"variant inventory FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "counts": counts(inventory),
                      "by_family": family_counts(inventory)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
