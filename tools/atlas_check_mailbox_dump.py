"""Check reported Atlas mailbox runs against independently held inputs and outputs.

The runner reports its bitstream identity. Matching bytes do not attest that
the selected bitstream executed or that its cache behavior is qualified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

HEX_SHA = re.compile(r"[0-9a-f]{64}\Z")
MAILBOX_OFFSETS = {"request_id": 0, "response_id": 64, "status": 68,
                   "input": 128, "output": 4224}
TENSOR_BYTES = 32 * 32 * 4
MAILBOX_BYTES = MAILBOX_OFFSETS["output"] + TENSOR_BYTES


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _package(handoff: Path) -> tuple[dict, bytes]:
    manifest = _json(handoff / "manifest.json")
    files = manifest.get("files")
    if (manifest.get("schema") != "atlas.attention_mlp_mailbox_package.v1"
            or manifest.get("status") != "diagnostic_unbound"
            or manifest.get("engine") != "merlin_native"
            or manifest.get("bitstream_bound") is not False
            or manifest.get("mailbox") != MAILBOX_OFFSETS
            or not isinstance(files, dict)):
        raise ValueError("mailbox handoff identity or ABI is incomplete")
    observed = {str(path.relative_to(handoff)) for path in handoff.rglob("*")
                if path.is_file() or path.is_symlink()}
    if observed != set(files) | {"manifest.json"}:
        raise ValueError("mailbox handoff file inventory differs from its manifest")
    for name, identity in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("mailbox handoff has an invalid file path")
        path = handoff / relative
        if path.is_symlink() or not path.is_file() or not isinstance(identity, dict):
            raise ValueError("mailbox handoff member is missing or linked")
        payload = path.read_bytes()
        if identity.get("bytes") != len(payload) or identity.get("sha256") != _sha(payload):
            raise ValueError(f"mailbox handoff member changed: {name}")
    elf = (handoff / "host/full-graph-mailbox.elf").read_bytes()
    status = _json(handoff / "host/handoff-status.json")
    if (status.get("schema") != "atlas.attention_mlp_mailbox_handoff.v1"
            or status.get("status") != "diagnostic_unbound"
            or status.get("elf_sha256") != _sha(elf)
            or status.get("mailbox_offsets") != MAILBOX_OFFSETS
            or status.get("mailbox_bytes") != MAILBOX_BYTES
            or status.get("mailbox_address") != manifest.get("mailbox_address")):
        raise ValueError("mailbox ELF or symbol identity differs from handoff")
    return manifest, elf


def _case_file(cases_path: Path, name: str) -> bytes:
    if not isinstance(name, str) or not name:
        raise ValueError("evaluator case has no file path")
    path = Path(name)
    if not path.is_absolute():
        path = cases_path.parent / path
    if not path.is_file():
        raise ValueError("evaluator case file is absent")
    return path.read_bytes()


def check(handoff: Path, receipt_path: Path, cases_path: Path) -> dict:
    _, elf = _package(handoff)
    receipt = _json(receipt_path)
    evaluator = _json(cases_path)
    reported = receipt.get("cases")
    cases = evaluator.get("cases")
    if (receipt.get("schema") != "atlas.external_mailbox_dump.v1"
            or receipt.get("elf_sha256") != _sha(elf)
            or not isinstance(receipt.get("run_id"), str)
            or not receipt["run_id"]
            or not isinstance(receipt.get("bitstream_sha256"), str)
            or not HEX_SHA.fullmatch(receipt["bitstream_sha256"])
            or evaluator.get("schema") != "atlas.mailbox_evaluation_cases.v1"
            or not isinstance(reported, list)
            or not isinstance(cases, list)
            or len(cases) < 2 or len(reported) != len(cases)):
        raise ValueError("runner receipt or evaluator case roster is incomplete")
    ids: set[int] = set()
    inputs: set[str] = set()
    rows = []
    for index, (case, observation) in enumerate(zip(cases, reported, strict=True)):
        if (not isinstance(case, dict) or not isinstance(observation, dict)
                or set(case) != {"request_id", "input", "expected", "dump"}
                or type(case["request_id"]) is not int
                or not 0 < case["request_id"] < 2**32
                or case["request_id"] in ids
                or observation.get("request_id") != case["request_id"]):
            raise ValueError("mailbox request identity is missing, repeated, or reordered")
        ids.add(case["request_id"])
        source = _case_file(cases_path, case["input"])
        expected = _case_file(cases_path, case["expected"])
        dump = _case_file(cases_path, case["dump"])
        if (len(source) != TENSOR_BYTES or len(expected) != TENSOR_BYTES
                or len(dump) != MAILBOX_BYTES):
            raise ValueError("mailbox input, expected output, or dump has an invalid size")
        input_sha = _sha(source)
        inputs.add(input_sha)
        if (observation.get("input_sha256") != input_sha
                or observation.get("dump_sha256") != _sha(dump)
                or dump[MAILBOX_OFFSETS["input"]:MAILBOX_OFFSETS["input"] + TENSOR_BYTES] != source):
            raise ValueError("mailbox dump differs from the evaluator challenge or runner receipt")
        request = struct.unpack_from("<I", dump, MAILBOX_OFFSETS["request_id"])[0]
        response = struct.unpack_from("<I", dump, MAILBOX_OFFSETS["response_id"])[0]
        status = struct.unpack_from("<I", dump, MAILBOX_OFFSETS["status"])[0]
        if request != case["request_id"] or response != request:
            raise ValueError("mailbox dump has a stale or incomplete response")
        output = dump[MAILBOX_OFFSETS["output"]:]
        matching = sum(left == right for left, right in zip(output, expected, strict=True))
        rows.append({
            "index": index,
            "request_id": request,
            "status": ("execution_failure" if status != 1 else
                       "numerical_mismatch" if matching != TENSOR_BYTES else "reported_match"),
            "status_word": f"0x{status:08x}",
            "matching_output_bytes": matching,
            "total_output_bytes": TENSOR_BYTES,
            "input_sha256": input_sha,
            "output_sha256": _sha(output),
            "expected_sha256": _sha(expected),
        })
    if len(inputs) < 2:
        raise ValueError("mailbox evaluation needs two distinct runtime inputs")
    failed = next((row["status"] for row in rows if row["status"] != "reported_match"), None)
    return {
        "schema": "atlas.external_mailbox_dump_check.v1",
        "status": failed or "reported_match_unqualified",
        "scope": "reported repeated mailbox responses and full independently held f32 outputs",
        "run_id": receipt["run_id"],
        "elf_sha256": _sha(elf),
        "reported_bitstream_sha256": receipt["bitstream_sha256"],
        "passing_cases": sum(row["status"] == "reported_match" for row in rows),
        "total_cases": len(rows),
        "cases": rows,
        "limitations": [
            "bitstream identity is reported by the external runner, not independently attested",
            "matching bytes do not establish cache coherence or a selected SoC execution trace",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        print(json.dumps({"status": "integrity_failure", "reason": "fresh report path required"}),
              file=sys.stderr)
        return 2
    try:
        result = check(args.handoff, args.receipt, args.cases)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        result = {"schema": "atlas.external_mailbox_dump_check.v1",
                  "status": "integrity_failure", "reason": str(error)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if result["status"] == "reported_match_unqualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
