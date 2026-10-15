"""Check a returned Atlas model dump against an evaluator-owned output file.

The runner reports where it ran the ELF. This tool checks bytes and identities;
it cannot attest that the runner used the reported bitstream.
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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def check(handoff: Path, receipt_path: Path, status_path: Path,
          outputs_path: Path, expected_path: Path) -> dict:
    package = _read_json(handoff / "handoff-status.json")
    receipt = _read_json(receipt_path)
    elf = (handoff / "host/full-graph-diagnostic.elf").read_bytes()
    files = package.get("files")
    if not isinstance(files, dict):
        raise TypeError("handoff file inventory is missing")
    host_file = files.get("host/full-graph-diagnostic.elf")
    if not isinstance(host_file, dict):
        raise TypeError("handoff ELF inventory is missing")
    if (package.get("schema") != "atlas.diagnostic_seven_region_handoff.v1"
            or package.get("status") != "diagnostic_unbound"
            or package.get("bitstream_bound") is not False
            or not isinstance(package.get("elf_sha256"), str)
            or _sha(elf) != package["elf_sha256"]
            or host_file.get("sha256") != package["elf_sha256"]):
        raise ValueError("handoff ELF identity is missing or changed")
    if (receipt.get("schema") != "atlas.external_model_dump.v1"
            or receipt.get("elf_sha256") != package["elf_sha256"]
            or not isinstance(receipt.get("run_id"), str)
            or not receipt["run_id"]
            or not isinstance(receipt.get("bitstream_sha256"), str)
            or not HEX_SHA.fullmatch(receipt["bitstream_sha256"])):
        raise ValueError("runner receipt lacks the selected ELF or bitstream identity")
    status = status_path.read_bytes()
    outputs = outputs_path.read_bytes()
    expected = expected_path.read_bytes()
    if (receipt.get("status_sha256") != _sha(status)
            or receipt.get("outputs_sha256") != _sha(outputs)):
        raise ValueError("runner dump differs from its receipt")
    if len(status) != 4 or len(outputs) != 2 * 32 * 32 * 4 or len(expected) != len(outputs):
        raise ValueError("status or two-panel f32 output has an invalid size")
    code = struct.unpack("<I", status)[0]
    different = sum(a != b for a, b in zip(outputs, expected, strict=True))
    result = {
        "schema": "atlas.external_model_dump_check.v1",
        "scope": "reported external dump compared with evaluator-owned two-panel f32 output",
        "status": (
            "execution_failure" if code != 1 else
            "numerical_mismatch" if different else
            "reported_match_unqualified"
        ),
        "run_id": receipt["run_id"],
        "reported_bitstream_sha256": receipt["bitstream_sha256"],
        "elf_sha256": package["elf_sha256"],
        "status_word": f"0x{code:08x}",
        "tile_status": (code >> 8) & 255 if code & 0x40000000 else None,
        "device_status": code & 255 if code & 0x40000000 else None,
        "matching_output_bytes": len(outputs) - different,
        "total_output_bytes": len(outputs),
        "output_sha256": _sha(outputs),
        "expected_sha256": _sha(expected),
        "limitations": [
            "bitstream identity is reported by the external runner, not independently attested",
            "matching bytes do not establish cache coherence or a selected SoC execution trace",
            "the fixed launcher has no challenge input, so a copied prior dump cannot be ruled out",
        ],
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        print(json.dumps({"status": "integrity_failure", "reason": "fresh report path required"}),
              file=sys.stderr)
        return 2
    try:
        result = check(args.handoff, args.receipt, args.status, args.outputs, args.expected)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        result = {"schema": "atlas.external_model_dump_check.v1",
                  "status": "integrity_failure", "reason": str(error)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if result["status"] == "reported_match_unqualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
