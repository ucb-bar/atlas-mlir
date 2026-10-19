"""Check an externally captured graph mailbox against a public Atlas fixture.

The result describes reported bytes. It does not attest the bitstream or prove
that the selected SoC executed the program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _target_f32(physical: bytes, outputs: int) -> bytes:
    if len(physical) != outputs * 2048:
        raise ValueError("expected BF16 output size differs from the handoff")
    decoded = bytearray(outputs * 4096)
    for output in range(outputs):
        for row in range(32):
            for column in range(32):
                source = (output * 2048 + (column // 16) * 1024 +
                          row * 32 + (column % 16) * 2)
                bits = int.from_bytes(physical[source:source + 2], "little")
                struct.pack_into("<I", decoded,
                                 output * 4096 + (row * 32 + column) * 4,
                                 bits << 16)
    return bytes(decoded)


def check(*, handoff_status: Path, execution_plan: Path, elf: Path,
          program: Path, dump: Path, inputs: Path, expected: Path,
          request_id: int) -> dict:
    status = _read_json(handoff_status)
    plan = _read_json(execution_plan)
    elf_bytes = elf.read_bytes()
    program_bytes = program.read_bytes()
    outputs = status.get("output_count")
    allowed_handoff = (
        (status.get("schema") == "atlas.diagnostic_graph_mailbox_handoff.v1" and
         status.get("status") == "diagnostic_unbound" and
         status.get("bitstream_bound") is False) or
        (status.get("schema") == "atlas.graph_mailbox_handoff.v2" and
         status.get("status") == "platform_map_checked_execution_unqualified" and
         status.get("bitstream_bound") is True and
         isinstance(status.get("bitstream_identity"), str) and
         bool(status["bitstream_identity"]) and
         status.get("cache_policy") == "callback_supplied_unqualified")
    )
    if (not allowed_handoff
            or status.get("engine") != "merlin_native"
            or status.get("elf_sha256") != _sha(elf_bytes)
            or status.get("program_sha256") != _sha(program_bytes)
            or status.get("input_count") != 6
            or type(outputs) is not int or not 1 <= outputs <= 16
            or plan.get("shape") != [32, 32]
            or plan.get("dtype") != "bf16"
            or plan.get("output_physical_layouts") != ["bf16_pair_halves"] * outputs
            or plan.get("output_bytes") != 2048):
        raise ValueError("graph handoff, program, or physical output layout differs")
    offsets = status.get("mailbox_offsets")
    expected_offsets = {
        "request_id": 0, "response_id": 4, "status": 8,
        "guard_before": 16, "inputs": 20, "outputs": 24596,
        "guard_after": 24596 + outputs * 4096,
    }
    size = 24600 + outputs * 4096
    if offsets != expected_offsets or status.get("mailbox_bytes") != size:
        raise ValueError("graph mailbox layout differs from the selected signature")
    if type(request_id) is not int or not 0 < request_id < 2**32:
        raise ValueError("request id must be a nonzero 32-bit integer")
    captured = dump.read_bytes()
    source = inputs.read_bytes()
    expected_bytes = expected.read_bytes()
    target = _target_f32(expected_bytes, outputs)
    if len(captured) != size or len(source) != 6 * 4096:
        raise ValueError("graph mailbox or source input size differs")
    request, response, device_status = struct.unpack_from("<III", captured, 0)
    guard_before = struct.unpack_from("<I", captured, 16)[0]
    guard_after = struct.unpack_from("<I", captured, expected_offsets["guard_after"])[0]
    observed = captured[expected_offsets["outputs"]:expected_offsets["guard_after"]]
    match_count = sum(left == right for left, right in zip(observed, target, strict=True))
    source_preserved = captured[20:24596] == source
    guard_preserved = guard_before == guard_after == 0x76A31C5B
    complete = request == response == request_id
    passed = (complete and device_status == 0 and source_preserved and
              guard_preserved and match_count == len(target))
    return {
        "schema": "atlas.external_graph_mailbox_check.v1",
        "status": "reported_match_unqualified" if passed else "execution_failure",
        "scope": "reported mailbox bytes for one graph invocation",
        "request_id": request_id,
        "observed_request_id": request,
        "observed_response_id": response,
        "status_word": f"0x{device_status:08x}",
        "source_inputs_preserved": source_preserved,
        "guards_preserved": guard_preserved,
        "matching_output_bytes": match_count,
        "total_output_bytes": len(target),
        "output_count": outputs,
        "elf_sha256": _sha(elf_bytes),
        "program_sha256": _sha(program_bytes),
        "mailbox_sha256": _sha(captured),
        "inputs_sha256": _sha(source),
        "expected_sha256": _sha(expected_bytes),
        "execution_plan_sha256": _sha(execution_plan.read_bytes()),
        "limitation": ("reported bytes do not attest bitstream identity or DMA/cache "
                       "behavior; a reused request id can return an earlier response"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("handoff-status", "execution-plan", "elf", "program",
                   "dump", "inputs", "expected", "out"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--request-id", required=True, type=int)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("fresh report path required")
    try:
        result = check(
            handoff_status=args.handoff_status, execution_plan=args.execution_plan,
            elf=args.elf, program=args.program, dump=args.dump, inputs=args.inputs,
            expected=args.expected, request_id=args.request_id,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        result = {"schema": "atlas.external_graph_mailbox_check.v1",
                  "status": "integrity_failure", "reason": str(error)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)},
                     sort_keys=True))
    return 0 if result["status"] == "reported_match_unqualified" else 2


if __name__ == "__main__":
    sys.exit(main())
