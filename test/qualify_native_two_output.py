"""Evaluator-only selected-core check of a Merlin-native two-output program.

The compiler sees only the public typed request and ABI. This evaluator supplies
runtime input panels and independently specified exact BF16 output bits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(phase: int) -> tuple[bytes, dict[str, bytes]]:
    # (input, log2, sqrt). All entries are exact BF16 values. Changing the
    # output labels or dropping either calculation changes observable bytes.
    anchors = (
        (0x3F80, 0x0000, 0x3F80),  # 1 -> 0, 1
        (0x4080, 0x4000, 0x4000),  # 4 -> 2, 2
        (0x4280, 0x40C0, 0x4100),  # 64 -> 6, 8
        (0x4380, 0x4100, 0x4180),  # 256 -> 8, 16
    )
    selected = [anchors[(index * 7 + index // 16 + phase) % len(anchors)]
                for index in range(1024)]
    def pack(column: int) -> bytes:
        return b"".join(struct.pack("<H", row[column]) for row in selected)

    return pack(0), {"log2": pack(1), "sqrt": pack(2)}


def load_program(directory: Path, source_revision: str) -> tuple[tuple[int, ...], dict]:
    manifest_path = directory / "manifest.json"
    plan_path = directory / "execution_plan.json"
    program_path = directory / "program.bin"
    manifest = json.loads(manifest_path.read_text())
    plan = json.loads(plan_path.read_text())
    raw = program_path.read_bytes()
    if (manifest.get("engine") != "merlin_native"
            or manifest.get("status") != "diagnostic_program"
            or manifest.get("source_revision") != source_revision):
        raise ValueError("expected a selected-source Merlin-native diagnostic program")
    if (not raw or len(raw) % 4
            or digest(program_path) != manifest.get("binary_sha256")
            or digest(program_path) != plan.get("program", {}).get("sha256")
            or digest(plan_path) != manifest.get("execution_plan_sha256")
            or plan.get("program", {}).get("word_count") != len(raw) // 4):
        raise ValueError("program or execution plan identity mismatch")
    inputs = plan.get("inputs")
    outputs = plan.get("outputs")
    if (not isinstance(inputs, list) or len(inputs) != 1
            or inputs[0].get("source") != "source"
            or inputs[0].get("byte_length") != 2048
            or not inputs[0].get("preserve")
            or not isinstance(outputs, list) or len(outputs) != 2
            or {item.get("source") for item in outputs} != {"log2", "sqrt"}
            or any(item.get("byte_length") != 2048 for item in outputs)):
        raise ValueError("unexpected two-output physical ABI")
    spans = [(item["byte_address"], item["byte_address"] + 2048)
             for item in [*inputs, *outputs]]
    spans.append((0x90006000, 0x90006020))  # untouched memory guard
    if any(left[0] < right[1] and right[0] < left[1]
           for index, left in enumerate(spans) for right in spans[index + 1:]):
        raise ValueError("physical ABI overlaps another buffer or guard")
    return struct.unpack("<" + "I" * (len(raw) // 4), raw), plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-model", type=Path, required=True)
    parser.add_argument("--arc-state", type=Path, required=True)
    parser.add_argument("--modelir", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-state-sha256", required=True)
    parser.add_argument("--expected-modelir-driver-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output path must be fresh")
    model = args.arc_model.resolve(strict=True)
    state = args.arc_state.resolve(strict=True)
    modelir = args.modelir.resolve(strict=True)
    driver = modelir / "mlc/backends/cosim_atlas.py"
    if (digest(model) != args.expected_model_sha256
            or digest(state) != args.expected_state_sha256
            or digest(driver) != args.expected_modelir_driver_sha256):
        parser.error("selected model, state, or ModeLIR driver identity mismatch")
    words, plan = load_program(args.program.resolve(strict=True), args.expected_source_revision)

    old_cwd = Path.cwd()
    sys.path.insert(0, str(modelir))
    os.chdir(modelir)  # ModeLIR resolves its interface cache relative to here.
    try:
        from mlc.backends import cosim_atlas

        original_core = cosim_atlas.CosimCore

        class SelectedCore(original_core):
            def peek(self, name: str) -> int:
                if name == "io_halted" and "scalar/halt_now" in self._S:
                    name = "scalar/halt_now"
                return super().peek(name)

            def poke(self, name: str, value: int) -> None:
                if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size") and name not in self._S:
                    return
                super().poke(name, value)

        cosim_atlas.CosimCore = SelectedCore
        try:
            observations = []
            for phase in (0, 5):
                source, expected = panel(phase)
                preload = [(plan["inputs"][0]["byte_address"], source)]
                preload.extend((item["byte_address"], b"\xA5" * 2048)
                               for item in plan["outputs"])
                preload.append((0x90006000, b"\x5A" * 32))
                run = cosim_atlas.run_program(model, state, words, preload=preload,
                                              max_cycles=8000)
                actual = {
                    item["source"]: run.slave.captured(item["byte_address"], 2048)
                    for item in plan["outputs"]
                }
                output_checks = {name: data == expected[name]
                                 for name, data in actual.items()}
                input_preserved = (run.slave.captured(plan["inputs"][0]["byte_address"], 2048)
                                   == source)
                guard_preserved = run.slave.captured(0x90006000, 32) == b"\x5A" * 32
                observations.append({
                    "phase": phase, "passed": run.halted and all(output_checks.values())
                    and input_preserved and guard_preserved,
                    "halted": run.halted, "cycles": run.cycles,
                    "reads": run.reads, "writes": run.writes,
                    "output_checks": output_checks,
                    "matching_output_bytes": {
                        name: sum(left == right for left, right in zip(data, expected[name]))
                        for name, data in actual.items()
                    },
                    "input_preserved": input_preserved,
                    "guard_preserved": guard_preserved,
                    "input_sha256": hashlib.sha256(source).hexdigest(),
                    "actual_output_sha256": {
                        name: hashlib.sha256(data).hexdigest()
                        for name, data in actual.items()
                    },
                    "expected_output_sha256": {
                        name: hashlib.sha256(data).hexdigest()
                        for name, data in expected.items()
                    },
                })
        finally:
            cosim_atlas.CosimCore = original_core
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(modelir))

    receipt = {
        "schema": "atlas.native_two_output_selected_core_bounded.v1",
        "model_sha256": digest(model), "state_sha256": digest(state),
        "modelir_driver_sha256": digest(driver),
        "program_sha256": plan["program"]["sha256"],
        "checker_sha256": digest(Path(__file__)),
        "selected_source_revision": args.expected_source_revision,
        "passed": sum(item["passed"] for item in observations),
        "total": len(observations), "observations": observations,
        "scope": "bounded standalone AtlasCore with diagnostic static delays; no integrated SoC or model invocation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": receipt["passed"], "total": receipt["total"],
                      "receipt": str(args.out)}, sort_keys=True))
    return 0 if receipt["passed"] == receipt["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
