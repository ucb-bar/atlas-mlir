"""Evaluator-only bounded AtlasCore check of Merlin-native program artifacts.

This script does not enter the compiler package. It reads compiled binaries and
supplies runtime tensors plus independent public expected results to ModeLIR.
The ARC model and state manifest must be selected separately by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

from test_vpu_exp2_reference import _panel as exp2_panel
from test_vpu_sqrt_reference import _panel as sqrt_panel

CASES = ("exp2", "sqrt", "log2", "minmax")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs_and_expected(case: str, phase: int) -> tuple[dict[str, bytes], bytes]:
    if case == "exp2":
        source, expected = exp2_panel(phase)
        return {"source": source}, expected
    if case == "sqrt":
        source, expected = sqrt_panel(phase)
        return {"source": source}, expected
    if case == "log2":
        exponents = tuple(range(-4, 5))
        values = [exponents[(i * 7 + i // 16 + phase) % len(exponents)]
                  for i in range(1024)]
        source = b"".join(struct.pack("<H", (127 + value) << 7)
                          for value in values)
        expected = b"".join(struct.pack("<H", int.from_bytes(
            struct.pack(">f", float(value))[:2], "big")) for value in values)
        return {"source": source}, expected
    if case == "minmax":
        inputs = {
            "activation_a": bytes([0x38 if phase == 0 else 0xB8]) * 1024,
            "weight_a": bytes([0x38]) * 1024,
            "activation_b": bytes([0x40]) * 1024,
            "weight_b": bytes([0x38]) * 1024,
        }
        # Thirty-two exact +1/-1 or +2 E4M3 products per output. The
        # selected BF16 rounding points cannot change these exact sums.
        return inputs, struct.pack("<H", 0x4200 if phase == 0 else 0xC200) * 1024
    raise ValueError(f"unknown public case: {case}")


def load_artifact(case: str, directory: Path) -> tuple[tuple[int, ...], dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    plan_path = directory / "execution_plan.json"
    plan = json.loads(plan_path.read_text())
    program = directory / "program.bin"
    raw = program.read_bytes()
    if manifest.get("engine") != "merlin_native" or manifest.get("status") != "diagnostic_program":
        raise ValueError(f"{case}: expected a successful Merlin-native diagnostic")
    if len(raw) % 4 or digest(program) != manifest["binary_sha256"]:
        raise ValueError(f"{case}: invalid program identity")
    if digest(program) != plan["program"]["sha256"]:
        raise ValueError(f"{case}: plan/program identity mismatch")
    if digest(plan_path) != manifest["execution_plan_sha256"]:
        raise ValueError(f"{case}: changed execution plan")
    if len(plan["outputs"]) != 1 or plan["outputs"][0]["byte_length"] != 2048:
        raise ValueError(f"{case}: unexpected bounded output ABI")
    return struct.unpack("<" + "I" * (len(raw) // 4), raw), plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-model", type=Path, required=True)
    parser.add_argument("--arc-state", type=Path, required=True)
    parser.add_argument("--modelir", type=Path, required=True)
    parser.add_argument("--program", action="append", required=True,
                        help="case=compiled artifact directory; supply all four cases")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = {}
    for item in args.program:
        case, separator, path = item.partition("=")
        if not separator or case not in CASES or case in selected:
            parser.error(f"invalid or duplicate --program {item!r}")
        selected[case] = Path(path).resolve(strict=True)
    if set(selected) != set(CASES):
        parser.error(f"supply exactly {', '.join(CASES)}")
    if args.out.exists():
        parser.error("output path must be fresh")

    model = args.arc_model.resolve(strict=True)
    state = args.arc_state.resolve(strict=True)
    modelir = args.modelir.resolve(strict=True)
    sys.path.insert(0, str(modelir))
    old_cwd = Path.cwd()
    os.chdir(modelir)  # ModeLIR bootstrap resolves its interface cache here.
    try:
        from mlc.backends import cosim_atlas

        original_core = cosim_atlas.CosimCore

        class Core(original_core):
            def peek(self, name: str) -> int:
                if name == "io_halted" and "scalar/halt_now" in self._S:
                    name = "scalar/halt_now"
                return super().peek(name)

            def poke(self, name: str, value: int) -> None:
                if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size") and name not in self._S:
                    return
                super().poke(name, value)

        cosim_atlas.CosimCore = Core
        try:
            observations = []
            for case in CASES:
                words, plan = load_artifact(case, selected[case])
                output = plan["outputs"][0]
                for phase in (0, 5):
                    sources, expected = inputs_and_expected(case, phase)
                    preload = [(item["byte_address"], sources[item["source"]])
                               for item in plan["inputs"]]
                    if any(len(data) != item["byte_length"]
                           for item, (_, data) in zip(plan["inputs"], preload)):
                        raise ValueError(f"{case}: input ABI length mismatch")
                    preload.extend(((output["byte_address"], b"\xA5" * 2048),
                                    (0x90004000, b"\x5A" * 32)))
                    run = cosim_atlas.run_program(model, state, words, preload=preload,
                                                  max_cycles=8000)
                    actual = run.slave.captured(output["byte_address"], 2048)
                    preserved_inputs = all(
                        run.slave.captured(item["byte_address"], item["byte_length"])
                        == sources[item["source"]] for item in plan["inputs"])
                    preserved_guard = run.slave.captured(0x90004000, 32) == b"\x5A" * 32
                    passed = run.halted and actual == expected and preserved_inputs and preserved_guard
                    observations.append({
                        "case": case, "phase": phase, "passed": passed,
                        "halted": run.halted, "cycles": run.cycles,
                        "reads": run.reads, "writes": run.writes,
                        "matching_bytes": sum(a == b for a, b in zip(actual, expected)),
                        "expected_bytes": len(expected),
                        "preserved_inputs": preserved_inputs,
                        "preserved_guard": preserved_guard,
                        "output_sha256": hashlib.sha256(actual).hexdigest(),
                        "input_sha256": {name: hashlib.sha256(data).hexdigest()
                                         for name, data in sources.items()},
                        "expected_sha256": hashlib.sha256(expected).hexdigest(),
                        "program_sha256": plan["program"]["sha256"],
                        "manifest_sha256": digest(selected[case] / "manifest.json"),
                    })
        finally:
            cosim_atlas.CosimCore = original_core
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(modelir))
    receipt = {
        "schema": "atlas.native_support_selected_core_bounded.v1",
        "model_sha256": digest(model), "state_sha256": digest(state),
        "checker_sha256": digest(Path(__file__)),
        "reference_source_sha256": {
            "exp2": digest(Path(__file__).with_name("test_vpu_exp2_reference.py")),
            "sqrt": digest(Path(__file__).with_name("test_vpu_sqrt_reference.py")),
        },
        "modelir_driver_sha256": digest(modelir / "mlc/backends/cosim_atlas.py"),
        "passed": sum(row["passed"] for row in observations),
        "total": len(observations), "observations": observations,
        "scope": "bounded standalone AtlasCore; diagnostic delays; no integrated SoC or model invocation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": receipt["passed"], "total": receipt["total"],
                      "receipt": str(args.out)}, sort_keys=True))
    return 0 if receipt["passed"] == receipt["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
