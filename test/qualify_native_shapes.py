"""Evaluator-only exact-anchor check for the public Atlas shape seed roster.

Runtime inputs and expected output bits are created here, after native
compilation. Neither this checker nor its results enter the compiler package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

from merlin.semantic_compiler.model import KernelRequest

from atlas_native_support.mxu0 import Mxu0Contract
from atlas_native_support.mxu0_launch import prepare_mxu0_launch
from atlas_native_support.mxu0_program_set import prepare_mxu0_program_set_launch


SHAPES = (
    (32, 32, 32), (64, 32, 96), (32, 64, 32), (64, 64, 64),
    (31, 32, 33), (33, 31, 65), (1, 65, 7), (7, 1, 33),
)
GUARD_ADDRESS = 0x9000F000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(m: int, k: int, n: int, phase: int) -> tuple[dict[str, bytes], bytes]:
    """Use E4M3 ±1 and zero; every active dot has an odd, nonzero exact sum."""
    activation = [
        1 if (row * 11 + index * 7 + phase) % 5 < 3 else -1
        for row in range(m) for index in range(k)
    ]
    weight = [
        0 if k % 2 == 0 and index == k - 1
        else 1 if (column * 13 + index * 3 + phase) % 7 < 4 else -1
        for column in range(n) for index in range(k)
    ]
    inputs = {
        "activation": bytes(0x38 if value == 1 else 0xB8 for value in activation),
        "weight": bytes(0x38 if value == 1 else 0xB8 if value == -1 else 0
                        for value in weight),
    }
    expected = bytearray()
    for row in range(m):
        for column in range(n):
            total = sum(
                activation[row * k + index] * weight[column * k + index]
                for index in range(k)
            )
            if total == 0 or abs(total) > 65:
                raise ValueError("exact anchor panel lost its nonzero finite domain")
            # Integer sums in this range are exact in binary32 and BF16.
            expected.extend(struct.pack(">f", float(total))[:2][::-1])
    return inputs, bytes(expected)


def decode_pair_tiles(
    physical: list[bytes], m: int, n: int, rows: int, columns: int,
) -> bytes:
    """Independently map physical BF16 column halves into source row order."""
    per_row = (n + columns - 1) // columns
    expected_tiles = ((m + rows - 1) // rows) * per_row
    if len(physical) != expected_tiles or columns % 2:
        raise ValueError("compiled output tile count differs from source shape")
    result = bytearray(m * n * 2)
    half = columns // 2
    for tile_index, payload in enumerate(physical):
        if len(payload) != rows * columns * 2:
            raise ValueError("compiled BF16 pair has wrong physical size")
        row_start = (tile_index // per_row) * rows
        column_start = (tile_index % per_row) * columns
        for row in range(min(rows, m - row_start)):
            for column in range(min(columns, n - column_start)):
                physical_index = ((column // half) * rows * half
                                  + row * half + column % half)
                logical_index = (row_start + row) * n + column_start + column
                result[2 * logical_index:2 * logical_index + 2] = (
                    payload[2 * physical_index:2 * physical_index + 2]
                )
    return bytes(result)


def source_request(artifact: Path) -> KernelRequest:
    path = artifact / "source_request.json"
    if not path.is_file():
        path = artifact / "request.json"
    return KernelRequest.from_record(json.loads(path.read_text()))


def _check_source(request: KernelRequest, shape: tuple[int, int, int], target: str) -> None:
    m, k, n = shape
    if (
        request.target_identity != target or request.lowering_policy != "strict-native"
        or len(request.nodes) != 3 or [node.op for node in request.nodes]
        != ["input", "input", "contraction"]
        or request.nodes[0].type.shape != (m, k)
        or request.nodes[1].type.shape != (n, k)
        or request.nodes[2].type.shape != (m, n)
        or request.outputs != (request.nodes[2].id,)
    ):
        raise ValueError("compiled source differs from required public shape and target")


def _run_one(cosim_atlas, model: Path, state: Path, program: Path,
             expected_hash: str, preloads: tuple[tuple[int, bytes], ...],
             output_rows: list[dict]) -> tuple[list[bytes], dict]:
    raw = program.read_bytes()
    if not raw or len(raw) % 4 or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("compiled program identity changed")
    spans = [(address, address + len(data)) for address, data in preloads]
    spans.extend((row["byte_address"], row["byte_address"] + row["byte_length"])
                 for row in output_rows)
    if any(low < GUARD_ADDRESS + 32 and GUARD_ADDRESS < high for low, high in spans):
        raise ValueError("memory guard overlaps a compiled buffer")
    loads = list(preloads)
    loads.extend((row["byte_address"], b"\xA5" * row["byte_length"])
                 for row in output_rows)
    loads.append((GUARD_ADDRESS, b"\x5A" * 32))
    words = struct.unpack("<" + "I" * (len(raw) // 4), raw)
    run = cosim_atlas.run_program(model, state, words, preload=loads, max_cycles=8000)
    preserved = all(run.slave.captured(address, len(data)) == data
                    for address, data in preloads)
    guard = run.slave.captured(GUARD_ADDRESS, 32) == b"\x5A" * 32
    outputs = [run.slave.captured(row["byte_address"], row["byte_length"])
               for row in output_rows]
    return outputs, {
        "halted": run.halted, "cycles": run.cycles,
        "reads": run.reads, "writes": run.writes,
        "preserved_inputs": preserved, "preserved_guard": guard,
        "program_sha256": expected_hash,
    }


def evaluate_case(cosim_atlas, model: Path, state: Path, artifact: Path,
                  shape: tuple[int, int, int], phase: int,
                  contract: Mxu0Contract) -> dict:
    source = source_request(artifact)
    _check_source(source, shape, contract.identity)
    m, k, n = shape
    runtime_inputs, expected = panel(m, k, n, phase)
    manifest = json.loads((artifact / "manifest.json").read_text())
    if manifest.get("engine") != "merlin_native":
        raise ValueError("shape seed was not compiled by Merlin native")
    physical: list[bytes] = []
    launch_records = []
    if manifest.get("artifact_kind") == "program_set":
        launches = prepare_mxu0_program_set_launch(
            artifact, runtime_inputs, expected_request_digest=source.digest(),
        )
        for launch in launches:
            outputs, record = _run_one(
                cosim_atlas, model, state, launch.program, launch.program_sha256,
                launch.preloads, [{"byte_address": launch.output_address,
                                   "byte_length": launch.output_length}],
            )
            physical.extend(outputs)
            launch_records.append(record)
    else:
        preloads = prepare_mxu0_launch(artifact, runtime_inputs)
        plan = json.loads((artifact / "execution_plan.json").read_text())
        outputs, record = _run_one(
            cosim_atlas, model, state, artifact / "program.bin",
            manifest["binary_sha256"], preloads, plan["outputs"],
        )
        physical.extend(outputs)
        launch_records.append(record)
    geometry = contract.record["geometry"]
    actual = decode_pair_tiles(
        physical, m, n, geometry["array_rows"], geometry["array_cols"],
    )
    passed = (actual == expected and all(
        row["halted"] and row["preserved_inputs"] and row["preserved_guard"]
        for row in launch_records
    ))
    return {
        "shape": [m, k, n], "phase": phase, "passed": passed,
        "artifact_kind": manifest.get("artifact_kind", "program"),
        "launches": launch_records,
        "matching_bytes": sum(left == right for left, right in zip(actual, expected)),
        "expected_bytes": len(expected),
        "actual_sha256": hashlib.sha256(actual).hexdigest(),
        "expected_sha256": hashlib.sha256(expected).hexdigest(),
        "source_request_digest": source.digest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-model", type=Path, required=True)
    parser.add_argument("--arc-state", type=Path, required=True)
    parser.add_argument("--modelir", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-state-sha256", required=True)
    parser.add_argument("--expected-modelir-driver-sha256", required=True)
    cases = parser.add_mutually_exclusive_group(required=True)
    cases.add_argument("--case", action="append",
                       help="MxKxN=compiled artifact directory; supply all eight seeds")
    cases.add_argument("--compiled-root", type=Path,
                       help="root produced by compile_native_shape_seeds.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output path must be fresh")
    selected = {}
    required = {"x".join(map(str, shape)): shape for shape in SHAPES}
    if args.compiled_root is not None:
        root = args.compiled_root.resolve(strict=True)
        selected = {name: (root / name / "compiled").resolve(strict=True)
                    for name in required}
    else:
        for item in args.case:
            name, marker, path = item.partition("=")
            if not marker or name not in required or name in selected:
                parser.error(f"invalid or duplicate --case {item!r}")
            selected[name] = Path(path).resolve(strict=True)
    if set(selected) != set(required):
        parser.error("all eight predeclared shape seeds are required")
    model, state, modelir = (path.resolve(strict=True)
                             for path in (args.arc_model, args.arc_state, args.modelir))
    driver = modelir / "mlc/backends/cosim_atlas.py"
    if (digest(model) != args.expected_model_sha256
            or digest(state) != args.expected_state_sha256
            or digest(driver) != args.expected_modelir_driver_sha256):
        parser.error("selected model, state, or ModeLIR driver identity mismatch")
    contract = Mxu0Contract.load()
    old_cwd = Path.cwd()
    sys.path.insert(0, str(modelir))
    os.chdir(modelir)
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
            observations = [
                evaluate_case(cosim_atlas, model, state, selected[name], shape, phase,
                              contract)
                for name, shape in required.items() for phase in (0, 5)
            ]
        finally:
            cosim_atlas.CosimCore = original_core
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(modelir))
    receipt = {
        "schema": "atlas.native_shape_seeds_selected_core_bounded.v1",
        "checker_sha256": digest(Path(__file__)),
        "model_sha256": digest(model), "state_sha256": digest(state),
        "modelir_driver_sha256": digest(driver),
        "selected_source_revision": contract.record["rtl_revision"],
        "passed": sum(row["passed"] for row in observations),
        "total": len(observations), "observations": observations,
        "scope": "exact E4M3 anchor panels on standalone AtlasCore with diagnostic delays; no integrated SoC or model invocation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": receipt["passed"], "total": receipt["total"],
                      "receipt": str(args.out)}, sort_keys=True))
    return 0 if receipt["passed"] == receipt["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
