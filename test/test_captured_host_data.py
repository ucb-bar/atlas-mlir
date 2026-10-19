"""Reject stale or misplaced constants before embedding a captured tile."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_captured_host_data import render


class CapturedHostDataTest(unittest.TestCase):
    def test_native_typed_multi_buffer_handoff_checks_verified_abi(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            compiled, capsule = root / "compiled", root / "capsule"
            compiled.mkdir()
            capsule.mkdir()
            program = b"\x73\x00\x00\x00"
            program_hash = hashlib.sha256(program).hexdigest()
            (compiled / "program.bin").write_bytes(program)
            (capsule / "program.bin").write_bytes(program)
            inputs = [0x90001800 + 0x800 * index for index in range(6)]
            outputs = [0x90011800, 0x90012000]
            plan = {
                "schema": "atlas.native_tile_execution_plan.v8",
                "shape": [32, 32], "dtype": "bf16",
                "input_address": inputs[0], "input_addresses": inputs.copy(),
                "input_payload_bytes": [1024] * 6,
                "output_address": outputs[0], "output_addresses": outputs.copy(),
                "output_bytes": 2048,
                "output_physical_layouts": ["bf16_pair_halves"] * 2,
                "output_numerical_policies": ["relu-bf16", "mxu-bf16"],
            }
            manifest = {
                "schema": "atlas.native_tile_compilation.v6",
                "engine": "merlin_native", "artifact_kind": "program",
                "compile_mode": "diagnostic", "qualification_scope": "diagnostic_only",
                "target_identity": f"atlas-rtl-{'0' * 40}:test-profile",
                "binary_sha256": program_hash, "request_digest": "1" * 64,
            }
            receipt = {
                "schema": "merlin.native_artifact_verification.v1",
                "status": "verified_artifacts", "engine": "merlin_native",
                "source_kind": "typed_request", "selected_record_checked": True,
                "artifact": str(compiled.resolve()), "binary_sha256": program_hash,
                "request_digest": manifest["request_digest"],
                "target_identity": manifest["target_identity"],
            }
            packed = {
                "schema": "atlas.boot-capsule.v1", "program_file": "program.bin",
                "program_sha256": program_hash,
                "dram": {"selected_rtl_revision": "0" * 40,
                         "regions": [
                             {"kind": "input", "address": 0x90001800,
                              "size_bytes": 11264},
                             {"kind": "output", "address": 0x90011800,
                              "size_bytes": 4096},
                         ]},
            }
            (capsule / "manifest.json").write_text(json.dumps(packed))
            verified = root / "verify.json"

            def write_verified() -> None:
                plan_bytes = json.dumps(plan).encode()
                (compiled / "execution_plan.json").write_bytes(plan_bytes)
                manifest["execution_plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
                manifest_bytes = json.dumps(manifest).encode()
                (compiled / "manifest.json").write_bytes(manifest_bytes)
                receipt["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
                verified.write_text(json.dumps(receipt))

            write_verified()
            header = render(compiled, capsule, native_verification=verified)
            self.assertIn("ATLAS_CAPTURED_INPUT_COUNT 6u", header)
            self.assertIn("ATLAS_CAPTURED_INPUT_ADDRESS_5 0x90004000u", header)
            self.assertIn("ATLAS_CAPTURED_OUTPUT_ADDRESS_1 0x90012000u", header)
            self.assertIn("ATLAS_CAPTURED_CONSTANT_COUNT 0u", header)
            self.assertNotIn("atlas_pack_runtime_input_f32", header)

            constant_address = 0x90004800
            constant = b"\x35\x3e" * 1024
            constant_path = "constants/constant-000.bin"
            (compiled / "constants").mkdir()
            (compiled / constant_path).write_bytes(constant)
            row = {
                "node_id": "attention_scale", "address": constant_address,
                "bytes": len(constant), "encoding": "bf16-le-raw",
                "physical_layout": "bf16_pair_halves", "path": constant_path,
                "sha256": hashlib.sha256(constant).hexdigest(),
            }
            plan["input_addresses"].append(constant_address)
            plan["input_payload_bytes"].append(2048)
            plan["constant_inputs"] = [row]
            manifest["constant_inputs"] = [row]
            manifest["stage_artifacts"] = {constant_path: row["sha256"]}
            receipt["checked_stage_artifacts"] = [constant_path]
            packed["dram"]["regions"][0]["size_bytes"] = 14336
            (capsule / "manifest.json").write_text(json.dumps(packed))
            write_verified()
            header = render(compiled, capsule, native_verification=verified)
            self.assertIn("ATLAS_CAPTURED_INPUT_COUNT 6u", header)
            self.assertIn("ATLAS_CAPTURED_CONSTANT_COUNT 1u", header)
            self.assertIn("0x90004800u, 2048u, atlas_constant_0", header)
            (compiled / constant_path).write_bytes(b"\x00" * len(constant))
            with self.assertRaisesRegex(ValueError, "constant payload differs"):
                render(compiled, capsule, native_verification=verified)
            (compiled / constant_path).write_bytes(constant)
            plan["input_addresses"].pop()
            plan["input_payload_bytes"].pop()
            plan.pop("constant_inputs")
            manifest.pop("constant_inputs")
            manifest.pop("stage_artifacts")
            receipt.pop("checked_stage_artifacts")
            packed["dram"]["regions"][0]["size_bytes"] = 11264
            (capsule / "manifest.json").write_text(json.dumps(packed))
            plan["input_addresses"][5] = plan["input_addresses"][4]
            write_verified()
            with self.assertRaisesRegex(ValueError, "overlapping physical buffers"):
                render(compiled, capsule, native_verification=verified)
            plan["input_addresses"][5] = inputs[5]
            plan["output_addresses"][1] += 0x800
            write_verified()
            with self.assertRaisesRegex(ValueError, "unsupported physical ABI"):
                render(compiled, capsule, native_verification=verified)
            plan["output_addresses"][1] = outputs[1]
            write_verified()
            receipt["request_digest"] = "2" * 64
            verified.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "verification does not match"):
                render(compiled, capsule, native_verification=verified)

    def test_combined_capture_binds_one_input_and_six_constants(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            compiled, capsule = root / "compiled", root / "capsule"
            (compiled / "constants").mkdir(parents=True)
            capsule.mkdir()
            program = b"checked program"
            program_hash = hashlib.sha256(program).hexdigest()
            (capsule / "program.bin").write_bytes(program)
            sizes = [1024, 1024, 1024, 1024, 2048, 1024, 1024]
            slots = []
            constants = []
            for index, size in enumerate(sizes):
                address = 0x90000000 + 2048 * index
                slots.append({"slot": index,
                              "source_kind": "input" if index == 0 else
                                             "generated" if index == 4 else "param",
                              "source_key": None if index in (0, 4) else f"weight.{index}",
                              "dram_address": address, "payload_bytes": size})
                if index:
                    data = bytes([index]) * size
                    relative = f"constants/slot-{index:02d}.bin"
                    (compiled / relative).write_bytes(data)
                    constants.append({"slot": index, "payload_file": relative,
                                      "payload_bytes": size,
                                      "payload_sha256": hashlib.sha256(data).hexdigest(),
                                      "source_key": slots[-1]["source_key"],
                                      "dram_address": address})
            imported = {
                "schema": "atlas.oot.captured_linalg_import.v1",
                "selected_rtl_revision": "0" * 40,
                "program": {"program_sha256": program_hash},
                "input_slots": slots,
                "output": {"dram_address": 0x90010000, "payload_bytes": 2048},
                "constants": constants,
            }
            packed = {
                "schema": "atlas.boot-capsule.v1", "program_sha256": program_hash,
                "program_file": "program.bin",
                "dram": {"selected_rtl_revision": "0" * 40,
                         "regions": [
                             {"kind": "input", "address": 0x90000000,
                              "size_bytes": 14336},
                             {"kind": "output", "address": 0x90010000,
                              "size_bytes": 2048},
                         ]},
            }
            (compiled / "import-manifest.json").write_text(json.dumps(imported))
            (capsule / "manifest.json").write_text(json.dumps(packed))
            header = render(compiled, capsule)
            self.assertIn("ATLAS_CAPTURED_INPUT_ADDRESS 0x90000000u", header)
            self.assertIn("ATLAS_CAPTURED_CONSTANT_COUNT 6u", header)
            self.assertIn("{0x90002000u, 2048u, atlas_constant_3}", header)

            omitted = imported["constants"].pop()
            (compiled / "import-manifest.json").write_text(json.dumps(imported))
            with self.assertRaisesRegex(ValueError, "declared constant slot"):
                render(compiled, capsule)
            imported["constants"].append(omitted)
            imported["outputs"] = [imported.pop("output"),
                                   {"index": 1, "dram_address": 0x90010800,
                                    "payload_bytes": 2048,
                                    "physical_layout": "bf16_pair_halves"}]
            packed["dram"]["regions"][1]["size_bytes"] = 4096
            packed["captured_output_count"] = 2
            (compiled / "import-manifest.json").write_text(json.dumps(imported))
            packed["captured_import_manifest_sha256"] = hashlib.sha256(
                (compiled / "import-manifest.json").read_bytes()).hexdigest()
            (capsule / "manifest.json").write_text(json.dumps(packed))
            two_outputs = render(compiled, capsule)
            self.assertIn("ATLAS_CAPTURED_OUTPUT_COUNT 2u", two_outputs)
            self.assertIn("ATLAS_CAPTURED_OUTPUT_ADDRESS_1 0x90010800u", two_outputs)
            self.assertIn("atlas_captured_outputs[ATLAS_CAPTURED_OUTPUT_COUNT]", two_outputs)
            imported["outputs"][1]["dram_address"] = 0x90011000
            (compiled / "import-manifest.json").write_text(json.dumps(imported))
            with self.assertRaisesRegex(ValueError, "changed after capsule packaging"):
                render(compiled, capsule)
            packed["captured_import_manifest_sha256"] = hashlib.sha256(
                (compiled / "import-manifest.json").read_bytes()).hexdigest()
            (capsule / "manifest.json").write_text(json.dumps(packed))
            with self.assertRaisesRegex(ValueError, "escape the packaged layout"):
                render(compiled, capsule)
            imported["outputs"][1]["dram_address"] = 0x90010800
            imported["output"] = imported.pop("outputs")[0]
            packed["dram"]["regions"][1]["size_bytes"] = 2048
            packed.pop("captured_output_count")
            packed.pop("captured_import_manifest_sha256")
            (capsule / "manifest.json").write_text(json.dumps(packed))
            imported["input_slots"].pop()
            (compiled / "import-manifest.json").write_text(json.dumps(imported))
            with self.assertRaisesRegex(ValueError, "declared constant slot"):
                render(compiled, capsule)
            imported["input_slots"].append(slots[-1])
            imported["constants"][0]["source_key"] = "wrong.weight"
            (compiled / "import-manifest.json").write_text(json.dumps(imported))
            with self.assertRaisesRegex(ValueError, "declared constant slot"):
                render(compiled, capsule)
            imported["constants"][0]["source_key"] = slots[1]["source_key"]
            imported["input_slots"][4]["payload_bytes"] = 1024
            (compiled / "import-manifest.json").write_text(json.dumps(imported))
            with self.assertRaisesRegex(ValueError, "declared constant slot"):
                render(compiled, capsule)

    def test_two_runtime_inputs_have_distinct_checked_slots(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            compiled, capsule = root / "compiled", root / "capsule"
            compiled.mkdir()
            capsule.mkdir()
            program = b"checked program"
            program_hash = hashlib.sha256(program).hexdigest()
            (capsule / "program.bin").write_bytes(program)
            imported = {
                "schema": "atlas.oot.captured_linalg_import.v1",
                "selected_rtl_revision": "0" * 40,
                "program": {"program_sha256": program_hash},
                "input_slots": [
                    {"source_kind": "input", "dram_address": address, "payload_bytes": 1024}
                    for address in (0x90000000, 0x90000800)
                ],
                "output": {"dram_address": 0x90010000, "payload_bytes": 2048},
                "constants": [],
            }
            packed = {
                "schema": "atlas.boot-capsule.v1", "program_sha256": program_hash,
                "program_file": "program.bin",
                "dram": {"selected_rtl_revision": "0" * 40,
                         "regions": [
                             {"kind": "input", "address": 0x90000000, "size_bytes": 4096},
                             {"kind": "output", "address": 0x90010000, "size_bytes": 2048},
                         ]},
            }
            (compiled / "import-manifest.json").write_text(json.dumps(imported))
            (capsule / "manifest.json").write_text(json.dumps(packed))
            header = render(compiled, capsule)
            self.assertIn("ATLAS_CAPTURED_INPUT_ADDRESS_0 0x90000000u", header)
            self.assertIn("ATLAS_CAPTURED_INPUT_ADDRESS_1 0x90000800u", header)
            self.assertIn("ATLAS_CAPTURED_CONSTANT_COUNT 0u", header)
            self.assertNotIn("atlas_captured_constants[", header)

            imported["input_slots"][1]["dram_address"] = 0x90000200
            (compiled / "import-manifest.json").write_text(json.dumps(imported))
            with self.assertRaisesRegex(ValueError, "escape"):
                render(compiled, capsule)

    def test_binds_exact_program_and_constants(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            compiled, capsule = root / "compiled", root / "capsule"
            (compiled / "constants").mkdir(parents=True)
            capsule.mkdir()
            data = bytes(range(32))
            (compiled / "constants/weight.bin").write_bytes(data)
            program_hash = hashlib.sha256(b"checked program").hexdigest()
            (capsule / "program.bin").write_bytes(b"checked program")
            source = {
                "schema": "atlas.oot.captured_linalg_import.v1",
                "selected_rtl_revision": "0" * 40,
                "program": {"program_sha256": program_hash},
                "input_slots": [{"source_kind": "input", "dram_address": 0x90000000,
                                 "payload_bytes": 1024}],
                "output": {"dram_address": 0x90010000, "payload_bytes": 2048},
                "constants": [{"payload_file": "constants/weight.bin",
                               "payload_bytes": len(data),
                               "payload_sha256": hashlib.sha256(data).hexdigest(),
                               "dram_address": 0x90000800}],
            }
            packed = {
                "schema": "atlas.boot-capsule.v1", "program_sha256": program_hash,
                "program_file": "program.bin",
                "dram": {"selected_rtl_revision": "0" * 40,
                         "regions": [{"kind": "input", "address": 0x90000000,
                                      "size_bytes": 4096},
                                     {"kind": "output", "address": 0x90010000,
                                      "size_bytes": 2048}]},
            }
            (compiled / "import-manifest.json").write_text(json.dumps(source))
            (capsule / "manifest.json").write_text(json.dumps(packed))
            header = render(compiled, capsule)
            self.assertIn("ATLAS_CAPTURED_INPUT_REGION_BYTES 4096u", header)
            self.assertIn("{0x90000800u, 32u, atlas_constant_0}", header)

            (compiled / "constants/weight.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "constant payload differs"):
                render(compiled, capsule)
            (compiled / "constants/weight.bin").write_bytes(data)
            (capsule / "program.bin").write_bytes(b"different program")
            with self.assertRaisesRegex(ValueError, "boot capsule program differs"):
                render(compiled, capsule)
            (capsule / "program.bin").write_bytes(b"checked program")
            source["program"]["program_sha256"] = "f" * 64
            (compiled / "import-manifest.json").write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "different instruction streams"):
                render(compiled, capsule)


if __name__ == "__main__":
    unittest.main()
