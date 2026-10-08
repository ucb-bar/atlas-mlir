"""The launch handoff checks a capsule before binding each runtime call."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/atlas_launch_plan.py"
LAYOUT = ROOT / "test/examples/vpu_square_mailbox_layout.json"
REVISION = "0079c0541111197741a231c002e3843fa6f545b2"
CODE = struct.pack("<II", 0x00000073, 0x00008067)
sys.path.insert(0, str(ROOT / "tools"))
from atlas_boot_pack import validate_layout
from atlas_launch_plan import read_launch


def capsule(root: pathlib.Path) -> pathlib.Path:
    path = root / "capsule"
    path.mkdir()
    layout = validate_layout(json.loads(LAYOUT.read_text()))
    call = layout["call"]
    manifest = {
        "schema": "atlas.boot-capsule.v1",
        "checked_atlas_source_word_match": True,
        "program_mailbox_binding_proved_by_packer": False,
        "dram": layout,
        "program_file": "program.bin",
        "program_sha256": hashlib.sha256(CODE).hexdigest(),
        "program_words": 2,
        "entry_symbol": "atlas_program",
        "entry_pc_word": 0,
        "imem_tl_byte_base": 0x20000,
        "imem_capacity_bytes": 0x20000,
        "start_csr_tl_byte_address": 0x18,
        "start_csr_value": 1,
        "completion": {"kind": "ecall_halt", "ecall_pc_word": 0,
                       "unreachable_llvm_ret_pc_word": 1},
        "arguments": [{"name": "input", "kind": "runtime_dram_pointer",
                       "mailbox_offset_bytes": call["input_pointer_offset_bytes"],
                       "region": call["input_region"], "size_bytes": call["tensor_bytes"]}],
        "returns": [{"name": "output", "kind": "runtime_dram_pointer",
                     "mailbox_offset_bytes": call["output_pointer_offset_bytes"],
                     "region": call["output_region"], "size_bytes": call["tensor_bytes"]}],
    }
    (path / "program.bin").write_bytes(CODE)
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def invoke(path: pathlib.Path, out: pathlib.Path, input_address: str = "0x90002000",
           output_address: str = "0x90006000", revision: str = REVISION):
    return subprocess.run(
        [sys.executable, str(TOOL), "--capsule", str(path),
         "--input-address", input_address, "--output-address", output_address,
         "--expected-rtl-revision", revision, "--out", str(out)],
        capture_output=True, text=True)


def verify(path: pathlib.Path, launch: pathlib.Path):
    return subprocess.run(
        [sys.executable, str(TOOL), "--capsule", str(path),
         "--verify-launch", str(launch), "--expected-rtl-revision", REVISION],
        capture_output=True, text=True)


class LaunchPlanTest(unittest.TestCase):
    def test_two_calls_share_program_and_bind_different_mailboxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            package = capsule(root)
            first, second = root / "first", root / "second"
            self.assertEqual(invoke(package, first).returncode, 0)
            self.assertEqual(invoke(package, second, "0x90003000", "0x90007000").returncode, 0)
            first_plan = json.loads((first / "launch.json").read_text())
            second_plan = json.loads((second / "launch.json").read_text())
            self.assertEqual(first_plan["program_sha256"], second_plan["program_sha256"])
            self.assertNotEqual(first_plan["mailbox_sha256"], second_plan["mailbox_sha256"])
            self.assertEqual(struct.unpack_from("<II", (first / "mailbox.bin").read_bytes()),
                             (0x90002000, 0x90006000))
            self.assertEqual(struct.unpack_from("<II", (second / "mailbox.bin").read_bytes()),
                             (0x90003000, 0x90007000))
            self.assertEqual(read_launch(package, first, REVISION)[1], CODE)
            self.assertEqual(read_launch(package, second, REVISION)[2],
                             (second / "mailbox.bin").read_bytes())
            self.assertEqual(json.loads(verify(package, first).stdout)["status"], "PASS")
            self.assertFalse(first_plan["program_mailbox_binding_proved"])
            self.assertNotEqual(invoke(package, first).returncode, 0)

    def test_loader_rechecks_artifacts_after_launch_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            package = capsule(root)
            launch = root / "call"
            self.assertEqual(invoke(package, launch).returncode, 0)
            descriptor = (launch / "mailbox.bin").read_bytes()
            (launch / "mailbox.bin").write_bytes(b"\x00" * len(descriptor))
            with self.assertRaisesRegex(ValueError, "mailbox bytes differ"):
                read_launch(package, launch, REVISION)
            self.assertNotEqual(verify(package, launch).returncode, 0)
            (launch / "mailbox.bin").write_bytes(descriptor)
            plan_path = launch / "launch.json"
            plan = json.loads(plan_path.read_text())
            altered = copy.deepcopy(plan)
            altered["start_csr_value"] = 2
            plan_path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "launch plan differs"):
                read_launch(package, launch, REVISION)
            plan_path.write_text(json.dumps(plan))
            (package / "program.bin").write_bytes(b"\x00" * len(CODE))
            with self.assertRaisesRegex(ValueError, "program hash differs"):
                read_launch(package, launch, REVISION)
            (package / "program.bin").write_bytes(CODE)
            manifest_path = package / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["scope"] = "changed after launch preparation"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "launch plan differs"):
                read_launch(package, launch, REVISION)

    def test_corruption_revision_and_bad_pointer_fail_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            package = capsule(root)
            cases = [
                (lambda: (package / "program.bin").write_bytes(CODE[:-1]), "program size"),
                (lambda: (package / "program.bin").write_bytes(b"\x00" * len(CODE)), "program hash"),
            ]
            for index, (mutate, message) in enumerate(cases):
                mutate()
                out = root / f"bad_code_{index}"
                result = invoke(package, out)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(out.exists())
                (package / "program.bin").write_bytes(CODE)
            for index, args in enumerate((("0x90002001", "0x90006000", REVISION),
                                          ("0x90002000", "0x90006000", "f" * 40))):
                out = root / f"bad_call_{index}"
                self.assertNotEqual(invoke(package, out, *args).returncode, 0)
                self.assertFalse(out.exists())
            manifest_path = package / "manifest.json"
            original = json.loads(manifest_path.read_text())
            for index, (field, changed) in enumerate((("start_csr_value", 2),
                                                      ("program_words", 3),
                                                      ("returns", []))):
                bad = copy.deepcopy(original)
                bad[field] = changed
                manifest_path.write_text(json.dumps(bad))
                out = root / f"bad_manifest_{index}"
                self.assertNotEqual(invoke(package, out).returncode, 0)
                self.assertFalse(out.exists())

    def test_missing_packer_fails_with_machine_readable_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            isolated = root / "atlas-launch-plan"
            isolated.write_bytes(TOOL.read_bytes())
            result = subprocess.run(
                [sys.executable, str(isolated), "--capsule", str(capsule(root)),
                 "--input-address", "0x90002000", "--output-address", "0x90006000",
                 "--expected-rtl-revision", REVISION, "--out", str(root / "call")],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(result.stderr)["status"], "FAIL")
            self.assertIn("unavailable", result.stderr)


if __name__ == "__main__":
    unittest.main()
