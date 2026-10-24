"""Synthetic admission/tamper tests; no model, simulator or board execution."""

from pathlib import Path
import sys

TOOL_ROOT = Path(__file__).absolute().parents[1] / "tools"
sys.path.insert(0, str(TOOL_ROOT))
sys.path.insert(0, str(TOOL_ROOT.parents[1] if TOOL_ROOT.name != "tools" else TOOL_ROOT))
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

import export_contraction_capsule as capsule


class CapsuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="atlas-capsule-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.producer = Path(capsule.__file__).absolute().parents[1]
        self.model = self.root / "model"
        self.rtl = self.root / "rtl"
        self.model.mkdir()
        self.rtl.mkdir()
        self.modelroot = self.root / "model-source"
        self.module = self.modelroot / "npu_model/__init__.py"
        self.module.parent.mkdir(parents=True)
        self.module.write_text("")
        self.curve = self.modelroot / "npu_model/configs/memory_curves/ee290sim_vcs_probe.json"
        self.curve.parent.mkdir(parents=True)
        self.curve.write_text("{}")
        self.input = self.root / "operands.npz"
        self.manifest = self.root / "manifest.json"
        self.zero = np.zeros((32, 32), dtype=np.float32)
        self.native = np.full((32, 32), 0.25, dtype=np.float32)
        self.source = {"A": self.zero, "B": self.zero, "native_C": self.native}
        np.savez(self.input, **self.source)
        manifest = {"schema_version": 1, "source_dtype": "float32", "source": {"kind": "synthetic"},
            "site": {"M": 32, "K": 32, "N": 32}, "npz_sha256": capsule.sha256(self.input)}
        capsule.write_json(self.manifest, manifest)
        np.savez(self.model / "source.npz", **self.source)
        np.savez(self.model / "encoded.npz", A=self.zero.astype(np.uint8), B=self.zero.astype(np.uint8))
        np.savez(self.model / "result.npz", target_C=self.zero, actual_bits=self.zero.astype(np.uint16),
            expected_bits=self.zero.astype(np.uint16))
        capsule.write_json(self.model / "source-manifest.json", manifest)
        self.expected = bytes(2048)
        self.initial = bytes(2048) + b"\xa5" * 256 + b"\xcd" * 2048 + b"\x5a" * 256
        for name, payload in (("expected-bf16-tiles.bin", self.expected), ("initial-dram.bin", self.initial),
            ("program.bin", b"\x73\0\0\0")):
            (self.model / name).write_bytes(payload)
        (self.model / "program.S").write_text(capsule.assembly(32, 32, 32, 2304))
        code = capsule.code_sources()
        (self.model / "producer-sources").mkdir()
        for name in code:
            (self.model / "producer-sources" / name).write_bytes((self.producer / "tools" / name).read_bytes())
        binding = {"revision": capsule.replay.MODEL_PIN, "tracked_source_diff_exit": 0,
            "imported_source_sha256": {"npu_model/__init__.py": capsule.sha256(self.module)},
            "curve_sha256": capsule.sha256(self.curve)}
        inputs = {"input": capsule.sha256(self.input), "manifest": capsule.sha256(self.manifest)}
        self.receipt = {"status": "passed_model_only", "recipe": capsule.RECIPES["unit"], "shape": [32, 32, 32],
            "code_before": code, "code_after": code.copy(), "source_inputs": inputs, "native_fidelity_claim": False,
            "raw_image_execution_claim": False,
            "invocations": "one_reset_core_per_repeat_all_tiles_in_one_instruction_program",
            "model_input": "new_assembler_decoded_instruction_objects", "model_binding_before": binding,
            "source_bookends": {"unchanged": True, "input_before": inputs, "input_after": inputs, "model_after": binding},
            "model": {**binding, "numerics": "rtl", "backend": "curve", "module_file": str(self.module)},
            "command": ["producer.py", "--input", str(self.input), "--manifest", str(self.manifest)],
            "conversion": {key: {"scale_exponent": 0, "scale_code": 127} for key in ("A", "B")},
            "output_rescale_exponent": 0, "instruction_words": 1, "requested_dma": {"load_bytes": 2048, "store_bytes": 2048},
            "memory": {"dram_base": 1 << 32, "input_bytes": 2048, "output_offset": 2304, "output_bytes": 2048,
            "guard_bytes_each": 256, "total_dram_bytes": 4608},
            "tiles": [{"row_range": [0, 32], "column_range": [0, 32], "k_range": [0, 32]}],
            "resident_operand_reuse_claim": False,
            "operand_reuse": {"mode": "none", "resident_b_bytes": 0, "resident_b_vmem_byte_base": None},
            "diagnostic_errors": {"target_vs_native_C": capsule.metrics(self.zero, self.native)}, "runs": []}
        for i in range(2):
            trace = []
            for beat in range(128):
                trace.append({"cycle": beat + 1, "s1_fire": beat == 0, "halted": False, "instruction": "nop", "s1_pc": 0,
                    "dma_a": [0, 0, beat >= 64], "dma_d": [0]})
            trace.append({"cycle": 129, "s1_fire": False, "halted": True, "instruction": "ecall", "s1_pc": 0, "dma_a": None,
                "dma_d": None})
            (self.model / f"cycles-{i}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in trace))
            (self.model / f"perfetto-{i}.json").write_text("{}")
            (self.model / f"actual-bf16-halves-{i}.bin").write_bytes(self.expected)
            self.receipt["runs"].append({"repeat": i, "initialization_seed": 42 + i, "halt_reason": "ecall",
                "halt_pc_word": 0, "cycles": 129, "issued_instructions": 1, "all_output_bytes_exact": True,
                "input_and_guards_preserved": True, "drained_before_halt_observation": True,
                "observed": {"load_bytes": 2048, "store_bytes": 2048, "response_beats": 128}})
        self.refresh_model()
        self.args = SimpleNamespace(model_receipt=self.model / "receipt.json", model_receipt_sha256="",
            rtl_receipt=self.rtl / "receipt.json", rtl_receipt_sha256="", input=self.input, manifest=self.manifest,
            producer_root=self.producer, model_root=self.modelroot, out=self.root / "capsule")
        self.args.model_receipt_sha256 = capsule.sha256(self.args.model_receipt)
        self.make_rtl()

    def refresh_model(self):
        self.receipt["artifacts"] = {p.name: capsule.sha256(p) for p in self.model.iterdir() if p.is_file()
            and p.name != "receipt.json"}
        capsule.write_json(self.model / "receipt.json", self.receipt)

    def make_rtl(self):
        self.records = []

        def record(path):
            self.records.append({"path": str(path), "sha256": capsule.sha256(path)})

        for name in ("program.bin", "initial-dram.bin", "expected-bf16-tiles.bin", "encoded.npz"):
            (self.rtl / name).write_bytes((self.model / name).read_bytes())
            record(self.rtl / name)
        for name in ("run_program.py", "prepare_program.py"):
            record(self.producer / "tools/rtl_core" / name)
        exe = self.root / "synthetic-runner"
        exe.write_text("not executable")
        build = self.root / "runner"
        build.mkdir()
        capsule.write_json(build / "receipt.json",
            {"status": "passed_build_only", "source_artifacts_preserved": True,
            "records": [{"path": str(exe), "sha256": capsule.sha256(exe)}],
            "executable": {"path": str(exe), "sha256": capsule.sha256(exe)}})
        record(build / "receipt.json")
        evidence = {"mode": "authenticated_complete_model_payload_reuse", "fresh_arithmetic_rebuild": False,
            "actual": {"path": str(self.model / "actual-bf16-halves-0.bin"),
            "sha256": capsule.sha256(self.model / "actual-bf16-halves-0.bin")},
            "producer_epoch": self.receipt["code_before"], "controls_enabled": False,
            "source_receipt": {"path": str(self.args.model_receipt), "sha256": self.args.model_receipt_sha256}}
        case = {"id": "positive", "program": str(self.rtl / "program.bin"), "expected": self.expected.hex(),
            "actual_oracle": self.expected.hex()}
        data = {"memory": self.receipt["memory"], "requested_dma": self.receipt["requested_dma"],
            "arithmetic_evidence": evidence, "initial": str(self.rtl / "initial-dram.bin"), "cases": [case]}
        capsule.write_json(self.rtl / "input.json", data)
        record(self.rtl / "input.json")
        runs = []
        for i in range(2):
            trace = [{"cycle": 1, "event": "host_a", "csr": 0, "address": 0x20000, "data": 0x73},
                {"cycle": 2, "event": "host_d"}, {"cycle": 3, "event": "host_a", "csr": 1, "address": 24, "data": 1},
                {"cycle": 4, "event": "host_d"},
                {"cycle": 5, "event": "state", "reset": 0, "halted": 0, "public_idle": 1, "outstanding": 0}]
            for beat in range(128):
                store = beat >= 64
                trace.extend([{"cycle": 6 + 2 * beat, "event": "dma_a",
                    "address": (1 << 32) + (2304 if store else 0) + (beat % 64) * 32, "source": 0, "opcode": 0 if store
                    else 4, "bytes": [0] * 32},
                    {"cycle": 7 + 2 * beat, "event": "dma_d", "source": 0, "opcode": 0 if store else 1}])
            trace.extend([{"cycle": 262, "event": "status_a", "address": 8}, {"cycle": 263, "event": "status_d", "data": 5}])
            trace.extend({"cycle": 264 + j, "event": "state", "reset": 0, "halted": 1, "public_idle": 1,
                "outstanding": 0} for j in range(32))
            actual = self.rtl / f"positive-{i}-actual.bin"
            actual.write_bytes(self.expected)
            record(actual)
            tracepath = self.rtl / f"positive-{i}-trace.jsonl"
            tracepath.write_text("".join(json.dumps(e) + "\n" for e in trace))
            record(tracepath)
            summary = {"halted": True, "cycles_from_start_csr": 260, "quiet_cycles": 32, "load_bytes": 2048,
                "store_bytes": 2048, "responses": 128, "csr_status": 5, "halt_reason": "ecall", "guards_preserved": True,
                "public_busy_idle": True, "vpu_busy_observed": False}
            capsule.write_json(self.rtl / f"positive-{i}-summary.json", summary)
            record(self.rtl / f"positive-{i}-summary.json")
            runs.append({"id": "positive", "repeat": i, "execution": {"returncode": 0}, "summary": summary,
                "physical_replay": capsule.replay.replay(data, case, tracepath, actual, self.rtl / "program.bin")})
        self.rtlreceipt = {"status": "passed_raw_rtl_functional", "source_artifacts_preserved": True, "quiet_cycles": 32,
            "public_idle_only": True, "vpu_busy_observed": False, "native_quality_claim": False,
            "platform_performance_claim": False, "records": self.records, "arithmetic_evidence": evidence, "runs": runs,
            "command": ["run_program.py", "--runner-root", str(build), "--runner-sha256",
            capsule.sha256(build / "receipt.json")]}
        capsule.write_json(self.args.rtl_receipt, self.rtlreceipt)
        self.args.rtl_receipt_sha256 = capsule.sha256(self.args.rtl_receipt)

    def export(self):
        # Only Git revision lookup is mocked. Every file hash/array/layout/trace check executes.
        with mock.patch("subprocess.check_output", return_value=capsule.replay.MODEL_PIN):
            return capsule.export(self.args)

    def test_physical_success_preserves_native_failure(self):
        status = self.export()
        self.assertFalse(status["native_gate"]["passed"])
        self.assertEqual(status["native_gate"]["metrics"]["max_abs"], 0.25)
        self.assertFalse(status["fpga_ready"])
        self.assertFalse(status["full_model"])
        self.assertFalse(status["launchable_host"])
        for line in (self.args.out / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split()
            self.assertEqual(capsule.sha256(self.args.out / name), digest)

    def test_model_receipt_pin_and_source_bookends(self):
        self.receipt["source_bookends"]["input_after"]["input"] = "0" * 64
        self.refresh_model()
        with self.assertRaisesRegex(ValueError, "Hash differs"):
            self.export()
        self.args.model_receipt_sha256 = capsule.sha256(self.args.model_receipt)
        with self.assertRaises(ValueError):
            self.export()

    def test_tampered_actual_even_with_matching_artifact_hash_refuses(self):
        (self.model / "actual-bf16-halves-1.bin").write_bytes(b"x" * 2048)
        self.refresh_model()
        self.args.model_receipt_sha256 = capsule.sha256(self.args.model_receipt)
        with self.assertRaisesRegex(ValueError, "Model actual differs"):
            self.export()

    def test_encoded_and_post_vpu_views_are_independent_checks(self):
        np.savez(self.model / "encoded.npz", A=np.ones((32, 32), dtype=np.uint8), B=self.zero.astype(np.uint8))
        self.refresh_model()
        self.args.model_receipt_sha256 = capsule.sha256(self.args.model_receipt)
        with self.assertRaisesRegex(ValueError, "Encoded operand"):
            self.export()
        np.savez(self.model / "encoded.npz", A=self.zero.astype(np.uint8), B=self.zero.astype(np.uint8))
        np.savez(self.model / "result.npz", target_C=self.native, actual_bits=self.zero.astype(np.uint16),
            expected_bits=self.zero.astype(np.uint16))
        self.refresh_model()
        self.args.model_receipt_sha256 = capsule.sha256(self.args.model_receipt)
        with self.assertRaisesRegex(ValueError, "exact target view"):
            self.export()

    def test_physical_half_layout_on_nonuniform_words(self):
        bits = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        expected = b"".join(int(bits[r, c]).to_bytes(2, "little") for tr in (0, 32) for tc in (0, 32) for half in (0,
            16) for r in range(tr, tr + 32) for c in range(tc + half, tc + half + 16))
        self.assertEqual(capsule.physical(bits), expected)

    def test_shape_coverage_guard_and_counter_tampering(self):
        for change in (lambda r: r.update(shape=[64, 32, 32]), lambda r: r.update(tiles=[]),
            lambda r: r["runs"][0]["observed"].update(load_bytes=32)):
            original = copy.deepcopy(self.receipt)
            change(self.receipt)
            self.refresh_model()
            self.args.model_receipt_sha256 = capsule.sha256(self.args.model_receipt)
            with self.assertRaises(ValueError):
                self.export()
            self.receipt = original
        changed = bytearray(self.initial)
        changed[2048] ^= 1
        (self.model / "initial-dram.bin").write_bytes(changed)
        self.refresh_model()
        self.args.model_receipt_sha256 = capsule.sha256(self.args.model_receipt)
        with self.assertRaisesRegex(ValueError, "guards"):
            self.export()

    def test_rtl_trace_tampering_cannot_hide_in_updated_record(self):
        p = self.rtl / "positive-1-trace.jsonl"
        p.write_text(p.read_text().replace('"data": 5', '"data": 7'))
        for r in self.records:
            if r["path"] == str(p):
                r["sha256"] = capsule.sha256(p)
        capsule.write_json(self.args.rtl_receipt, self.rtlreceipt)
        self.args.rtl_receipt_sha256 = capsule.sha256(self.args.rtl_receipt)
        with self.assertRaisesRegex(AssertionError, "halt status"):
            self.export()

    def test_rtl_counter_tampering_cannot_hide_in_updated_receipt(self):
        summary = self.rtlreceipt["runs"][0]["summary"]
        summary["load_bytes"] = 32
        path = self.rtl / "positive-0-summary.json"
        capsule.write_json(path, summary)
        for record in self.records:
            if record["path"] == str(path):
                record["sha256"] = capsule.sha256(path)
        capsule.write_json(self.args.rtl_receipt, self.rtlreceipt)
        self.args.rtl_receipt_sha256 = capsule.sha256(self.args.rtl_receipt)
        with self.assertRaisesRegex(ValueError, "summary completion/traffic"):
            self.export()

    def test_output_collision_and_symlink(self):
        self.args.out.mkdir()
        with self.assertRaisesRegex(ValueError, "Fresh"):
            self.export()
        self.args.out.rmdir()
        self.args.out.symlink_to(self.model, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic"):
            self.export()

    def test_traversal_path_rejected_before_metadata_and_archive_before_hash(self):
        # These are lexical strings/mocked ZIP metadata; no unsafe file/member is created.
        with mock.patch.object(Path, "is_symlink") as metadata:
            with self.assertRaises(ValueError):
                capsule.safe_path(self.root / ".." / "file")
            metadata.assert_not_called()
        for invalid in (SimpleNamespace(filename="../unadmitted.npy", external_attr=0),
            SimpleNamespace(filename="A.npy", external_attr=0o120777 << 16)):
            with mock.patch.object(zipfile.ZipFile, "infolist", return_value=[invalid]), mock.patch.object(Path,
                "read_bytes") as contents:
                with self.assertRaises(ValueError):
                    capsule.Admission().bind(self.input)
                contents.assert_not_called()


if __name__ == "__main__":
    unittest.main()
