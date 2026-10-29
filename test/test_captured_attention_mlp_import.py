"""The captured QKV/attention/MLP block crosses the parsed Linalg boundary."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from atlas_boot_pack import pack_direct
from compile_atlas_linalg_tile import (
    Importer,
    UnsupportedLinalg,
    parse_module,
    stage_constants,
)
from test_virtual_lowering import BIN, lower, run

EXAMPLES = ROOT / "test/examples"
SOURCE = (EXAMPLES / "captured_attention_mlp32_linalg.mlir").read_text()
ARGUMENTS = json.loads((EXAMPLES / "captured_attention_mlp32_arguments.json").read_text())
POLICY = json.loads((EXAMPLES / "atlas_attention_mlp_numeric_policy.json").read_text())
SCALED_POLICY = json.loads((EXAMPLES / "atlas_attention_mlp_power2_candidate.json").read_text())
SOURCE_BOUND_POLICY = json.loads((EXAMPLES / "atlas_attention_mlp_source_bound_policy.json").read_text())


def imported(source: str = SOURCE, policy: dict | None = None):
    return Importer(parse_module(source), ARGUMENTS,
                    POLICY if policy is None else policy,
                    0x90000000, 0x90010000,
                    source_sha256=hashlib.sha256(source.encode()).hexdigest()).compile()


class CapturedAttentionMLPImportTest(unittest.TestCase):
    def test_preserves_two_ordered_outputs_and_shared_producers(self) -> None:
        source = (EXAMPLES / "captured_attention_mlp32_two_outputs_linalg.mlir").read_text()
        self.assertNotEqual(source, SOURCE)
        virtual, report = imported(source)
        self.assertEqual([row["index"] for row in report["outputs"]], [0, 1])
        self.assertEqual([row["dram_address"] for row in report["outputs"]],
                         [0x90010000, 0x90010800])
        self.assertNotIn("output", report)
        self.assertEqual(virtual.count('"atlas.virtual_mxu_matmul"'), 7)
        self.assertEqual(virtual.count('"atlas.virtual_output_bf16"'), 2)
        self.assertLess(virtual.index('index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16)'),
                        virtual.index('index = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16)'))
        machine = lower(virtual)
        self.assertEqual(machine.count('direction = "store"'), 4)
        self.assertEqual(run("atlas-opt", machine,
                             "--verify-atlas-generated-schedule").returncode, 0)
        self.assertEqual(run("atlas-opt", machine,
                             "--verify-atlas-machine-stream").returncode, 0)

        reversed_source = source.replace(
            "func.return %80, %90 : tensor<32x32xf32>, tensor<32x32xf32>",
            "func.return %90, %80 : tensor<32x32xf32>, tensor<32x32xf32>", 1,
        )
        reversed_virtual, _ = imported(reversed_source)
        self.assertNotEqual(virtual, reversed_virtual)
        self.assertNotEqual(run("atlas-emit", machine).stdout,
                            run("atlas-emit", lower(reversed_virtual)).stdout)
        with self.assertRaisesRegex(UnsupportedLinalg, "overlap"):
            Importer(parse_module(source), ARGUMENTS, POLICY,
                     0x90000000, 0x90003000).compile()

        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            machine_path = root / "machine.mlir"
            machine_path.write_text(machine)
            words = tuple(int(word, 16) for word in run("atlas-emit", machine).stdout.splitlines())
            report["program"] = {
                "program_sha256": hashlib.sha256(struct.pack(f"<{len(words)}I", *words)).hexdigest()
            }
            report["selected_rtl_revision"] = json.loads(
                (ROOT / "examples/handoff/captured_attention_mlp/layout.json").read_text()
            )["selected_rtl_revision"]
            manifest_path = root / "import-manifest.json"
            manifest_path.write_text(json.dumps(report))
            layout_path = ROOT / "examples/handoff/captured_attention_mlp/layout.json"
            with self.assertRaisesRegex(ValueError, "captured output 1 escapes"):
                pack_direct(machine_path, BIN / "atlas-emit", layout_path,
                            root / "too_small", manifest_path)
            self.assertFalse((root / "too_small").exists())

            two_output_layout = (ROOT / "examples/handoff/captured_attention_mlp"
                                 / "two-output-layout.json")
            capsule = pack_direct(machine_path, BIN / "atlas-emit", two_output_layout,
                                  root / "capsule", manifest_path)
            self.assertEqual(capsule["captured_output_count"], 2)
            self.assertEqual(capsule["captured_import_manifest_sha256"],
                             hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            self.assertEqual(capsule["program_words"], len(words))
            self.assertEqual((root / "capsule/program.bin").read_bytes(),
                             struct.pack(f"<{len(words)}I", *words))

    def test_compiles_all_seven_contractions_and_attention_support(self) -> None:
        virtual, report = imported()
        self.assertEqual(report["source_operations_accounted"], 61)
        self.assertEqual(report["mxu_units"], [0, 1, 0, 1, 0, 1, 0])
        self.assertEqual([row["source_argument"] for row in report["input_slots"]],
                         [5, 0, 1, 2, None, 3, 4])
        self.assertEqual(virtual.count('"atlas.virtual_mxu_matmul"'), 7)
        self.assertEqual(virtual.count('"atlas.virtual_xlu_transpose"'), 1)
        self.assertEqual(virtual.count('"atlas.virtual_vpu_reduce"'), 2)
        self.assertEqual(virtual.count('"atlas.virtual_output_bf16"'), 1)
        self.assertIn('"atlas.virtual_vpu_unary"', virtual)
        self.assertIn('"atlas.virtual_vpu_binary"', virtual)
        machine = lower(virtual)
        self.assertEqual(machine.count('"atlas.mxu_matmul"'), 7)
        self.assertEqual(machine.count('"atlas.xlu_transpose"'), 1)
        self.assertEqual(run("atlas-opt", machine,
                             "--verify-atlas-generated-schedule").returncode, 0)
        self.assertEqual(run("atlas-opt", machine,
                             "--verify-atlas-machine-stream").returncode, 0)

    def test_index_map_reduction_and_precision_mutations_fail(self) -> None:
        changes = (
            SOURCE.replace('permutation = [1, 0]', 'permutation = [0, 1]', 1),
            SOURCE.replace('arith.maximumf', 'arith.minimumf', 1),
            SOURCE.replace('arith.divf', 'arith.addf', 1),
            SOURCE.replace('0.000000e+00 : f32', '-0.000000e+00 : f32', 1),
        )
        for changed in changes:
            self.assertNotEqual(changed, SOURCE)
            with self.assertRaises((UnsupportedLinalg, ValueError)):
                imported(changed)
        with self.assertRaisesRegex(UnsupportedLinalg, "reciprocal policy"):
            imported(policy={**POLICY, "softmax_realization": "unknown"})

    def test_positive_zero_splat_can_reset_a_matmul(self) -> None:
        fill = next(line for line in SOURCE.splitlines()
                    if line.strip().startswith("%10 = linalg.fill"))
        splat = '    %10 = tensor.splat %9 {prov.region_id = "matmul_0"} : tensor<32x32xf32>'
        source = SOURCE.replace(fill, splat, 1)
        self.assertNotEqual(source, SOURCE)
        self.assertEqual(imported(source)[0], imported()[0])

        for changed in (
            source.replace("0.000000e+00 : f32", "-0.000000e+00 : f32", 1),
            source.replace("0.000000e+00 : f32", "1.000000e+00 : f32", 1),
            source.replace('prov.region_id = "matmul_0"} : tensor<32x32xf32>',
                           'semantics = "unknown"} : tensor<32x32xf32>', 1),
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                UnsupportedLinalg, "zero-filled accumulator"
            ):
                imported(changed)

    def test_integer_zero_compare_select_relu_pattern(self) -> None:
        old = ("      %83 = arith.constant 0.000000e+00 : f32\n"
               "      %84 = arith.maximumf %81, %83 : f32\n"
               "      linalg.yield %84 : f32")
        new = ("      %c0 = arith.constant 0 : i64\n"
               "      %z = arith.sitofp %c0 : i64 to f32\n"
               "      %lt = arith.cmpf olt, %81, %z : f32\n"
               "      %relu = arith.select %lt, %z, %81 : f32\n"
               "      linalg.yield %relu : f32")
        self.assertIn(old, SOURCE)
        source = SOURCE.replace(old, new, 1)
        self.assertEqual(imported(source)[0], imported()[0])

        for changed in (
            source.replace("arith.constant 0 : i64", "arith.constant 1 : i64", 1),
            source.replace("arith.cmpf olt", "arith.cmpf ogt", 1),
            source.replace("arith.select %lt, %z, %81", "arith.select %lt, %81, %z", 1),
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                UnsupportedLinalg, "ReLU compare/select region"
            ):
                imported(changed)

    def test_declared_pack_and_weight_scales_follow_the_matmul_edges(self) -> None:
        virtual, report = imported(policy=SCALED_POLICY)
        self.assertEqual(report["matmul_activation_pack_scale_codes"],
                         [127, 127, 127, 127, 127, 122, 122])
        self.assertEqual(virtual.count("scale_code = 127 : i32"), 4)
        self.assertEqual(virtual.count("scale_code = 122 : i32"), 2)
        scaled_slots = [row for row in report["input_slots"]
                        if "compensates_activation_scale_code" in row]
        self.assertEqual([(row["source_key"], row["compensates_activation_scale_code"])
                          for row in scaled_slots],
                         [("fc1.weight", 122), ("fc2.weight", 122)])
        machine = lower(virtual)
        self.assertEqual(machine.count('"atlas.mxu_matmul"'), 7)
        self.assertEqual(run("atlas-opt", machine,
                             "--verify-atlas-generated-schedule").returncode, 0)
        baseline_machine = lower(imported()[0])
        baseline_words = json.loads(run("atlas-emit", baseline_machine,
                                        "--program-json").stdout)["instructions"]
        scaled_words = json.loads(run("atlas-emit", machine,
                                      "--program-json").stdout)["instructions"]
        changed = [(before, after) for before, after in
                   zip(baseline_words, scaled_words, strict=True)
                   if before["word_u32"] != after["word_u32"]]
        self.assertEqual([(b["fields"]["offset"], a["fields"]["offset"])
                          for b, a in changed], [(127, 122)] * 2)
        from safetensors.numpy import load_file

        weights = load_file(EXAMPLES / "captured_attention_mlp32_weights.safetensors")
        with tempfile.TemporaryDirectory() as temporary:
            staged = stage_constants(
                EXAMPLES / "captured_attention_mlp32_weights.safetensors",
                report["input_slots"], Path(temporary))
            for row in staged:
                if row["source_key"] in ("fc1.weight", "fc2.weight"):
                    source = weights[row["source_key"]]
                    self.assertEqual(set(source.flat), {0.0, 0.5})
                    expected = bytes(0x08 if value == 0.5 else 0
                                     for value in source.flat)
                    self.assertEqual(expected.count(0x08), 32)
                    self.assertEqual((Path(temporary) / row["payload_file"]).read_bytes(),
                                     expected)

    def test_source_bound_policy_keeps_decisions_with_their_regions(self) -> None:
        baseline, baseline_report = imported(policy=SCALED_POLICY)
        virtual, report = imported(policy=SOURCE_BOUND_POLICY)
        self.assertEqual(virtual, baseline)
        self.assertEqual(report["mxu_units"], baseline_report["mxu_units"])
        self.assertEqual(report["matmul_activation_pack_scale_codes"],
                         baseline_report["matmul_activation_pack_scale_codes"])
        self.assertEqual(report["source_contraction_regions"],
                         [f"matmul_{index}" for index in range(7)])
        reversed_rows = dict(reversed(list(SOURCE_BOUND_POLICY["matmul_regions"].items())))
        self.assertEqual(imported(policy={**SOURCE_BOUND_POLICY,
                                          "matmul_regions": reversed_rows})[0], virtual)
        for change, message in (
            ({**SOURCE_BOUND_POLICY, "source_sha256": "0" * 64}, "differs from captured"),
            ({**SOURCE_BOUND_POLICY, "matmul_regions": {
                **SOURCE_BOUND_POLICY["matmul_regions"],
                "matmul_7": {"mxu_unit": 0, "activation_pack_scale_code": 127},
            }}, "unused contraction"),
            ({**SOURCE_BOUND_POLICY, "matmul_regions": {
                key: value for key, value in SOURCE_BOUND_POLICY["matmul_regions"].items()
                if key != "matmul_5"
            }}, "unique source-bound"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(UnsupportedLinalg, message):
                imported(policy=change)
        changed_source = SOURCE.replace('prov.region_id = "matmul_5"',
                                        'prov.region_id = "another_region"', 1)
        with self.assertRaisesRegex(UnsupportedLinalg, "differs from captured"):
            imported(source=changed_source, policy=SOURCE_BOUND_POLICY)

    def test_nonunit_scale_rejects_uncompensated_edges_and_bad_codes(self) -> None:
        for codes, message in (
            ([127] * 6, "one raw E8M0"),
            ([127] * 6 + [256], "one raw E8M0"),
            ([122] + [127] * 6, "external activation"),
            ([127] * 3 + [122] + [127] * 3, "computed weight"),
        ):
            with self.subTest(codes=codes), self.assertRaisesRegex(UnsupportedLinalg, message):
                imported(policy={**SCALED_POLICY,
                                 "matmul_activation_pack_scale_codes": codes})

    def test_one_weight_used_at_two_scales_gets_two_initialized_slots(self) -> None:
        changed = SOURCE.replace(
            "linalg.transpose ins(%4:tensor<32x32xf32>)",
            "linalg.transpose ins(%3:tensor<32x32xf32>)", 1)
        self.assertNotEqual(changed, SOURCE)
        policy = {**SCALED_POLICY,
                  "matmul_activation_pack_scale_codes": [127] * 5 + [122, 126]}
        _, report = imported(source=changed, policy=policy)
        uses = [row for row in report["input_slots"]
                if row["source_key"] == "fc1.weight"]
        self.assertEqual([row["compensates_activation_scale_code"] for row in uses],
                         [122, 126])
        self.assertNotEqual(uses[0]["dram_address"], uses[1]["dram_address"])
        with tempfile.TemporaryDirectory() as temporary:
            staged = stage_constants(
                EXAMPLES / "captured_attention_mlp32_weights.safetensors",
                report["input_slots"], Path(temporary))
            payloads = [(Path(temporary) / row["payload_file"]).read_bytes()
                        for row in staged if row["source_key"] == "fc1.weight"]
            self.assertEqual(len(payloads), 2)
            self.assertNotEqual(payloads[0], payloads[1])


if __name__ == "__main__":
    unittest.main()
