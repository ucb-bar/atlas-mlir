"""Structural Linalg importer tests, independent of sample inputs and goldens."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compile_atlas_linalg_tile import Importer, UnsupportedLinalg, parse_module  # noqa: E402
from test_virtual_lowering import lower, run  # noqa: E402


SOURCE = (ROOT / "test/examples/captured_mlp32_linalg.mlir").read_text()
MANIFEST = json.loads((ROOT / "test/examples/captured_mlp32_arguments.json").read_text())
POLICY = json.loads((ROOT / "test/examples/atlas_mlp_numeric_policy.json").read_text())


def imported(source: str = SOURCE, policy: dict | None = None):
    return Importer(parse_module(source), MANIFEST,
                    POLICY if policy is None else policy,
                    0x90000000, 0x90010000).compile()


class CapturedMLPImportTest(unittest.TestCase):
    def test_parsed_capture_lowers_all_ops_and_preserves_argument_roles(self) -> None:
        virtual, report = imported()
        self.assertEqual(report["source_operations_accounted"], 19)
        self.assertEqual([row["source_argument"] for row in report["input_slots"]],
                         [4, 0, 1, 2, 3])
        self.assertEqual([row["physical_role"] for row in report["input_slots"]],
                         ["activation_fp8", "weight_fp8_nk", "bias_bf16",
                          "weight_fp8_nk", "bias_bf16"])
        self.assertEqual(virtual.count('"atlas.virtual_mxu_matmul"'), 2)
        self.assertEqual(virtual.count('"atlas.virtual_vpu_binary"'), 2)
        self.assertEqual(virtual.count('"atlas.virtual_vpu_unary"'), 1)
        self.assertEqual(virtual.count('"atlas.virtual_pack_fp8"'), 1)
        self.assertEqual(virtual.count('"atlas.virtual_output_bf16"'), 1)
        machine = lower(virtual)
        self.assertEqual(machine.count('"atlas.mxu_matmul"'), 2)
        self.assertEqual(machine.count('"atlas.vpu_binary"'), 2)
        self.assertEqual(
            run("atlas-opt", machine,
                "--verify-atlas-generated-schedule").returncode, 0,
        )

    def test_source_metadata_and_function_name_do_not_choose_implementation(self) -> None:
        original, _ = imported()
        modified = SOURCE.replace("@forward", "@another_name")
        modified = modified.replace('prov.region_id = "matmul_0"',
                                    'prov.region_id = "unrelated_metadata"')
        changed, _ = imported(modified)
        self.assertEqual(changed, original)

    def test_indexing_initialization_region_and_policy_mutations_fail(self) -> None:
        changed_sources = (
            SOURCE.replace("permutation = [1, 0]", "permutation = [0, 1]", 1),
            SOURCE.replace("affine_map<(d0, d1) -> (d1)>",
                           "affine_map<(d0, d1) -> (d0)>", 1),
            SOURCE.replace("0.000000e+00 : f32", "1.000000e+00 : f32", 1),
            SOURCE.replace("arith.maximumf", "arith.minimumf", 1),
        )
        for changed in changed_sources:
            with self.subTest(changed=changed != SOURCE):
                self.assertNotEqual(changed, SOURCE)
                with self.assertRaises((UnsupportedLinalg, ValueError)):
                    imported(changed)
        wrong_policy = dict(POLICY, interlayer_pack_scale_code=128)
        with self.assertRaisesRegex(UnsupportedLinalg,
                                    "interlayer_pack_scale_code"):
            imported(policy=wrong_policy)


if __name__ == "__main__":
    unittest.main()
