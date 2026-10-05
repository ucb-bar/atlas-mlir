"""Checks for the generated diagnostic Atlas machine-IR view."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from atlas_native_support.dialect_plan import generate_dialect, selected_dialect_plan
from merlin.targetgen.generate import mlir_scaffold, typed_mlir


class GeneratedDialectPlanTest(unittest.TestCase):
    def test_reviewed_descriptors_generate_typed_mlir(self) -> None:
        plan, coverage = selected_dialect_plan()
        self.assertEqual(len(plan["ops"]), 97)
        self.assertEqual(len(coverage["covered_descriptors"]), 97)
        self.assertEqual(coverage["omitted_descriptors"], {})
        self.assertTrue(coverage["omitted_families"])
        self.assertTrue(coverage["unrepresented_obligations"])
        self.assertEqual(len({op["name"] for op in plan["ops"]}), 97)
        for op in plan["ops"]:
            signature = op["signature"]
            self.assertEqual(signature["effects"], ["read", "write"])
            for role in ("operands", "results"):
                self.assertTrue(all(value["type"].startswith("!atlas.") for value in signature[role]))
        typed_mlir.validate(plan)
        artifacts = mlir_scaffold.generate(plan)
        names = {artifact.relpath for artifact in artifacts}
        self.assertTrue(any(str(name).endswith("AtlasOps.td") for name in names))
        self.assertTrue(any(str(name).endswith("CMakeLists.txt") for name in names))

    def test_rejects_missing_effects_or_unqualified_type(self) -> None:
        plan, _ = selected_dialect_plan()
        without_effects = copy.deepcopy(plan)
        del without_effects["ops"][0]["signature"]["effects"]
        with self.assertRaisesRegex(ValueError, "requires operands, results, attributes, and effects"):
            typed_mlir.validate(without_effects)

        unqualified = copy.deepcopy(plan)
        unqualified["ops"][0]["signature"]["operands"][0]["type"] = "external_i8_raw_byte_copy"
        with self.assertRaisesRegex(ValueError, "unsupported typed dialect value type"):
            typed_mlir.validate(unqualified)

    def test_source_bound_generation_and_compiled_parser(self) -> None:
        rtl = os.environ.get("ATLAS_RTL_ROOT")
        merlin = os.environ.get("MERLIN_ROOT")
        tool = os.environ.get("ATLAS_GENERATED_OPT")
        if not (rtl and merlin and tool):
            self.skipTest("set ATLAS_RTL_ROOT, MERLIN_ROOT, and ATLAS_GENERATED_OPT")
        requirements = json.loads(
            (Path(__file__).resolve().parents[1] / "src/atlas_native_support/requirements.json").read_text()
        )
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=merlin, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        self.assertEqual(revision, requirements["merlin_revision"])

        with tempfile.TemporaryDirectory(prefix="atlas-generated-dialect-") as temporary:
            root = Path(temporary)
            destination = root / "generated"
            coverage = generate_dialect(
                destination,
                target_source=Path(rtl),
                software_spec=Path(merlin) / "examples/atlas/target/software-spec.yaml",
            )
            self.assertEqual(len(coverage["covered_descriptors"]), 97)
            self.assertTrue((destination / "CMakeLists.txt").is_file())
            self.assertEqual(json.loads((destination / "coverage.json").read_text()), coverage)
            with self.assertRaises(FileExistsError):
                generate_dialect(
                    destination,
                    target_source=Path(rtl),
                    software_spec=Path(merlin) / "examples/atlas/target/software-spec.yaml",
                )

            program = root / "roundtrip.mlir"
            program.write_text(
                "module {\n"
                "  func.func @dma_load(%source: !atlas.external_i8_raw_byte_copy) "
                "-> !atlas.vmem_i8_raw_byte_copy {\n"
                "    %result = \"atlas.dma_load_wait\"(%source) "
                "{source0Address = 0 : i64, resultAddress = 64 : i64} "
                ": (!atlas.external_i8_raw_byte_copy) -> !atlas.vmem_i8_raw_byte_copy\n"
                "    return %result : !atlas.vmem_i8_raw_byte_copy\n"
                "  }\n"
                "}\n"
            )
            parsed = subprocess.run(
                [tool, str(program)], capture_output=True, text=True, timeout=30,
                check=False,
            )
            self.assertEqual(parsed.returncode, 0, parsed.stderr)
            self.assertIn("atlas.dma_load_wait", parsed.stdout)
            self.assertIn("resultAddress = 64 : i64", parsed.stdout)

            for name, mutated in {
                "wrong_source_type": program.read_text().replace(
                    "!atlas.external_i8_raw_byte_copy", "!atlas.vmem_i8_raw_byte_copy"
                ),
                "missing_address": program.read_text().replace(
                    ", resultAddress = 64 : i64", ""
                ),
            }.items():
                with self.subTest(name=name):
                    invalid = root / f"{name}.mlir"
                    invalid.write_text(mutated)
                    rejected = subprocess.run(
                        [tool, str(invalid)], capture_output=True, text=True,
                        timeout=30, check=False,
                    )
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn("error:", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
