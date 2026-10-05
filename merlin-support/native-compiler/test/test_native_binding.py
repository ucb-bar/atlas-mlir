"""Installed native entry point smoke; hardware execution is a separate gate."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from atlas_native_support.host_build import ee290_baremetal_recipe
from merlin.semantic_compiler.target_binding import load_native_target_binding
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "test/fixtures"


def _command(*args: object) -> subprocess.CompletedProcess[str]:
    cli = Path(sys.executable).with_name("merlin-targetgen")
    if not cli.is_file():
        raise AssertionError("install Merlin's merlin-targetgen entry point")
    return subprocess.run(
        [str(cli), *(str(arg) for arg in args)],
        text=True, capture_output=True, timeout=180, check=False,
    )


class NativeBindingTest(unittest.TestCase):
    def test_installed_ee290_host_recipe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="atlas-host-recipe-") as temporary:
            root = Path(temporary)
            compiler, linker, specs = (root / name for name in ("cc", "link.ld", "htif.specs"))
            for path in (compiler, linker, specs):
                path.write_text("selected tool input\n")
            recipe = ee290_baremetal_recipe(
                compiler=compiler, link_script=linker, specs=specs,
            )
            self.assertIs(type(recipe), HarnessBuildRecipe)
            driver = recipe.support_sources[0]
            self.assertTrue(driver.is_file())
            self.assertTrue(driver.with_suffix(".h").is_file())
            self.assertIn("atlas_ee290_run", driver.read_text())
            self.assertNotIn("golden", driver.read_text().lower())
            command = recipe.command(sources=(root / "host.c",), output=root / "host.elf")
            self.assertEqual(command.count(str(driver)), 1)
            self.assertIn("-march=rv64imafd", command)
            self.assertIn(f"-specs={specs}", command)
            with self.assertRaises(FileNotFoundError):
                ee290_baremetal_recipe(
                    compiler=compiler, link_script=linker, specs=root / "absent.specs",
                )

    def test_imported_source_identity(self) -> None:
        package = importlib.resources.files("atlas_native_support")
        record = json.loads(package.joinpath("source_import.json").read_text())
        self.assertEqual(record["schema"], "atlas.native_support_source_import.v1")
        modified = record.get("modified_after_import", {})
        self.assertEqual(
            set(modified),
            {
                "atlas_native_support/dialect_plan.py",
                "atlas_native_support/mxu0_binding.py",
                "atlas_native_support/mxu0_emit.py",
                "atlas_native_support/mxu0_program_set.py",
                "atlas_native_support/requirements.json",
            },
        )
        unchanged = 0
        for relative, digest in record["source_file_sha256"].items():
            actual = hashlib.sha256(
                package.joinpath(Path(relative).name).read_bytes()
            ).hexdigest()
            if relative in modified:
                self.assertNotEqual(actual, digest, relative)
                self.assertEqual(actual, modified[relative]["sha256"], relative)
                self.assertTrue(modified[relative]["reason"])
            else:
                self.assertEqual(actual, digest, relative)
                unchanged += 1
        self.assertEqual(unchanged, 9)

    def test_installed_profile(self) -> None:
        binding = load_native_target_binding("atlas_tensor")
        profile = binding.profile()
        self.assertEqual(len(profile.descriptors), 95)
        self.assertEqual(len(profile.banks), 10)
        self.assertTrue(profile.target_identity.startswith("atlas-tensor-selection-"))
        names = {descriptor.name for descriptor in profile.descriptors}
        self.assertTrue({"vpu_log2_bf16_raw", "vpu_sqrt_bf16_raw",
                         "vpu_exp2_bf16_bounded"} <= names)

    def test_build_compile_and_wrong_target_refusal(self) -> None:
        merlin = os.environ.get("MERLIN_ROOT")
        atlas = os.environ.get("ATLAS_RTL_ROOT")
        if not merlin or not atlas:
            self.skipTest("set MERLIN_ROOT and ATLAS_RTL_ROOT for the installed compiler smoke")
        merlin_root = Path(merlin).resolve(strict=True)
        atlas_root = Path(atlas).resolve(strict=True)
        requirements = json.loads((ROOT / "src/atlas_native_support/requirements.json").read_text())
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=merlin_root, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
        self.assertEqual(revision, requirements["merlin_revision"])

        with tempfile.TemporaryDirectory(prefix="atlas-native-binding-") as temporary:
            work = Path(temporary)
            cargo = Path(os.environ.get("ATLAS_NATIVE_TEST_CARGO_TARGET_DIR", work / "cargo"))
            snapshot = work / "snapshot"
            built = _command(
                "native-build", "--engine", "merlin_native", "--support", "atlas_tensor",
                "--cargo-target-dir", cargo, "--source-revision", revision,
                "--out", snapshot,
            )
            self.assertEqual(built.returncode, 0, built.stderr + built.stdout)
            self.assertEqual(json.loads(built.stdout)["status"], "selection_only")

            request = FIXTURES / "log2-request.json"
            abi = FIXTURES / "log2-abi.json"
            for name in ("log2", "sqrt", "exp2", "minmax", "shared-log2-sqrt"):
                with self.subTest(name=name):
                    artifact = work / f"{name}-program"
                    selected_request = FIXTURES / f"{name}-request.json"
                    status_file = work / f"{name}-status.json"
                    selected_abi = FIXTURES / (
                        f"{name}-abi.json" if name in {"minmax", "shared-log2-sqrt"}
                        else "log2-abi.json"
                    )
                    search_limits = (
                        ["--search-limits", str(FIXTURES / "minmax-search-limits.json")]
                        if name == "minmax" else []
                    )
                    compiled = _command(
                        "native-compile", "--engine", "merlin_native", "--support", "atlas_tensor",
                        "--snapshot", snapshot, "--request", selected_request, "--abi", selected_abi,
                        *search_limits,
                        "--target-source", atlas_root, "--mode", "strict-native",
                        "--out", artifact, "--status-file", status_file,
                    )
                    self.assertEqual(compiled.returncode, 0, compiled.stderr + compiled.stdout)
                    status = json.loads(status_file.read_text())
                    manifest = json.loads((artifact / "manifest.json").read_text())
                    binary = (artifact / "program.bin").read_bytes()
                    self.assertEqual(status["status"], "emitted")
                    self.assertEqual(status["engine"], "merlin_native")
                    self.assertEqual(status["binary_sha256"], hashlib.sha256(binary).hexdigest())
                    self.assertEqual(manifest["binary_sha256"], status["binary_sha256"])
                    self.assertGreater(len(binary), 0)
                    if name == "minmax":
                        self.assertGreater(manifest["selected_instructions"], 1)
                    if name == "shared-log2-sqrt":
                        plan = json.loads((artifact / "execution_plan.json").read_text())
                        self.assertEqual([row["source"] for row in plan["inputs"]], ["source"])
                        self.assertEqual(
                            [row["source"] for row in plan["outputs"]],
                            ["log2", "sqrt"],
                        )
                        self.assertEqual(
                            [row["byte_address"] for row in plan["outputs"]],
                            [0x90002000, 0x90004000],
                        )
                        self.assertEqual(
                            [row["byte_length"] for row in plan["outputs"]],
                            [2048, 2048],
                        )
                        self.assertGreaterEqual(manifest["selected_instructions"], 2)

                        swapped_request = json.loads(selected_request.read_text())
                        swapped_request["outputs"].reverse()
                        swapped_request["source_identity"] = "public-swapped-bf16-outputs"
                        swapped_path = work / "swapped-request.json"
                        swapped_path.write_text(json.dumps(swapped_request))
                        swapped_artifact = work / "swapped-program"
                        swapped_status = work / "swapped-status.json"
                        swapped = _command(
                            "native-compile", "--engine", "merlin_native",
                            "--support", "atlas_tensor", "--snapshot", snapshot,
                            "--request", swapped_path, "--abi", selected_abi,
                            "--target-source", atlas_root, "--mode", "strict-native",
                            "--out", swapped_artifact, "--status-file", swapped_status,
                        )
                        self.assertEqual(swapped.returncode, 0, swapped.stderr + swapped.stdout)
                        swapped_plan = json.loads(
                            (swapped_artifact / "execution_plan.json").read_text()
                        )
                        self.assertEqual(
                            [row["source"] for row in swapped_plan["outputs"]],
                            ["sqrt", "log2"],
                        )
                        self.assertEqual(
                            [row["byte_address"] for row in swapped_plan["outputs"]],
                            [0x90002000, 0x90004000],
                        )
                        self.assertNotEqual(
                            manifest["request_digest"],
                            json.loads((swapped_artifact / "manifest.json").read_text())["request_digest"],
                        )

            changed = json.loads(request.read_text())
            changed["target_identity"] = "foreign-target"
            changed_path = work / "wrong-target.json"
            changed_path.write_text(json.dumps(changed))
            refused = _command(
                "native-compile", "--engine", "merlin_native", "--support", "atlas_tensor",
                "--snapshot", snapshot, "--request", changed_path, "--abi", abi,
                "--target-source", atlas_root, "--mode", "strict-native",
                "--out", work / "wrong-program", "--status-file", work / "wrong-status.json",
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertEqual(json.loads((work / "wrong-status.json").read_text())["status"], "compile_error")
            self.assertFalse((work / "wrong-program").exists())


if __name__ == "__main__":
    unittest.main()
