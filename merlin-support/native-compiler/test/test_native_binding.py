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

from merlin.semantic_compiler.target_binding import load_native_target_binding


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
    def test_imported_source_identity(self) -> None:
        package = importlib.resources.files("atlas_native_support")
        record = json.loads(package.joinpath("source_import.json").read_text())
        self.assertEqual(record["schema"], "atlas.native_support_source_import.v1")
        unchanged = 0
        for relative, digest in record["source_file_sha256"].items():
            if Path(relative).name == "requirements.json":
                continue
            self.assertEqual(
                hashlib.sha256(package.joinpath(Path(relative).name).read_bytes()).hexdigest(),
                digest, relative,
            )
            unchanged += 1
        self.assertEqual(unchanged, 13)

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
                "--crate", merlin_root / "src/merlin/semantic_compiler/egg_bridge",
                "--cargo-target-dir", cargo, "--source-revision", revision,
                "--out", snapshot,
            )
            self.assertEqual(built.returncode, 0, built.stderr + built.stdout)
            self.assertEqual(json.loads(built.stdout)["status"], "selection_only")

            request = FIXTURES / "log2-request.json"
            abi = FIXTURES / "log2-abi.json"
            for name in ("log2", "sqrt", "exp2"):
                with self.subTest(name=name):
                    artifact = work / f"{name}-program"
                    selected_request = FIXTURES / f"{name}-request.json"
                    status_file = work / f"{name}-status.json"
                    compiled = _command(
                        "native-compile", "--engine", "merlin_native", "--support", "atlas_tensor",
                        "--snapshot", snapshot, "--request", selected_request, "--abi", abi,
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
