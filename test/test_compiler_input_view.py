"""The target consumes only checked manifest provenance into a compiler view."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "python"))

from xdsl.dialects.builtin import ModuleOp, StringAttr
from atlas_mlir_native.capture import compiler_input_view
from merlin.frontends.compile_inputs import StagedCompileInputs
from merlin.xdsl_dialects._common import text


class CompilerInputViewTests(unittest.TestCase):
    def fixture(self):
        manifest = {"schema": "m2m.quantization_manifest.v2", "adapter_id": "atlas_fp8"}
        digest = sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        module = ModuleOp([], attributes={"prov.level": StringAttr("linalg-on-tensors")})
        baseline = text(module, generic=True) + "\n"
        module.attributes["prov.quantization_manifest_sha256"] = StringAttr(digest)
        module.attributes["prov.quantization"] = StringAttr("external:atlas_fp8")
        archived = text(module, generic=True) + "\n"
        staged = StagedCompileInputs(archived, "forward", None, None, "a" * 64,
                                    sha256(archived.encode()).hexdigest(), None)
        return staged, manifest, baseline

    def test_only_two_selected_attributes_are_consumed_and_archive_is_unchanged(self):
        staged, manifest, baseline = self.fixture()
        compiler, receipt = compiler_input_view(staged, manifest)
        self.assertEqual(compiler.program, baseline)
        self.assertEqual(compiler.program_sha256, sha256(baseline.encode()).hexdigest())
        self.assertEqual(receipt["byte_delta"], 146)
        self.assertEqual(receipt["staged_program_sha256"], staged.program_sha256)
        self.assertEqual(receipt["compiler_program_sha256"], compiler.program_sha256)
        self.assertIn("prov.quantization_manifest_sha256", staged.program)
        self.assertEqual(compiler.argument_bindings, staged.argument_bindings)

    def test_tampered_staged_bytes_refused_before_view(self):
        staged, manifest, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "differs from validated staged"):
            compiler_input_view(replace(staged, program=staged.program + " "), manifest)

    def test_other_manifest_refused_before_view(self):
        staged, manifest, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "digest differs"):
            compiler_input_view(staged, dict(manifest, tampered=True))

    def test_changed_scheme_is_refused(self):
        staged, manifest, _ = self.fixture()
        program = staged.program.replace("external:atlas_fp8", "external:other")
        with self.assertRaisesRegex(ValueError, "scheme differs"):
            compiler_input_view(replace(staged, program=program,
                program_sha256=sha256(program.encode()).hexdigest()), manifest)

    def test_non_roundtripping_text_refused(self):
        staged, manifest, _ = self.fixture()
        program = staged.program + "\n"
        with self.assertRaisesRegex(ValueError, "roundtrip"):
            compiler_input_view(replace(staged, program=program,
                program_sha256=sha256(program.encode()).hexdigest()), manifest)


if __name__ == "__main__":
    unittest.main()
