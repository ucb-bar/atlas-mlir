"""Model pin checks must admit editable installs without relaxing the pin."""

import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from model_provenance import require_model_revision


def distribution(origin):
    return SimpleNamespace(read_text=lambda name: json.dumps(origin))


class ModelProvenanceTest(unittest.TestCase):
    def editable(self):
        temporary = tempfile.TemporaryDirectory(prefix="atlas model pin ")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        module = root / "npu_model/__init__.py"
        module.parent.mkdir()
        module.write_text("# pinned model fixture\n")
        subprocess.run(["git", "init", "--quiet", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "npu_model"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--quiet",
                "-m",
                "Create model fixture",
            ],
            check=True,
        )
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        origin = distribution(dict(url=root.as_uri(), dir_info=dict(editable=True)))
        return root, module, revision, origin

    def test_editable_checkout_with_encoded_path(self):
        _, module, revision, origin = self.editable()
        require_model_revision(origin, module, revision)

    def test_editable_wrong_revision(self):
        _, module, _, origin = self.editable()
        with self.assertRaisesRegex(ValueError, "expected"):
            require_model_revision(origin, module, "0" * 40)

    def test_editable_modified_source(self):
        _, module, revision, origin = self.editable()
        module.write_text("# modified model\n")
        with self.assertRaisesRegex(ValueError, "tracked changes"):
            require_model_revision(origin, module, revision)

    def test_editable_staged_changes(self):
        root, module, revision, origin = self.editable()
        module.write_text("# staged model changes\n")
        subprocess.run(["git", "-C", str(root), "add", "npu_model"], check=True)
        with self.assertRaisesRegex(ValueError, "tracked changes"):
            require_model_revision(origin, module, revision)

    def test_import_from_another_checkout(self):
        root, _, revision, origin = self.editable()
        with self.assertRaisesRegex(ValueError, "outside"):
            require_model_revision(
                origin, root.parent / "other/npu_model/__init__.py", revision
            )

    def test_pinned_vcs_wheel(self):
        revision = "1" * 40
        origin = distribution(
            dict(
                url="https://example.invalid/npu_model",
                vcs_info=dict(commit_id=revision),
            )
        )
        require_model_revision(origin, "/unused", revision)

    def test_pinned_archive(self):
        revision = "1" * 40
        origin = distribution(
            dict(url=f"https://example.invalid/npu_model/archive/{revision}.zip")
        )
        require_model_revision(origin, "/unused", revision)

    def test_wrong_vcs_commit_cannot_use_url_to_bypass_pin(self):
        revision = "1" * 40
        origin = distribution(
            dict(
                url=f"https://example.invalid/{revision}",
                vcs_info=dict(commit_id="2" * 40),
            )
        )
        with self.assertRaisesRegex(ValueError, "documented"):
            require_model_revision(origin, "/unused", revision)

    def test_missing_provenance(self):
        with self.assertRaisesRegex(ValueError, "documented"):
            require_model_revision(distribution({}), "/unused", "1" * 40)


if __name__ == "__main__":
    unittest.main()
