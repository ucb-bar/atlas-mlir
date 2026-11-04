"""Check pinned model provenance for wheel, archive and editable installs."""

import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
from urllib.request import url2pathname


def require_model_revision(distribution, module_file, expected):
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    url = origin.get("url", "")
    if origin.get("dir_info", {}).get("editable"):
        # Editable VCS installs record only a file URL in PEP 610 metadata.
        # Verify the checkout actually supplying the imported module instead.
        parsed = urlsplit(url)
        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
            raise ValueError("editable npu_model needs a local checkout URL")
        root = Path(url2pathname(parsed.path)).resolve()
        if not Path(module_file).resolve().is_relative_to(root):
            raise ValueError("imported npu_model is outside its editable checkout")

        def git(*args):
            result = subprocess.run(
                ["git", "-C", str(root), *args], text=True, capture_output=True
            )
            if result.returncode:
                raise ValueError(
                    f"cannot verify editable npu_model: {result.stderr.strip()}"
                )
            return result.stdout.strip()

        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            raise ValueError("editable npu_model must identify its own Git checkout")
        observed = git("rev-parse", "HEAD")
        if observed != expected:
            raise ValueError(f"npu_model revision {observed}; expected {expected}")
        result = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"],
            capture_output=True,
        )
        if result.returncode:
            raise ValueError(
                "editable npu_model has tracked changes or cannot be verified"
            )
        return

    commit = origin.get("vcs_info", {}).get("commit_id")
    if commit == expected or (commit is None and expected in url):
        return
    raise ValueError(
        f"install the documented npu_model revision {expected}; origin is {origin}"
    )
