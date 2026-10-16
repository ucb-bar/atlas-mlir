"""Selected Atlas assembly in the model's Python environment."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


def _run(source: str, root: str | Path | None, beat_bytes: int | None = None) -> dict:
    selected = root or os.environ.get("ATLAS_ASSEMBLER_ROOT")
    if not selected:
        raise unittest.SkipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
    root = Path(selected).resolve()
    model = Path(os.environ.get("NPU_MODEL_DIR", root.parent / "npu-model")).resolve()
    for path in (root / "atlas_asm.py", model / "pyproject.toml", model / "uv.lock",
                 model / "npu_model/util/converter.py"):
        if not path.is_file():
            raise AssertionError(f"missing selected assembler dependency: {path}")
    environment = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if not environment:
        raise AssertionError("set UV_PROJECT_ENVIRONMENT to a prepared model environment")
    environment = Path(environment).resolve()
    if environment.is_relative_to(root.parent) or environment.is_relative_to(model):
        raise AssertionError("UV_PROJECT_ENVIRONMENT must be outside the Atlas/model source trees")
    if not (environment / "bin/python").is_file():
        raise AssertionError(f"missing selected model environment: {environment}")
    try:
        pin = subprocess.check_output(
            ["git", "-C", str(root.parent), "ls-tree", "HEAD", "--", "npu-model"],
            text=True, stderr=subprocess.PIPE).split()
        revision = subprocess.check_output(
            ["git", "-C", str(model), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.PIPE).strip()
        if len(pin) != 4 or revision != pin[2]:
            raise AssertionError(f"selected model {model} does not match Atlas's npu-model gitlink")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", UV_PROJECT_ENVIRONMENT=str(environment),
                   PYTHONPATH=str(model) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        run = subprocess.run(
            ["uv", "run", "--locked", "--no-sync", "--project", str(model), "--python", "3.14",
             "python", str(Path(__file__).resolve()), str(root), str(model),
             "" if beat_bytes is None else str(beat_bytes)],
            input=source, text=True, capture_output=True, check=True, env=env, timeout=120)
        result = json.loads(run.stdout)
        words = result.get("words") if isinstance(result, dict) else None
        if not isinstance(words, list) or any(type(word) is not int or not 0 <= word < 2**32 for word in words):
            raise AssertionError("selected Atlas assembler must return a list of uint32 words")
        return result
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise AssertionError(f"selected Atlas assembler failed: {detail.strip()}") from error


def assemble(source: str, root: str | Path | None = None) -> list[int]:
    return _run(source, root)["words"]


def word(source: str, root: str | Path | None = None) -> int:
    return assemble(source, root)[0]


def inline_golden(source: str, beat_bytes: int, root: str | Path | None = None):
    return _run(source, root, beat_bytes)["golden"]


if __name__ == "__main__":
    root, model, beat = sys.argv[1:]
    if sys.version_info < (3, 14):
        raise RuntimeError("selected Atlas assembler requires Python 3.14")
    from npu_model.util import converter

    if not Path(converter.__file__).resolve().is_relative_to(Path(model)):
        raise RuntimeError(f"selected model differs from imported converter: {converter.__file__}")
    spec = importlib.util.spec_from_file_location("selected_atlas_assembler", Path(root) / "atlas_asm.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = sys.stdin.read()
    result = {"words": module.assemble(source)}
    if beat:
        result["golden"] = module._parse_inline_golden(source, int(beat))
    json.dump(result, sys.stdout)
