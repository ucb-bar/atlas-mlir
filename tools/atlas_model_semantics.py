"""Refuse changes to model modules before importing the selected checkout."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from atlas_model_source import GUARDS, UnsupportedModelTiming, ast_digest as _digest, read_source


def module_contract(source):
    """Digest the module AST, ignoring comments and leading docstrings."""
    tree = ast.parse(source)
    for item in ast.walk(tree):
        if isinstance(item, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (item.body and isinstance(item.body[0], ast.Expr)
                    and isinstance(item.body[0].value, ast.Constant)
                    and isinstance(item.body[0].value.value, str)):
                item.body = item.body[1:]
    return _digest(tree)


def check_python():
    if sys.version_info[:2] != (3, 14):
        raise UnsupportedModelTiming(
            "model timing requires Python 3.14 for reviewed AST fingerprints; "
            f"running {sys.version_info.major}.{sys.version_info.minor}")


def check_model_semantics(model):
    """Refuse unsupported source before importing or executing Core."""
    check_python()
    checked = {}
    for filename, expected in GUARDS["files"].items():
        try:
            source = read_source(Path(model) / "npu_model" / filename)
            actual = module_contract(source)
        except (OSError, SyntaxError) as error:
            raise UnsupportedModelTiming(f"unsupported semantics: {filename}: {error}") from error
        if actual != expected:
            raise UnsupportedModelTiming(f"unsupported semantics: {filename}: module contract changed")
        checked[filename] = actual
    return checked


def source_pins(model):
    """Recompute source contracts without importing the selected checkout."""
    check_python()
    from atlas_derive_vpu import source_shape as vpu_shape
    from atlas_derive_mxu import source_shape as mxu_shape
    root = Path(model) / "npu_model"
    files = {name: module_contract(read_source(root / name)) for name in GUARDS["files"]}
    shapes = {}
    for name in GUARDS["source_shapes"]:
        source = read_source(root / "hardware" / name)
        shapes[name] = (vpu_shape(source) if name == "vpu.py" else
                        mxu_shape(source) if name == "mxu.py" else
                        _digest(ast.parse(source)))
    return dict(files=files, source_shapes=shapes)
