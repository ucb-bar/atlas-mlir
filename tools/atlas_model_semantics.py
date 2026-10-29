"""Check helpers and module/class bindings independently of their callers."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from atlas_model_source import read_source
from atlas_model_source import UnsupportedModelTiming
from atlas_model_source import GUARDS, ast_digest as _digest, canonical, sha256


# These reviewed classes do not inspect method-definition order. Other classes
# (notably ISA metaclasses/dataclasses) retain their exact statement order.
_ORDER_INDEPENDENT = frozenset({"Core", "ArchState", "InstructionMemory",
    "InstructionFetch", "StageData", "Uop", "Program", "InstantiableProgram"})
_NAMESPACE_NAMES = frozenset({"eval", "exec", "compile", "locals", "globals",
                            "vars", "dir", "inspect", "getattr", "setattr", "delattr"})


def _docstrings(node):
    """Strip only real leading docstrings, not arbitrary string expressions."""
    for item in ast.walk(node):
        if isinstance(item, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (item.body and isinstance(item.body[0], ast.Expr)
                    and isinstance(item.body[0].value, ast.Constant)
                    and isinstance(item.body[0].value.value, str)):
                item.body = item.body[1:]
    return node


def _local_names(function):
    """Normalize locals only in simple scopes; preserve arguments and reflected names."""
    body = ast.Module(body=function.body, type_ignores=[])
    nodes = list(ast.walk(body))
    complex_scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                      ast.Lambda, ast.comprehension, ast.Global, ast.Nonlocal,
                      ast.Import, ast.ImportFrom, ast.ExceptHandler, ast.Match)
    if any(isinstance(n, complex_scopes) for n in nodes):
        return function
    if any(isinstance(n, ast.Name) and n.id in _NAMESPACE_NAMES for n in nodes):
        return function
    if any(isinstance(n, ast.Attribute) and (n.attr.startswith("__") or
            n.attr in {"_getframe", "currentframe", "f_locals", "f_globals"}) for n in nodes):
        return function
    arguments = {n.arg for n in ast.walk(function.args) if isinstance(n, ast.arg)}
    names = {}
    for node in nodes:
        if (isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.id not in arguments and node.id not in names):
            names[node.id] = f"<local:{len(names)}>"
    for node in nodes:
        if isinstance(node, ast.Name) and node.id in names:
            node.id = names[node.id]
    return function


def _reorderable(node):
    if not isinstance(node, ast.FunctionDef):
        return False
    if any(not isinstance(d, ast.Name) or d.id not in {"property", "staticmethod"}
           for d in node.decorator_list):
        return False
    annotations = [node.returns, *(arg.annotation for arg in ast.walk(node.args)
                                  if isinstance(arg, ast.arg))]
    if any(value is not None and not isinstance(value, (ast.Name, ast.Constant))
           for value in annotations):
        return False
    return all(value is None or isinstance(value, ast.Constant)
               for value in (*node.args.defaults, *node.args.kw_defaults))


def _normalized(source):
    tree = _docstrings(ast.parse(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            _local_names(node)
        if isinstance(node, ast.ClassDef) and node.name in _ORDER_INDEPENDENT:
            # Reorder only adjacent ordinary definitions; class assignments,
            # aliases, decorators and nonliteral defaults are barriers.
            result, run = [], []
            for member in [*node.body, None]:
                if _reorderable(member):
                    run.append(member)
                    continue
                if len({item.name for item in run}) != len(run):
                    raise UnsupportedModelTiming(f"unsupported semantics: duplicate method in {node.name}")
                result.extend(sorted(run, key=lambda item: item.name))
                run = []
                if member is not None:
                    result.append(member)
            node.body = result
    return tree


def module_contract(source):
    """Digest the helper definitions and all remaining source bindings."""
    tree = _normalized(source)
    helpers = {}

    def collect(body, prefix=""):
        for node in body:
            if isinstance(node, ast.ClassDef):
                collect(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                if name in helpers:
                    raise UnsupportedModelTiming(f"unsupported semantics: duplicate helper {name}")
                helpers[name] = _digest(node)
                # Keep signature/decorators and definition placement in bindings.
                node.body = [ast.Pass()]

    collect(tree.body)
    return sha256(canonical(dict(helpers=helpers, bindings=_digest(tree))))


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
