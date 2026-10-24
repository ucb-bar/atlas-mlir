"""Reviewed helper closure for fatal errors, frontend dispatch and storage.

This is a source compatibility boundary, not a simulator or timing estimator.
Each helper is checked independently, together with module/class bindings: a
caller's fingerprint never stands for its callees. Engine timing interpreters
remain responsible for their separately inventoried source domains.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path

from atlas_derive_scratchpad import read_source
from atlas_model_diagnostics import UnsupportedModelTiming


class UnsupportedModelSemantics(UnsupportedModelTiming):
    """A behavior-bearing helper has no reviewed semantic interpretation."""


CONTRACT = Path(__file__).with_name("model-semantics.json")
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
    """Alpha-normalize ordinary locals only in an unambiguous single scope.

Arguments stay named (keyword calls and runtime signature inspection matter).
Closures, comprehensions, pattern/exception bindings and namespace reflection
retain literal names. No expression, assertion, logging call or annotation is
removed. New globals cannot collide with the canonical, non-Python identifiers.
"""
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
                    raise UnsupportedModelSemantics(f"unsupported semantics: duplicate method in {node.name}")
                result.extend(sorted(run, key=lambda item: item.name))
                run = []
                if member is not None:
                    result.append(member)
            node.body = result
    return tree


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def module_contract(source):
    """Return independent helper fingerprints and all remaining source bindings."""
    tree = _normalized(source)
    helpers = {}

    def collect(body, prefix=""):
        for node in body:
            if isinstance(node, ast.ClassDef):
                collect(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                if name in helpers:
                    raise UnsupportedModelSemantics(f"unsupported semantics: duplicate helper {name}")
                helpers[name] = _digest(node)
                # Keep signature/decorators and definition placement in bindings.
                node.body = [ast.Pass()]

    collect(tree.body)
    return dict(helpers=helpers, bindings=_digest(tree))


def check_model_semantics(model):
    """Check selected source before importing/allocating/executing native Core.

The emitted inventory is provenance, bound by the existing model source digest
in atlas.derived_facts.v1; it adds no scheduling value or measured timing fact.
"""
    if sys.version_info[:2] != (3, 14):
        raise UnsupportedModelSemantics(
            "model timing requires Python 3.14 for reviewed AST fingerprints; "
            f"running {sys.version_info.major}.{sys.version_info.minor}")
    contract = json.loads(read_source(CONTRACT))
    checked = {}
    for filename, expected in contract["files"].items():
        try:
            source = read_source(Path(model) / "npu_model" / filename)
            actual = module_contract(source)
        except (OSError, SyntaxError) as error:
            raise UnsupportedModelSemantics(f"unsupported semantics: {filename}: {error}") from error
        for helper in sorted(set(expected["helpers"]) | set(actual["helpers"])):
            if actual["helpers"].get(helper) != expected["helpers"].get(helper):
                raise UnsupportedModelSemantics(f"unsupported semantics: {filename}:{helper}")
        if actual["bindings"] != expected["bindings"]:
            raise UnsupportedModelSemantics(f"unsupported semantics: {filename}:module/class bindings")
        checked[filename] = actual
    return dict(schema=contract["schema"], files=checked, dependencies=deepcopy(contract["dependencies"]),
                binding="model.source_sha256 in atlas.derived_facts.v1 semantic payload")
