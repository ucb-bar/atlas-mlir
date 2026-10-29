"""Shared AST helpers for timing source extraction."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path

from atlas_model_diagnostics import UnsupportedModelTiming


class UnsupportedScratchpad(UnsupportedModelTiming):
    """The source no longer has the supported sampling/staging boundary."""


def read_source(path):
    return Path(path).read_text()


def method(source, cls, name):
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls]
    methods = [node for item in classes for node in item.body
               if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(methods) != 1:
        raise UnsupportedScratchpad(f"expected one source method {cls}.{name}")
    return methods[0]


def shape(node):
    class StripDocstrings(ast.NodeTransformer):
        def visit_Expr(self, item):
            if isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                return None
            return self.generic_visit(item)
    node = StripDocstrings().visit(deepcopy(node))
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
