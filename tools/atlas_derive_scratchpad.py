"""Explicit SRAM contract boundary; source exposes samples, not response events.

No response latency can be recovered from synchronous tensor reads. The legacy
one-edge target assumption is inactive in production: complete derived releases
override the test-only fallback. It is never observed validation.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import os
from pathlib import Path

from atlas_model_diagnostics import UnsupportedModelTiming


class UnsupportedScratchpad(UnsupportedModelTiming):
    """The source no longer has the supported sampling/staging boundary."""


def scratchpad_assumptions():
    # Inactive production compatibility metadata; only legacy test facts use the
    # fallback arithmetic. Production requires complete derived releases.
    return {"scratchpad_read_latency": 1}


def read_source(path):
    path = Path(path)
    if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
        raise UnsupportedScratchpad("prohibited source path")
    if ".." in path.parts:
        raise UnsupportedScratchpad("parent traversal is unsupported")
    path = Path(os.path.abspath(path))
    if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
        raise UnsupportedScratchpad("prohibited source path")
    if any(parent.is_symlink() for parent in reversed((path, *path.parents))):
        raise UnsupportedScratchpad("source symlinks are unsupported")
    return path.read_text()


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


# Reviewed method structure only: numeric DMA store staging is extracted below.
_SHAPES = {('vpu.py', 'VectorExecutionUnit', '_read'): 'e083516d26264797ff20dfd29aba04590a4ce968e7517409cc47aaec58b49bfc',
 ('vpu.py', 'VectorExecutionUnit', '_advance'): '8a8d8ce1964fbfdea57fdd05bcffb421f2f599fc00d51406a5a4e9d23b7ec931',
 ('mxu.py', '_MatrixExecutionUnit', '_sample'): 'a661911a4a2041d6c7fcdb4aa0cb57dce2e9d67893ff76254850b1136df0e76e',
 ('mxu.py', '_MatrixExecutionUnit', 'tick'): '5e30aecc76ffe9b89c745c9e1fa591821560a7c9e3a1f96bf5fb6affce257db8',
 ('lsu.py', 'LoadStoreUnit', '_tick_scalar'): '0b10a168b3ae77b090fd75a869ac26e618395b331420bd139c08ccbe96a8dc87',
 ('lsu.py', 'LoadStoreUnit', '_tick_vector'): '0ec17a2290c9036a9b809ec004bf27aa4eb995d81dca2c483e5fdd4cfa1ce692',
 ('xlu.py', 'CrossLaneExecutionUnit', 'tick'): 'd0d7eed5f87c49e5d4b8d59f1271c535aedb620098eaa01fb55bd38e4f26264a',
 ('dma.py', 'DmaExecutionUnit', 'tick'): '10e7114c0424562758f87c02178d9d6f77b1e3d1a41a90a4c93794f3a4b08944'}


def derive_scratchpad_sources(sources):
    checked = []
    pipeline = None
    for filename, cls, name in _SHAPES:
        node = method(sources[filename], cls, name)
        if filename == "dma.py":
            beats = [item for item in ast.walk(node) if isinstance(item, ast.Call)
                     and isinstance(item.func, ast.Name) and item.func.id == "_StoreBeat"]
            if len(beats) != 1 or len(beats[0].args) != 3:
                raise UnsupportedScratchpad("unsupported DMA SRAM/store-queue staging")
            ready = beats[0].args[2]
            if (not isinstance(ready, ast.BinOp) or not isinstance(ready.op, ast.Add)
                    or not isinstance(ready.left, ast.Name) or ready.left.id != "cycle"
                    or not isinstance(ready.right, ast.Constant) or type(ready.right.value) is not int
                    or not 1 <= ready.right.value < 4096):
                raise UnsupportedScratchpad("unsupported DMA SRAM/store-queue readiness expression")
            pipeline = ready.right.value
            ready.right = ast.Name(id="DERIVED_STORE_PIPELINE", ctx=ast.Load())
        if shape(node) != _SHAPES[(filename, cls, name)]:
            raise UnsupportedScratchpad(f"unsupported scratchpad sampling/response semantics in {cls}.{name}; "
                                        "the model provides no separate SRAM response edge")
        checked.append(f"{filename}:{cls}.{name}")
    return dict(schema="atlas.scratchpad_contract.v1", status="explicit-target-assumption",
                assumptions=scratchpad_assumptions(), response_latency_derived=False,
                dma_store_read_to_request_cycles=pipeline,
                limitation="VPU/MXU/LSU/XLU sample tensors on request edges; DMA combines SRAM and store-queue staging. "
                           "No model-owned executable field separates SRAM response latency from that pipeline. "
                           "scratchpad_read_latency is inactive production compatibility metadata; "
                           "complete derived reservation releases override the legacy test-only fallback.",
                checked_source_methods=checked)


def derive_scratchpad(cfg):
    root = Path(inspect.getfile(cfg.isa)).parent / "hardware"
    return derive_scratchpad_sources({name: read_source(root / name) for name in sorted({key[0] for key in _SHAPES})})
