"""Derive DMA service bounds from model source, configuration, and curves."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from pathlib import Path

from atlas_model_source import (UnsupportedModelTiming, literal, method as _method, module_tree,
                                one as _one)


def _unsupported(field):
    raise UnsupportedModelTiming("unsupported capability: " + field)


def resolve_curve_path(cfg):
    """Resolve the selected file exactly as the native backend does, safely."""
    from npu_model.hardware import memory_backend
    params = cfg.dma_memory_params
    if not isinstance(params, dict):
        _unsupported("dma_memory_params")
    curves = params.get("curves")
    if not isinstance(curves, str) or not curves:
        _unsupported("dma_memory_params.curves")
    path = Path(curves)
    if not path.is_absolute():
        local = (Path.cwd() / path)
        if local.exists():
            path = local
        else:
            path = Path(inspect.getsourcefile(memory_backend)).parent.parent / "configs" / "memory_curves" / path
    return path.absolute()


def _engine_values(tree):
    cls = _one((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DmaExecutionUnit"), "DMA class")
    tick = _method(cls, "tick")
    increment = _one((n for n in ast.walk(tick) if isinstance(n, ast.AugAssign) and
                      isinstance(n.target, ast.Attribute) and n.target.attr == "cycle"), "DMA clock increment")
    if not isinstance(increment.op, ast.Add):
        raise UnsupportedModelTiming("unsupported DMA clock progression")
    edge = literal(increment.value, "DMA timing literal", integral=False, maximum=None)
    entry = _one((n for n in ast.walk(tick) if isinstance(n, ast.Call) and
                  isinstance(n.func, ast.Name) and n.func.id == "_StoreBeat"), "store data queue entry")
    available = entry.args[-1]
    if not (isinstance(available, ast.BinOp) and isinstance(available.op, ast.Add) and
            isinstance(available.left, ast.Name) and available.left.id == "cycle"):
        raise UnsupportedModelTiming("unsupported store data availability")
    delay = literal(available.right, "DMA timing literal", integral=False, maximum=None)
    # tick snapshots the old slots and enqueues launches at the end; the reviewed
    # full AST pins this ordering. Stores first read at the next tick, then wait
    # the availability expression above before issuing their first request.
    return dict(load=edge, store=edge + delay), edge


def derive_curve(cfg):
    from npu_model.hardware import dma, memory_backend
    module_tree(memory_backend, reviewed=True, label="DMA source")
    dma_tree = module_tree(dma, reviewed=True, label="DMA source")
    if cfg.dma_memory_backend != "curve":
        _unsupported("dma_memory_backend")
    path = resolve_curve_path(cfg)
    params = dict(cfg.dma_memory_params)
    curve_bytes = path.read_bytes()
    curve_sha = hashlib.sha256(curve_bytes).hexdigest()
    try:
        data = json.loads(curve_bytes)
    except ValueError as error:
        raise UnsupportedModelTiming("unsupported capability: dma_memory_params.curves") from error
    if not isinstance(data, dict):
        _unsupported("dma_memory_params.curves")
    # Pass the selected bytes to the converter without a second file read.
    params["curves"] = data
    params.setdefault("beat_bytes", cfg.dma_beat_bytes)
    params.setdefault("queue_depth", cfg.dma_max_in_flight)
    try:
        backend = memory_backend.CurveMemoryBackend(**params)
    except (TypeError, ValueError, KeyError, ArithmeticError) as error:
        raise UnsupportedModelTiming("unsupported capability: dma_memory_params: " + str(error)) from error
    for field, supported in (("latency_mode", backend.latency_mode == "unloaded"),
                             ("converge", backend.converge == 1.0),
                             ("enforce_peak", backend.enforce_peak is True)):
        if not supported:
            _unsupported("dma_memory_params." + field)
    positive = (backend.beat_bytes, backend.channels, backend.frequency_ghz, backend.queue_depth,
                backend.window_beats, backend.min_latency,
                cfg.dma_max_in_flight, cfg.dma_num_channels, cfg.dma_max_transfer_bytes)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in positive):
        raise UnsupportedModelTiming("unsupported DMA geometry or scaling")
    if backend.beat_bytes != cfg.dma_beat_bytes:
        raise UnsupportedModelTiming("backend and engine beat sizes disagree")
    if (any(type(value) is not int for value in (cfg.dma_beat_bytes, cfg.dma_max_in_flight,
                                               cfg.dma_num_channels, cfg.dma_max_transfer_bytes)) or
            cfg.dma_max_transfer_bytes % cfg.dma_beat_bytes):
        raise UnsupportedModelTiming("unsupported integral DMA engine geometry")
    startup, issue_interval = _engine_values(dma_tree)
    rows = []
    for pct in backend._pcts:
        points = backend._curves[pct]
        if not 0 <= pct <= 100 or any(not math.isfinite(x) or x <= 0 for point in points for x in point):
            raise UnsupportedModelTiming("unsupported DMA curve coordinates")
        peak, _ = backend.curve_peak(pct)
        rows.append(dict(unloaded_latency_cycles=backend.latency_at(0.0, pct),
                         cycles_per_beat=1.0 / peak))
    if issue_interval != 1:
        raise UnsupportedModelTiming("unsupported DMA issue interval")
    return dict(curve_sha256=curve_sha, beat_bytes=backend.beat_bytes,
                max_transfer_bytes=cfg.dma_max_transfer_bytes, min_latency=backend.min_latency,
                startup_cycles=startup, issue_interval_cycles=issue_interval, curves=rows,
                retire_to_wait_issue_cycles=issue_interval)
