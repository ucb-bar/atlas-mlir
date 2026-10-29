"""Derive DMA service bounds from model source, configuration, and curves."""
from __future__ import annotations

from atlas_derive_scratchpad import read_source as _read
from atlas_model_diagnostics import UnsupportedModelTiming

import ast
import hashlib
import inspect
import json
import math
from pathlib import Path

from atlas_derive_mxu import _one, _method
from atlas_model_schema import SOURCE_SHAPES, ast_digest


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


def _positive(value, field, *, integral=False):
    valid_type = type(value) is int if integral else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value <= 0:
        _unsupported(field)


def _check_curve_configuration(cfg, params, data):
    known = {"curves", "window_beats", "converge", "latency_mode", "knee", "frequency_ghz",
             "beat_bytes", "channels", "enforce_peak", "queue_depth", "min_latency"}
    for key in params:
        if key not in known:
            _unsupported("dma_memory_params." + str(key))
    for field in ("window_beats", "beat_bytes", "channels", "queue_depth", "min_latency"):
        if field in params and params[field] is not None:
            _positive(params[field], "dma_memory_params." + field, integral=True)
        elif field in params and field != "channels":
            _unsupported("dma_memory_params." + field)
    for field in ("frequency_ghz", "knee"):
        if params.get(field) is not None:
            _positive(params[field], "dma_memory_params." + field)
    if params.get("knee", 0.9) is None:
        _unsupported("dma_memory_params.knee")
    if params.get("converge", 1.0) != 1.0 or type(params.get("converge", 1.0)) not in (int, float):
        _unsupported("dma_memory_params.converge")
    if params.get("enforce_peak", True) is not True:
        _unsupported("dma_memory_params.enforce_peak")
    mode = params.get("latency_mode")
    if (data.get("latencyMode", "knee") if mode is None else mode) != "unloaded":
        _unsupported("dma_memory_params.latency_mode")
    if params.get("beat_bytes", cfg.dma_beat_bytes) != cfg.dma_beat_bytes:
        _unsupported("dma_memory_params.beat_bytes")
    for field in ("measuredChannels", "accessBytes"):
        _positive(data.get(field, 1 if field == "measuredChannels" else 64), "curve." + field, integral=True)
    if params.get("frequency_ghz") is None:
        _positive(data.get("frequencyGHz", 1.0), "curve.frequencyGHz")
    rows = data.get("curves")
    if not isinstance(rows, dict) or not rows:
        _unsupported("curve.curves")
    for key, points in rows.items():
        try:
            pct = float(key)
        except (TypeError, ValueError):
            _unsupported("curve.read_percentage")
        if not math.isfinite(pct) or not 0 <= pct <= 100:
            _unsupported("curve.read_percentage")
        if not isinstance(points, list) or not points:
            _unsupported("curve.points")
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                _unsupported("curve.points")
            for value in point:
                _positive(value, "curve.points")


def _review(module):
    path = Path(inspect.getsourcefile(module))
    source = _read(path)
    tree = ast.parse(source)
    shape = ast_digest(tree)
    # One AST digest pins the DMA/backend ordering used by the service bound.
    if shape != SOURCE_SHAPES.get(path.name):
        raise UnsupportedModelTiming("unsupported DMA source semantics: " + path.name)
    return tree


def _literal(node):
    value = ast.literal_eval(node)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise UnsupportedModelTiming("unsupported DMA timing literal")
    return value


def _engine_values(tree):
    cls = _one((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DmaExecutionUnit"), "DMA class")
    tick = _method(cls, "tick")
    increment = _one((n for n in ast.walk(tick) if isinstance(n, ast.AugAssign) and
                      isinstance(n.target, ast.Attribute) and n.target.attr == "cycle"), "DMA clock increment")
    if not isinstance(increment.op, ast.Add):
        raise UnsupportedModelTiming("unsupported DMA clock progression")
    edge = _literal(increment.value)
    entry = _one((n for n in ast.walk(tick) if isinstance(n, ast.Call) and
                  isinstance(n.func, ast.Name) and n.func.id == "_StoreBeat"), "store data queue entry")
    available = entry.args[-1]
    if not (isinstance(available, ast.BinOp) and isinstance(available.op, ast.Add) and
            isinstance(available.left, ast.Name) and available.left.id == "cycle"):
        raise UnsupportedModelTiming("unsupported store data availability")
    delay = _literal(available.right)
    # tick snapshots the old slots and enqueues launches at the end; the reviewed
    # full AST pins this ordering. Stores first read at the next tick, then wait
    # the availability expression above before issuing their first request.
    return dict(load=edge, store=edge + delay), edge


def _proposal_contract(rows, minimum, interval):
    """Bound beat service under the guarded DMA/backend and Core tick ordering."""
    maximum = max(minimum, *(r["unloaded_latency_cycles"] for r in rows),
                  *(r["cycles_per_beat"] for r in rows))
    maximum_cycle = 2147483647
    if not math.isfinite(maximum) or maximum > maximum_cycle:
        raise UnsupportedModelTiming("unsupported DMA proposal arithmetic domain")
    # Do not subtract the backend rounding epsilon here: IEEE addition followed
    # by subtraction can round across an integer even on small cycle counts.
    # Rounding upward without that reduction preserves the service envelope.
    quantum = math.ceil(maximum) + interval
    if type(interval) is not int or interval != 1 or minimum < interval or quantum <= interval:
        raise UnsupportedModelTiming("unsupported DMA bound service/positive response latency")
    if quantum > maximum_cycle:
        raise UnsupportedModelTiming("unsupported DMA proposal arithmetic domain")
    # Compiler policy: serialize DMA estimates by one cycle between commands.
    intercommand_serialization_cycles = 1
    return dict(retire_to_wait_issue_cycles=interval, per_beat_bound_cycles=quantum,
                policy=dict(intercommand_serialization_cycles=intercommand_serialization_cycles))


def derive_curve(cfg):
    from npu_model.hardware import dma, memory_backend
    backend_tree = _review(memory_backend)
    dma_tree = _review(dma)
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
    _check_curve_configuration(cfg, params, data)
    # Pass the already checked bytes to the converter: no second implicit file IO.
    params["curves"] = data
    params.setdefault("beat_bytes", cfg.dma_beat_bytes)
    params.setdefault("queue_depth", cfg.dma_max_in_flight)
    backend = memory_backend.CurveMemoryBackend(**params)
    if backend.latency_mode != "unloaded" or backend.converge != 1.0 or not backend.enforce_peak:
        raise UnsupportedModelTiming("DMA proposal facts support unloaded, converge=1, peak-paced curves only")
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
    proposal = _proposal_contract(rows, backend.min_latency, issue_interval)
    return dict(curve_sha256=curve_sha, beat_bytes=backend.beat_bytes,
                max_transfer_bytes=cfg.dma_max_transfer_bytes, min_latency=backend.min_latency,
                startup_cycles=startup, issue_interval_cycles=issue_interval, curves=rows,
                proposal_contract=proposal)
