"""Derive proposal-only curve DMA facts from the reviewed source and curve bytes.

Derivation constructs the backend's pure configuration converter, but never issues
beats, calls a tick, or allocates architectural state. Numerical observations live
in atlas_validate_curve.py and cannot supply values to this extractor.
"""
from __future__ import annotations

from atlas_model_diagnostics import UnsupportedModelTiming

import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

from atlas_derive_mxu import _read, _one, _method
from atlas_model_schema import LOCK, SOURCE_SHAPES, ast_digest


class UnsupportedCurve(UnsupportedModelTiming):
    """Source, curve or configuration has no reviewed DMA interpretation."""


def _unsupported(field):
    raise UnsupportedCurve("unsupported capability: " + field)


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
        local = _safe(Path.cwd() / path)
        if local.exists():
            path = local
        else:
            path = Path(inspect.getsourcefile(memory_backend)).parent.parent / "configs" / "memory_curves" / path
    return _safe(path)


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
    if params.get("knee", 0.9) is None or params.get("knee", 0.9) > 1:
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


def _safe(path):
    path = Path(path)
    if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
        raise UnsupportedCurve("prohibited DMA source or curve path")
    if ".." in path.parts:
        raise ValueError("parent traversal is unsupported")
    path = Path(os.path.abspath(path))
    if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
        raise UnsupportedCurve("prohibited DMA source or curve path")
    # Check root toward leaf, so no stat follows a previously unchecked ancestor.
    for parent in reversed((path, *path.parents)):
        if parent.is_symlink():
            raise UnsupportedCurve("DMA source or curve path contains a symlink")
    return path


def _review(module):
    path = _safe(inspect.getsourcefile(module))
    source = _read(path)
    tree = ast.parse(source)
    shape = ast_digest(tree)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if (shape != SOURCE_SHAPES.get(path.name) or
            digest != LOCK.get("dma_source_sha256", {}).get(path.name)):
        raise UnsupportedCurve("unsupported DMA source semantics or bytes: " + path.name)
    return tree, digest


def _literal(node):
    value = ast.literal_eval(node)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise UnsupportedCurve("unsupported DMA timing literal")
    return value


def _engine_values(tree):
    cls = _one((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DmaExecutionUnit"), "DMA class")
    tick = _method(cls, "tick")
    increment = _one((n for n in ast.walk(tick) if isinstance(n, ast.AugAssign) and
                      isinstance(n.target, ast.Attribute) and n.target.attr == "cycle"), "DMA clock increment")
    if not isinstance(increment.op, ast.Add):
        raise UnsupportedCurve("unsupported DMA clock progression")
    edge = _literal(increment.value)
    entry = _one((n for n in ast.walk(tick) if isinstance(n, ast.Call) and
                  isinstance(n.func, ast.Name) and n.func.id == "_StoreBeat"), "store data queue entry")
    available = entry.args[-1]
    if not (isinstance(available, ast.BinOp) and isinstance(available.op, ast.Add) and
            isinstance(available.left, ast.Name) and available.left.id == "cycle"):
        raise UnsupportedCurve("unsupported store data availability")
    delay = _literal(available.right)
    # tick snapshots the old slots and enqueues launches at the end; the reviewed
    # full AST pins this ordering. Stores first read at the next tick, then wait
    # the availability expression above before issuing their first request.
    return dict(load=edge, store=edge + delay), edge


def derive_curve(cfg):
    from npu_model.hardware import dma, memory_backend
    from atlas_derive_mxu import UnsupportedMXU
    try:
        backend_tree, backend_sha = _review(memory_backend)
        dma_tree, dma_sha = _review(dma)
    except UnsupportedMXU as error:
        raise UnsupportedCurve(str(error)) from error
    if cfg.dma_memory_backend != "curve":
        _unsupported("dma_memory_backend")
    path = resolve_curve_path(cfg)
    params = dict(cfg.dma_memory_params)
    curve_bytes = path.read_bytes()
    curve_sha = hashlib.sha256(curve_bytes).hexdigest()
    try:
        data = json.loads(curve_bytes)
    except ValueError as error:
        raise UnsupportedCurve("unsupported capability: dma_memory_params.curves") from error
    if not isinstance(data, dict):
        _unsupported("dma_memory_params.curves")
    _check_curve_configuration(cfg, params, data)
    # Pass the already checked bytes to the converter: no second implicit file IO.
    params["curves"] = data
    params.setdefault("beat_bytes", cfg.dma_beat_bytes)
    params.setdefault("queue_depth", cfg.dma_max_in_flight)
    backend = memory_backend.CurveMemoryBackend(**params)
    if backend.latency_mode != "unloaded" or backend.converge != 1.0 or not backend.enforce_peak:
        raise UnsupportedCurve("DMA proposal facts support unloaded, converge=1, peak-paced curves only")
    positive = (backend.beat_bytes, backend.channels, backend.frequency_ghz, backend.queue_depth,
                backend.window_beats, backend.min_latency,
                cfg.dma_max_in_flight, cfg.dma_num_channels, cfg.dma_max_transfer_bytes)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in positive):
        raise UnsupportedCurve("unsupported DMA geometry or scaling")
    if backend.beat_bytes != cfg.dma_beat_bytes:
        raise UnsupportedCurve("backend and engine beat sizes disagree")
    if (any(type(value) is not int for value in (cfg.dma_beat_bytes, cfg.dma_max_in_flight,
                                               cfg.dma_num_channels, cfg.dma_max_transfer_bytes)) or
            cfg.dma_max_transfer_bytes % cfg.dma_beat_bytes):
        raise UnsupportedCurve("unsupported integral DMA engine geometry")
    startup, issue_interval = _engine_values(dma_tree)
    backend_cls = _one((n for n in backend_tree.body if isinstance(n, ast.ClassDef) and n.name == "CurveMemoryBackend"), "curve class")
    issue = _method(backend_cls, "issue")
    rounding = _one((n for n in ast.walk(issue) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "ceil"), "ready cycle ceiling")
    if not isinstance(rounding.args[0], ast.BinOp) or not isinstance(rounding.args[0].op, ast.Sub):
        raise UnsupportedCurve("unsupported ready cycle rounding")
    epsilon = _literal(rounding.args[0].right)
    reset = _method(backend_cls, "reset")
    ready_reset = _one((n for n in ast.walk(reset) if isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Attribute) and t.attr == "_last_ready" for t in n.targets)),
                      "fractional ready reset")
    reset_last_ready = _literal(ready_reset.value)
    rows = []
    for pct in backend._pcts:
        points = backend._curves[pct]
        if not 0 <= pct <= 100 or any(not math.isfinite(x) or x <= 0 for point in points for x in point):
            raise UnsupportedCurve("unsupported DMA curve coordinates")
        peak, _ = backend.curve_peak(pct)
        rows.append(dict(read_percentage=pct, unloaded_latency_cycles=backend.latency_at(0.0, pct),
                         peak_beats_per_cycle=peak, cycles_per_beat=1.0 / peak,
                         points=[list(point) for point in points]))
    return dict(schema="atlas.dma_timing.v1", backend="curve", resolved=True, estimate_only=True,
                curve_sha256=curve_sha, source_sha256={"memory_backend.py": backend_sha, "dma.py": dma_sha},
                source_shapes={name: SOURCE_SHAPES[name] for name in ("memory_backend.py", "dma.py")},
                beat_bytes=backend.beat_bytes, max_in_flight=cfg.dma_max_in_flight,
                num_channels=cfg.dma_num_channels, max_transfer_bytes=cfg.dma_max_transfer_bytes,
                queue_depth=backend.queue_depth, window_beats=backend.window_beats,
                frequency_ghz=backend.frequency_ghz, channels=backend.channels,
                measured_channels=int(data.get("measuredChannels", 1)), access_bytes=backend.access_bytes,
                latency_mode=backend.latency_mode, converge=backend.converge, knee=backend.knee,
                enforce_peak=backend.enforce_peak, min_latency=backend.min_latency,
                startup_cycles=startup, issue_interval_cycles=issue_interval,
                rounding_epsilon_cycles=epsilon, curves=rows,
                reset_last_ready_cycles=reset_last_ready,
                semantics=dict(history="last_window_beats_issued_including_current_beat",
                               curve_selection="nearest_read_percentage_tie_lower",
                               acceptance="pending_count_below_queue_depth_including_presented_response",
                               ready="max_issue_plus_unloaded_latency_and_previous_fractional_ready_plus_curve_spacing",
                               response="oldest_ready_then_issue_order_held_until_pop",
                               history_lifetime="reset_only_not_command_boundary",
                               address_dependence="none",
                               engine="registered_slots_and_in_flight;D_before_A;retire_after_last_ack"),
                limits=["Proposal estimate only; explicit DMA completion waits remain required.",
                        "Full native model scoring determines candidate selection.",
                        "Isolated startup assumes no VMEM grant denial and an empty engine/backend.",
                        "Mixed command streams retain issue history and fractional response pacing across commands.",
                        "No off-chip bank mapping; channels scale aggregate curve bandwidth."])
