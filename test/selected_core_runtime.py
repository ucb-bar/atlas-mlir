"""Explicit halt binding for the source-linked standalone AtlasCore tests.

The selected ScalarCore assigns io.halted := halt_now. Optimization can remove
the public io_halted port while retaining scalar/halt_now. Current ModeLIR
validates the named state before peeking; a Python peek alias alone is therefore
insufficient. The historical runner has no halt_signal argument and uses the
alias path. Neither path supplies engine-drain or integrated-SoC evidence.
"""

from __future__ import annotations

import inspect
import json
from collections import deque
import os
from pathlib import Path
import stat


HALT_SIGNAL = "scalar/halt_now"


def _allowed_path(value) -> Path:
    path = Path(value)
    def check(candidate):
        if any("hammer" in part.lower() or "vlsi" in part.lower()
               for part in candidate.parts):
            raise ValueError("restricted selected-core artifact path")
    check(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    check(path)
    pending, current, links = deque(path.parts[1:]), Path(path.anchor), set()
    while pending:
        part = pending.popleft()
        if part == ".":
            continue
        if part == "..":
            current = current.parent
            continue
        candidate = current / part
        check(candidate)
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            # Fake-core contract tests use a nonexistent library. Real callers
            # still require a built library; path checking never builds one.
            current = candidate
            continue
        if stat.S_ISLNK(mode):
            if candidate in links or len(links) >= 64:
                raise ValueError("recursive selected-core artifact symlink")
            links.add(candidate)
            target = Path(os.readlink(candidate))
            check(target)  # Reject before lstat/readlink of any target component.
            if target.is_absolute():
                current = Path(target.anchor)
                pending.extendleft(reversed(target.parts[1:]))
            else:
                pending.extendleft(reversed(target.parts))
        else:
            current = candidate
    return current


def _validate_halt(manifest_path) -> None:
    manifest = json.loads(_allowed_path(manifest_path).read_text())
    if not isinstance(manifest, list) or len(manifest) != 1:
        raise ValueError("selected-core manifest must contain exactly one model")
    model = manifest[0]
    if not isinstance(model, dict) or model.get("name") != "AtlasCore":
        raise ValueError("selected-core manifest must name AtlasCore")
    states = model.get("states")
    if not isinstance(states, list) or not all(isinstance(s, dict) for s in states):
        raise ValueError("selected-core manifest requires a state list")
    matches = [s for s in states if s.get("name") == HALT_SIGNAL]
    if (len(matches) != 1 or type(matches[0].get("numBits")) is not int
            or matches[0]["numBits"] != 1):
        raise ValueError(f"selected-core manifest requires one observed i1 {HALT_SIGNAL}")


def run_selected_program(cosim_atlas, model_path, manifest_path, words, **kwargs):
    """Use ModeLIR's runner with the selected core's source-proven halt state.

    Existing caller-specific CosimCore subclasses remain in force, including
    their guards for optimized-away DMA response fields. Modern runners receive
    an explicit halt_signal. Historical runners receive a scoped peek alias;
    failures propagate without retrying or guessing another completion signal.
    The caller must provide a matching native library and state manifest.
    """
    model_path = _allowed_path(model_path)
    manifest_path = _allowed_path(manifest_path)
    _validate_halt(manifest_path)
    requested = kwargs.pop("halt_signal", HALT_SIGNAL)
    if requested != HALT_SIGNAL:
        raise ValueError(f"selected-core halt signal must be {HALT_SIGNAL}")
    runner = cosim_atlas.run_program
    parameter = inspect.signature(runner).parameters.get("halt_signal")
    if parameter is not None:
        if parameter.kind not in (inspect.Parameter.KEYWORD_ONLY,
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD):
            raise TypeError("ModeLIR halt_signal must accept a keyword argument")
        return runner(model_path, manifest_path, words,
                      halt_signal=HALT_SIGNAL, **kwargs)

    original = cosim_atlas.CosimCore

    class SelectedHaltCore(original):
        def peek(self, name):
            return super().peek(HALT_SIGNAL if name == "io_halted" else name)

    cosim_atlas.CosimCore = SelectedHaltCore
    try:
        return runner(model_path, manifest_path, words, **kwargs)
    finally:
        cosim_atlas.CosimCore = original
