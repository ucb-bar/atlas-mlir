"""Derive reviewed Core edge facts and emit one common graph projection IR.

This is a projected virtual graph interpreter, not full instruction/Core replay.
The source contract refuses changed control structures before publishing facts.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from atlas_derive_frontend import derive_frontend
from atlas_derive_scratchpad import method, read_source
from atlas_model_diagnostics import UnsupportedModelTiming
from atlas_model_replay import SCHEMA, validate_program


def projection_program(source):
    """Instantiate the declared compiler policy using checked source edge facts."""
    from atlas_model_replay import instantiate_program
    return validate_program(instantiate_program(source))


def derive_replay(cfg):
    # This independent frontend extraction checks complete Core.tick/_tick_s1
    # source shapes, preserving S1-before-EXU and issue-before-fetch ordering.
    frontend = derive_frontend(cfg)
    root = Path(inspect.getfile(cfg.isa)).parent
    tick = method(read_source(root / "hardware" / "core.py"), "Core", "tick")
    calls = [(item.lineno, ast.unparse(item.func)) for item in ast.walk(tick) if isinstance(item, ast.Call)]
    def line(name):
        found = [position for position, target in calls if target == name]
        if len(found) != 1:
            raise UnsupportedModelTiming("unsupported Core transition ordering: " + name)
        return found[0]
    if not line("self._tick_s1") < line("exu.tick") < line("self.ifu.tick"):
        raise UnsupportedModelTiming("unsupported Core transition phase ordering")
    # Core reads DMA flags in S1 before EXU completion updates on each edge.
    # The frontend shape also excludes implicit engine/dependency stalls.
    source = dict(schema="atlas.core_edges.v1", clock_step=frontend["ordinary_issue_gap"],
                  phases=["s1_previous_state", "execution_units", "fetch"],
                  dma_wait_observes="previous_edge_flags",
                  implicit_exu_stalls=False,
                  checked_methods=["Core.tick", "Core._tick_s1"],
                  source_contract="atlas.model_frontend.v1")
    return projection_program(source)
