"""Derive reviewed Core edge facts and emit a fixed graph projection descriptor."""
from __future__ import annotations

from atlas_derive_frontend import derive_frontend
from atlas_model_replay import validate_program


def projection_program(source):
    """Instantiate the declared compiler policy using checked source edge facts."""
    from atlas_model_replay import instantiate_program
    return validate_program(instantiate_program(source))


def derive_replay(cfg):
    # The frontend guards complete Core.tick/_tick_s1 ordering and stall logic.
    frontend = derive_frontend(cfg)
    source = dict(schema="atlas.core_edges.v1", clock_step=frontend["ordinary_issue_gap"],
                  phases=["s1_previous_state", "execution_units", "fetch"],
                  dma_wait_observes="previous_edge_flags",
                  implicit_exu_stalls=False,
                  checked_methods=["Core.tick", "Core._tick_s1"],
                  source_contract="atlas.model_frontend.v1")
    return projection_program(source)
