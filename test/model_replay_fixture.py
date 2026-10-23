"""Explicit synthetic projected-Core contract for pure unit tests."""
from atlas_model_replay import instantiate_program, semantic_digest


def replay_fixture():
    return instantiate_program(dict(schema="atlas.core_edges.v1", clock_step=1,
        phases=["s1_previous_state", "execution_units", "fetch"],
        dma_wait_observes="previous_edge_flags", implicit_exu_stalls=False,
        checked_methods=["Core.tick", "Core._tick_s1"], source_contract="atlas.model_frontend.v1"))


def replay_identity():
    return semantic_digest(replay_fixture())
