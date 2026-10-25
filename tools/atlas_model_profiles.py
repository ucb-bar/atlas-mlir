"""Named scoring targets; synthetic choices are data, not derived model facts."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

from atlas_model_timing import configuration_values, encoded, sha256


TARGETS = Path(__file__).with_name("model-scoring-targets.json")


@dataclass(frozen=True)
class PeriodicAcceptance:
    """Synthetic acceptance prefix, phased from absolute DMA/backend cycle zero."""
    period_cycles: int
    accepted_prefix_cycles: int

    def __post_init__(self):
        if (type(self.period_cycles) is not int or type(self.accepted_prefix_cycles) is not int
                or self.period_cycles <= 0 or not 0 <= self.accepted_prefix_cycles <= self.period_cycles):
            raise ValueError("invalid periodic memory acceptance pattern")

    def __call__(self, cycle):
        return cycle % self.period_cycles < self.accepted_prefix_cycles


def synthetic_targets():
    return json.loads(TARGETS.read_text())


def bind_synthetic_configuration(cfg, target):
    """Configure the native model; its fixed backend still owns all execution."""
    from npu_model.hardware.memory_backend import make_memory_backend

    cfg.dma_memory_backend = target["dma_memory_backend"]
    cfg.dma_memory_params = deepcopy(target["dma_memory_params"])
    cfg.dma_memory_params["accept_fn"] = PeriodicAcceptance(**target["accept_pattern"])
    # Validate the parameters before recording a receipt; Core creates its own backend.
    make_memory_backend(cfg)


def scoring_configuration_receipt(cfg, profile, identity, *, synthetic_target=None):
    values = deepcopy(configuration_values(cfg))
    if synthetic_target is not None:
        # Bind declarative callback inputs because Python callables lack a stable JSON identity.
        callback = values["dma_memory_params"]["accept_fn"]
        values["dma_memory_params"]["accept_fn"] = {
            "adapter": "absolute-cycle-periodic-prefix-v1",
            "period_cycles": callback.period_cycles,
            "accepted_prefix_cycles": callback.accepted_prefix_cycles,
        }
    payload = dict(schema="atlas.model_scoring_configuration.v1", profile=profile,
                   model_identity=identity, configuration=values,
                   provenance="synthetic-rtl-harness-target" if synthetic_target is not None else "selected-model-configuration")
    if synthetic_target is not None:
        payload["synthetic_target"] = deepcopy(synthetic_target)
    return dict(payload, sha256=sha256(encoded(payload)))
