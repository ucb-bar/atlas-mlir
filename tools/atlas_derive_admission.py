"""Check the engine construction and lifetime rules used by scheduling."""
from __future__ import annotations

import ast
from pathlib import Path

from atlas_derive_resources import _source, _named, _method
from atlas_derive_memory import _source as memory_source
from atlas_derive_frontend import derive_frontend
from atlas_model_diagnostics import UnsupportedModelTiming
from atlas_model_schema import sha256
from atlas_model_timing import configuration

CORE_INIT_SHAPE = "5abd67591485003d3f345e7b4f15564b3b3d8a4ff8449ed3efd5a392cf03adf9"


def _vpu_live_ages(cfg, blocks):
    """Read native write_last, which controls issue liveness and retirement."""
    from atlas_derive_vpu import _plans
    final = {name: operation.write_last for name, _, _, operation in _plans(cfg)}
    if set(final) != {name for name, block in blocks.items() if block["engine"] == "vpu"}:
        raise UnsupportedModelTiming("VPU source live endpoint coverage differs from selected operations")
    for name, endpoint in final.items():
        block = blocks[name]
        writes = [value["last_age"] for key, value in block["events"].items()
                  if key.startswith("write")]
        if not writes or max(writes) > endpoint or block["first_free_age"] != endpoint + 1:
            raise UnsupportedModelTiming("VPU source live endpoint differs from lifetime or precedes row writes")
    return final


def derive_admission(model, per_op):
    """Refuse engine routing or VPU lifetime changes the tables cannot describe."""
    model = Path(model)
    cfg, _ = configuration()
    from npu_model.hardware import lsu, xlu
    # Memory extractor guards full control flow, including pre-progress issue.
    memory_source(lsu)
    memory_source(xlu)
    derive_frontend(cfg)
    _, core = _source(model, "core.py")
    init = _method(_named(core, "Core"), "__init__")
    if sha256(ast.dump(init, include_attributes=False).encode()) != CORE_INIT_SHAPE:
        raise UnsupportedModelTiming("unsupported engine construction/routing semantics")
    units = cfg.execution_units
    required = {"VectorExecutionUnit", "CrossLaneExecutionUnit", "LoadStoreUnit"}
    if any(list(units.values()).count(name) != 1 for name in required):
        raise UnsupportedModelTiming("engine admission requires unique independent native units")
    blocks = {block["name"]: block for block in per_op["op_timing"]}
    final = _vpu_live_ages(cfg, blocks)
    if not final:
        raise UnsupportedModelTiming("missing VPU admission coverage")
