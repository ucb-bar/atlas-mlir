#!/usr/bin/env python3
"""Extract compiler timing facts from npu_model source and configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from atlas_model_diagnostics import UnsupportedModelTiming, model_timing_cli
from atlas_model_schema import LOCK
from atlas_model_capabilities import check_configuration
from atlas_model_semantics import check_model_semantics
from atlas_model_timing import (configuration, encoded, isa_cases, numerical_data,
                                sha256, source_digest, validate_derived)


def identity(root):
    source, sources = source_digest(root)
    numerical, data = numerical_data(root)
    revision = LOCK["model_revision"] if source == LOCK["source_sha256"] else source[:40]
    kind = "locked_source" if source == LOCK["source_sha256"] else "source_sha256_prefix"
    return dict(revision=revision, revision_kind=kind, source_sha256=source,
                numerical_data_sha256=numerical), sources, data


def write(directory, name, value):
    path = directory / name
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return sha256(path.read_bytes())


def regenerate(model, out):
    model, out = Path(model).absolute(), Path(out).absolute()
    if out.exists():
        raise ValueError("--out must be a new directory")
    check_model_semantics(model)
    sys.path.insert(0, str(model))
    import torch
    from atlas_derive_vpu import derive_vpu
    from atlas_derive_mxu import derive_mxu, derive_weight_timing
    from atlas_derive_memory import derive_memory
    from atlas_derive_curve import derive_curve
    from atlas_derive_frontend import derive_frontend
    from atlas_derive_arch import derive_arch
    from atlas_derive_resources import derive_resources
    from atlas_derive_costs import derive_costs

    torch.set_num_threads(1)
    cfg, config = configuration()
    check_configuration(cfg)
    architecture = derive_arch(cfg)
    blocks, releases = {}, {}
    for extract in (derive_vpu, derive_mxu, derive_memory):
        timing, reservations = extract(cfg)
        blocks.update(timing)
        releases.update(reservations)
    derive_costs(cfg, architecture)
    names = [name for name, _, _ in isa_cases(cfg)]
    if set(blocks) != set(names) or set(releases) != set(blocks):
        raise UnsupportedModelTiming("source derivation does not cover the selected ISA")
    model_values = dict(identity(model)[0], configuration=config,
                        configuration_json=encoded(config).decode(),
                        configuration_sha256=sha256(encoded(config)))
    payload = dict(schema="atlas.model_timing.v1", model=model_values,
                   geometry=dict(rows=cfg.arch_state_config.mrf_depth,
                                 columns=cfg.arch_state_config.mrf_width,
                                 mreg_count=cfg.arch_state_config.num_m_registers),
                   architecture=architecture, age_convention="issue_edge_zero",
                   observation=dict(accepted=True, drained=True),
                   op_timing=[blocks[name] for name in names], reservation_releases=releases,
                   dma_timing=derive_curve(cfg), mxu_weight_timing=derive_weight_timing(cfg),
                   frontend_timing=derive_frontend(cfg), resources=derive_resources(model))
    # The central Core contract pins routing; each independent engine occurs once.
    required = {"VectorExecutionUnit", "CrossLaneExecutionUnit", "LoadStoreUnit"}
    if any(list(cfg.execution_units.values()).count(name) != 1 for name in required):
        raise UnsupportedModelTiming("engine admission requires unique independent native units")
    out.mkdir(parents=True)
    validate_derived(cfg, blocks, releases, out)
    facts_hash = write(out, "model-timing.json", payload)
    write(out, "facts.lock.json", dict(op_timing_sha256=facts_hash))
    print(json.dumps(dict(operations=len(blocks), passed=True)), flush=True)
    return payload


@model_timing_cli
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    regenerate(args.npu_model, args.out)


if __name__ == "__main__":
    main()
