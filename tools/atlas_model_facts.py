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
from atlas_derived_contract import seal_derived, overlap_semantics
from atlas_model_timing import (configuration, encoded, isa_cases, numerical_data,
                                sha256, source_digest, validate_derived)


def safe_path(value):
    return Path(value).absolute()


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
    from atlas_derive_scratchpad import derive_scratchpad
    from atlas_derive_frontend import derive_frontend
    from atlas_derive_replay import derive_replay
    from atlas_derive_costs import derive_costs
    from atlas_derive_arch import derive_arch
    from atlas_derive_vocabulary import validate_vocabulary, derive_isa_metadata
    from atlas_derive_admission import derive_admission
    from atlas_derive_dma_admission import derive_dma_admission
    from atlas_derive_overlap import generate

    torch.set_num_threads(1)
    cfg, config = configuration()
    check_configuration(cfg)
    architecture = derive_arch(cfg)
    vocabulary = validate_vocabulary(cfg)
    isa_metadata = derive_isa_metadata(cfg)
    blocks, releases = {}, {}
    for extract in (derive_vpu, derive_mxu, derive_memory):
        timing, reservations = extract(cfg)
        blocks.update(timing)
        releases.update(reservations)
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
                   architecture=architecture, operand_vocabulary=vocabulary,
                   isa_metadata=isa_metadata, age_convention="issue_edge_zero",
                   observation=dict(accepted=True, drained=True),
                   op_timing=[blocks[name] for name in names], reservation_releases=releases,
                   dma_timing=derive_curve(cfg), mxu_weight_timing=derive_weight_timing(cfg),
                   scratchpad_contract=derive_scratchpad(cfg), frontend_timing=derive_frontend(cfg),
                   replay_timing=derive_replay(cfg),
                   proposal_costs=derive_costs(cfg, architecture, isa_metadata),
                   vocabulary_sha256=sha256(encoded(vocabulary)))
    payload["engine_admission"] = derive_admission(model, payload)
    payload["dma_admission"] = derive_dma_admission(model, payload)
    out.mkdir(parents=True)
    validate_derived(cfg, blocks, releases, out)
    write(out, "model-timing.json", payload)
    overlap = generate(model, out / "model-timing.json", out / "model-overlap.json")
    payload["overlap_rules_sha256"] = sha256(encoded(overlap_semantics(overlap)))
    seal_derived(payload)
    facts_hash = write(out, "model-timing.json", payload)
    # Bind overlap to the final timing file after the semantic payload is sealed.
    overlap["per_op_sha256"] = facts_hash
    overlap_hash = write(out, "model-overlap.json", overlap)
    write(out, "facts.lock.json", dict(schema="atlas.model_facts.lock.v1",
                                      op_timing_sha256=facts_hash, overlap_sha256=overlap_hash))
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
