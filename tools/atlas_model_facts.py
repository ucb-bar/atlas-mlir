#!/usr/bin/env python3
"""Extract compiler timing facts from npu_model source and configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess

sys.dont_write_bytecode = True

from atlas_model_diagnostics import UnsupportedModelTiming, model_timing_cli
from atlas_model_capabilities import check_configuration
from atlas_model_semantics import check_model_semantics, source_pins
from atlas_model_schema import GUARDS, PIN_PATH
from atlas_model_timing import configuration, encoded, sha256, validate_derived


def write(directory, name, value):
    path = directory / name
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def accept_source(model):
    pins = source_pins(model)
    pins["model_revision"] = subprocess.check_output(
        ["git", "-C", str(model), "rev-parse", "HEAD"], text=True).strip()
    for group in ("files", "source_shapes"):
        for name, digest in pins[group].items():
            if digest != GUARDS[group].get(name):
                print("changed pin: " + name)
    if pins["model_revision"] != GUARDS["model_revision"]:
        print("changed pin: model_revision")
    write(PIN_PATH.parent, PIN_PATH.name, pins)


def regenerate(model, out):
    model, out = Path(model).absolute(), Path(out).absolute()
    if out.exists():
        raise ValueError("--out must be a new directory")
    check_model_semantics(model)
    sys.path.insert(0, str(model))
    import npu_model
    if Path(npu_model.__file__).resolve() != (model / "npu_model/__init__.py").resolve():
        raise UnsupportedModelTiming("imported npu_model differs from --npu-model")
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
    names = list(blocks)
    if set(releases) != set(blocks):
        raise UnsupportedModelTiming("source derivation does not cover the selected ISA")
    revision = subprocess.check_output(["git", "-C", str(model), "rev-parse", "HEAD"], text=True).strip()
    payload = dict(model=dict(revision=revision, configuration_sha256=sha256(encoded(config))),
                   architecture=architecture,
                   op_timing=[blocks[name] for name in names], reservation_releases=releases,
                   dma_timing=derive_curve(cfg), mxu_weight_timing=derive_weight_timing(cfg),
                   frontend_timing=derive_frontend(cfg), resources=derive_resources(model))
    # The central Core contract pins routing; each independent engine occurs once.
    required = {"VectorExecutionUnit", "CrossLaneExecutionUnit", "LoadStoreUnit"}
    if any(list(cfg.execution_units.values()).count(name) != 1 for name in required):
        raise UnsupportedModelTiming("engine admission requires unique independent native units")
    out.mkdir(parents=True)
    native_blocks = validate_derived(cfg, blocks)
    write(out, "model-timing.json", payload)
    print(json.dumps(dict(operations=len(blocks), native_blocks=native_blocks, passed=True)), flush=True)
    return payload


@model_timing_cli
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-model", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path)
    mode.add_argument("--accept-source", action="store_true",
                      help="accept reviewed source changes and rewrite model-pins.json")
    args = parser.parse_args()
    if args.accept_source:
        accept_source(args.npu_model)
    else:
        regenerate(args.npu_model, args.out)


if __name__ == "__main__":
    main()
