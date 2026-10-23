#!/usr/bin/env python3
"""Derive isolated model timing, then validate it with the numerical observer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

from atlas_model_schema import LOCK, VOCABULARY
from atlas_model_timing import (configuration, encoded, isa_cases, numerical_data, observer_manifest,
                                sha256, source_digest, validate_derived)


def safe_path(value):
    path = Path(value)
    if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
        raise ValueError("prohibited path")
    if ".." in path.parts:
        raise ValueError("parent traversal is unsupported")
    path = Path(os.path.abspath(path))
    if any(word in part.lower() for part in path.parts for word in ("hammer", "vlsi")):
        raise ValueError("prohibited path")
    if any(parent.is_symlink() for parent in reversed((path, *path.parents))):
        raise ValueError("model and output paths must not contain symlinks")
    return path


def identity(root):
    source, sources = source_digest(root)
    numerical, data = numerical_data(root)
    # A source-only copy has no Git revision; name it by its actual source digest.
    revision, kind = source[:40], "source_sha256_prefix"
    if source == LOCK["source_sha256"]:
        revision, kind = LOCK["model_revision"], "locked_source"
    return dict(revision=revision, revision_kind=kind, source_sha256=source,
                numerical_data_sha256=numerical), sources, data


def write(directory, name, value):
    path = directory / name
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return sha256(path.read_bytes())


def tool_manifest():
    names = ["atlas_model_facts.py", "atlas_derive_vpu.py", "atlas_derive_mxu.py",
             "atlas_derive_memory.py", "atlas_derive_curve.py", "atlas_validate_curve.py",
             "atlas_model_schema.py", "model-timing.lock.json",
             "model-timing-vocabulary.json"]
    return observer_manifest() + [{"path": name, "sha256": sha256(Path(__file__).with_name(name).read_bytes())}
                                  for name in names]


def regenerate(model, out):
    model, out = safe_path(model), safe_path(out)
    if out.exists():
        raise ValueError("--out must be a new directory")
    before = identity(model)
    generator_sources = tool_manifest()
    sys.path.insert(0, str(model))
    import npu_model
    import torch
    from npu_model.hardware.core import Core
    from npu_model.logging.logger import Logger
    from unittest.mock import Mock
    from atlas_derive_vpu import derive_vpu
    from atlas_derive_mxu import derive_mxu
    from atlas_derive_memory import derive_memory
    from atlas_derive_curve import derive_curve
    from atlas_validate_curve import validate_curve

    if safe_path(Path(npu_model.__file__).parent.parent) != model:
        raise ValueError("imported model does not match --npu-model")
    torch.set_num_threads(1)
    cfg, config = configuration()
    # Preserve the exporter's fatal runtime-error propagation check for every model.
    core = Core(config=cfg, logger=Mock(spec=Logger))
    try:
        if core.ignore_runtime_errors is not False:
            raise ValueError("selected model Core does not propagate runtime errors")
    finally:
        core.arch_state.close()
    inputs = [dict(name=name, engine=engine, assembly=assembly) for name, engine, assembly in isa_cases(cfg)]
    start = time.monotonic()
    memory, memory_releases = derive_memory(cfg)
    vpu, vpu_releases = derive_vpu(cfg)
    mxu, mxu_releases = derive_mxu(cfg)
    dma_timing = derive_curve(cfg)
    blocks = vpu | mxu | memory
    releases = vpu_releases | mxu_releases | memory_releases
    if set(blocks) != {case["name"] for case in inputs} or set(releases) != set(blocks):
        raise ValueError("source derivation does not cover the selected ISA probes")
    derive_seconds = time.monotonic() - start
    out.mkdir(parents=True)
    ordered = [blocks[case["name"]] for case in inputs]
    write(out, "derived-blocks.json", ordered)
    write(out, "reservation-releases.json", releases)
    print(f"Derived {len(blocks)} blocks in {derive_seconds:.3f}s; validating numerical observations", flush=True)
    start = time.monotonic()
    report, _ = validate_derived(cfg, blocks, releases, out)
    dma_report = validate_curve(cfg, dma_timing, out / "dma-validation")
    dma_report_hash = sha256((out / "dma-validation" / "validation-report.json").read_bytes())
    dma_timing["validation"] = dict(passed=True, report_sha256=dma_report_hash,
                                    backend_streams=dma_report["backend_streams"],
                                    isolated_dma_runs=dma_report["isolated_dma_runs"],
                                    mixed_dma_runs=dma_report["mixed_dma_runs"])
    report["dma_validation"] = dma_timing["validation"]
    validate_seconds = time.monotonic() - start
    if before != identity(model):
        raise ValueError("model source or numerical data changed during regeneration")
    if generator_sources != tool_manifest():
        raise ValueError("derivation or observer tools changed during regeneration")
    report.update(derive_seconds=derive_seconds, validation_seconds=validate_seconds,
                  derived_families=["vpu", "mxu0", "mxu1", "vlsu", "xlu", "scalar_lsu", "dma"], measured_families=[])
    report_hash = write(out, "validation-report.json", report)
    source_identity, sources, data = before
    observer_sources = observer_manifest()
    model_values = dict(source_identity, configuration=config, configuration_json=encoded(config).decode(),
                       configuration_sha256=sha256(encoded(config)), observer_sha256=sha256(encoded(observer_sources)),
                       exporter_sha256=sha256(encoded(generator_sources)), input_sha256=sha256(encoded(inputs)))
    payload = dict(schema="atlas.model_timing.v1", model=model_values,
                   geometry=dict(rows=cfg.arch_state_config.mrf_depth, columns=cfg.arch_state_config.mrf_width,
                                 mreg_count=cfg.arch_state_config.num_m_registers),
                   age_convention="issue_edge_zero", observation=dict(accepted=True, drained=True),
                   op_timing=ordered, unresolved=[], reservation_releases=releases,
                   dma_timing=dma_timing,
                   vocabulary_sha256=sha256(encoded(VOCABULARY)),
                   validation=dict(passed=True, operation_count=len(blocks), observed_runs=report["observed_runs"],
                                   unobserved_runs=report["unobserved_runs"], gate_report_sha256=report_hash),
                   limits=["Operation timings cover isolated disjoint operands; no pair, capacity or RTL evidence.",
                           "DMA facts are proposal estimates validated against isolated native model probes only.",
                           "Unit releases are serialized lifetime envelopes, not overlap/throughput admissions.",
                           "VMEM releases describe physical port windows, not whole-operation reservations.",
                           "Logical MREG release requests take effect at the next begin_cycle.",
                           "ACC/weight row events retain conservative provider lifetime policies."],
                   lineage=dict(source_manifest_sha256=sha256(encoded(sources)), inputs=inputs,
                                numerical_data_manifest_sha256=sha256(encoded(data)), observer_sources=observer_sources,
                                generator_sources=generator_sources, contract_lock=LOCK))
    facts_hash = write(out, "model-timing.json", payload)
    write(out, "source-manifest.json", sources)
    write(out, "numerical-data-manifest.json", data)
    write(out, "facts.lock.json", dict(schema="atlas.model_facts.lock.v1", model=source_identity,
                                     configuration_sha256=model_values["configuration_sha256"],
                                     op_timing_sha256=facts_hash, validation_sha256=report_hash,
                                     vocabulary_sha256=payload["vocabulary_sha256"]))
    print(json.dumps(dict(passed=True, operations=len(blocks), facts_sha256=facts_hash,
                          derive_seconds=derive_seconds, validation_seconds=validate_seconds)), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        regenerate(args.npu_model, args.out)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"model facts regeneration failed: {error}\n")


if __name__ == "__main__":
    main()
