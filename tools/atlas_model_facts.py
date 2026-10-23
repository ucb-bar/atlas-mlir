#!/usr/bin/env python3
"""Derive model timing and overlap rules, then run their native validation gates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

from atlas_model_diagnostics import UnsupportedModelTiming, ModelTimingValidationError, model_timing_cli
from atlas_model_schema import LOCK
from atlas_model_capabilities import check_configuration
from atlas_derived_contract import seal_derived, verify_derived, overlap_semantics
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


def note_unqualified_model(source_sha256, configuration_sha256):
    """Derivation can inspect changes; compilation checks capabilities."""
    if (source_sha256 != LOCK["source_sha256"] or
            configuration_sha256 != LOCK["configuration_sha256"]):
        print("Note: results are not from the qualified revision " + LOCK["model_revision"] +
              " (source or configuration content differs from tools/model-timing.lock.json). "
              "Compilation requires supported capabilities and reviewed source contracts.",
              file=sys.stderr)


def write(directory, name, value):
    path = directory / name
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return sha256(path.read_bytes())


def tool_manifest():
    names = ["atlas_model_facts.py", "atlas_derive_vpu.py", "atlas_derive_mxu.py",
             "atlas_derive_memory.py", "atlas_derive_curve.py", "atlas_validate_curve.py",
             "atlas_derive_overlap.py", "atlas_overlap_rules.py", "atlas_validate_overlap.py",
             "atlas_model_schema.py", "atlas_model_diagnostics.py", "atlas_derive_arch.py",
             "atlas_derive_vocabulary.py", "atlas_model_capabilities.py", "model-timing.lock.json",
             "model-timing-vocabulary.json", "atlas_derived_contract.py",
             "atlas_derive_scratchpad.py", "atlas_derive_frontend.py", "atlas_validate_mxu_weights.py"]
    return observer_manifest() + [{"path": name, "sha256": sha256(Path(__file__).with_name(name).read_bytes())}
                                  for name in names]


def overlap_bundle(model, out):
    """Freeze source proposals before native observations; publish only on success."""
    from atlas_derive_overlap import generate
    from atlas_validate_overlap import validate
    candidate, samples = out / "overlap-candidate.json", out / "overlap-samples.json"
    start = time.monotonic()
    try:
        rules = generate(model, out / "model-timing.json", candidate, samples)
        facts = json.loads((out / "model-timing.json").read_text())
        if "overlap_rules_sha256" in facts and sha256(encoded(overlap_semantics(rules))) != facts["overlap_rules_sha256"]:
            raise ValueError("overlap proposal differs from the sealed derived semantics")
    except ValueError as error:
        raise UnsupportedModelTiming(str(error)) from error
    derive_seconds = time.monotonic() - start
    print(f"Derived {len(rules['pairs'])} overlap rows without model runs; validating native boundaries", flush=True)
    try:
        # Sequential execution also works where native process semaphores are unavailable.
        report = validate(model, out / "model-timing.json", candidate, samples,
                          out / "overlap-validation", out / "model-overlap.json", workers=1)
    except (ValueError, RuntimeError) as error:
        raise ModelTimingValidationError(str(error)) from error
    return dict(overlap_sha256=sha256((out / "model-overlap.json").read_bytes()),
                overlap_validation_sha256=sha256((out / "overlap-validation" / "validation-report.json").read_bytes()),
                overlap_derive_seconds=derive_seconds, overlap_validation_seconds=report["seconds"])


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
    from atlas_derive_mxu import derive_mxu, derive_weight_timing
    from atlas_validate_mxu_weights import validate_weight_timing
    from atlas_derive_memory import derive_memory
    from atlas_derive_curve import derive_curve
    from atlas_derive_scratchpad import derive_scratchpad
    from atlas_derive_frontend import derive_frontend
    from atlas_validate_curve import validate_curve
    from atlas_derive_arch import derive_arch
    from atlas_derive_vocabulary import validate_vocabulary, derive_isa_metadata

    if safe_path(Path(npu_model.__file__).parent.parent) != model:
        raise UnsupportedModelTiming("imported model does not match --npu-model")
    torch.set_num_threads(1)
    cfg, config = configuration()
    check_configuration(cfg)
    note_unqualified_model(before[0]["source_sha256"], sha256(encoded(config)))
    architecture = derive_arch(cfg)
    vocabulary = validate_vocabulary(cfg)
    isa_metadata = derive_isa_metadata(cfg)
    # Preserve the exporter's fatal runtime-error propagation check for every model.
    core = Core(config=cfg, logger=Mock(spec=Logger))
    try:
        if core.ignore_runtime_errors is not False:
            raise UnsupportedModelTiming("selected model Core does not propagate runtime errors")
    finally:
        core.arch_state.close()
    inputs = [dict(name=name, engine=engine, assembly=assembly) for name, engine, assembly in isa_cases(cfg)]
    start = time.monotonic()
    memory, memory_releases = derive_memory(cfg)
    vpu, vpu_releases = derive_vpu(cfg)
    mxu, mxu_releases = derive_mxu(cfg)
    mxu_weight_timing = derive_weight_timing(cfg)
    dma_timing = derive_curve(cfg)
    scratchpad_contract = derive_scratchpad(cfg)
    frontend_timing = derive_frontend(cfg)
    blocks = vpu | mxu | memory
    releases = vpu_releases | mxu_releases | memory_releases
    if set(blocks) != {case["name"] for case in inputs} or set(releases) != set(blocks):
        raise UnsupportedModelTiming("source derivation does not cover the selected ISA probes")
    derive_seconds = time.monotonic() - start
    out.mkdir(parents=True)
    ordered = [blocks[case["name"]] for case in inputs]
    write(out, "derived-blocks.json", ordered)
    write(out, "reservation-releases.json", releases)
    print(f"Derived {len(blocks)} blocks in {derive_seconds:.3f}s; validating numerical observations", flush=True)
    start = time.monotonic()
    report, _ = validate_derived(cfg, blocks, releases, out)
    report["mxu_weight_validation"] = validate_weight_timing(cfg, mxu_weight_timing)
    dma_report = validate_curve(cfg, dma_timing, out / "dma-validation")
    dma_report_hash = sha256((out / "dma-validation" / "validation-report.json").read_bytes())
    dma_timing["validation"] = dict(passed=True, report_sha256=dma_report_hash,
                                    backend_streams=dma_report["backend_streams"],
                                    isolated_dma_runs=dma_report["isolated_dma_runs"],
                                    mixed_dma_runs=dma_report["mixed_dma_runs"])
    report["dma_validation"] = dma_timing["validation"]
    validate_seconds = time.monotonic() - start
    if before != identity(model):
        raise ModelTimingValidationError("model source or numerical data changed during regeneration")
    if generator_sources != tool_manifest():
        raise ModelTimingValidationError("derivation or observer tools changed during regeneration")
    report.update(derived_families=["vpu", "mxu0", "mxu1", "vlsu", "xlu", "scalar_lsu", "dma"], measured_families=[])
    report_hash = write(out, "validation-report.json", report)
    source_identity, sources, data = before
    observer_sources = observer_manifest()
    model_values = dict(source_identity, configuration=config, configuration_json=encoded(config).decode(),
                       configuration_sha256=sha256(encoded(config)), observer_sha256=sha256(encoded(observer_sources)),
                       exporter_sha256=sha256(encoded(generator_sources)), input_sha256=sha256(encoded(inputs)))
    payload = dict(schema="atlas.model_timing.v1", model=model_values,
                   geometry=dict(rows=cfg.arch_state_config.mrf_depth, columns=cfg.arch_state_config.mrf_width,
                                 mreg_count=cfg.arch_state_config.num_m_registers),
                   architecture=architecture, operand_vocabulary=vocabulary, isa_metadata=isa_metadata,
                   age_convention="issue_edge_zero", observation=dict(accepted=True, drained=True),
                   op_timing=ordered, unresolved=[], reservation_releases=releases,
                   dma_timing=dma_timing,
                   mxu_weight_timing=mxu_weight_timing,
                   scratchpad_contract=scratchpad_contract, frontend_timing=frontend_timing,
                   vocabulary_sha256=sha256(encoded(vocabulary)),
                   validation=dict(passed=True, operation_count=len(blocks), observed_runs=report["observed_runs"],
                                   unobserved_runs=report["unobserved_runs"], gate_report_sha256=report_hash),
                   limits=["Operation timings cover isolated disjoint operands; no pair, capacity or RTL evidence.",
                           "DMA facts are proposal estimates validated against isolated native model probes only.",
                           "Unit releases are serialized lifetime envelopes, not overlap/throughput admissions.",
                           "VMEM releases describe physical port windows, not whole-operation reservations.",
                           "Logical MREG release requests take effect at the next begin_cycle.",
                           "ACC row events retain conservative provider lifetime policies.",
                           "Weight windows describe isolated native model uses; they do not admit operation overlap."],
                   lineage=dict(source_manifest_sha256=sha256(encoded(sources)), inputs=inputs,
                                numerical_data_manifest_sha256=sha256(encoded(data)), observer_sources=observer_sources,
                                generator_sources=generator_sources, contract_lock=LOCK))
    payload["validation"]["mxu_weight_validation"] = report["mxu_weight_validation"]
    # Freeze rule semantics before sealing operation facts. The native gate then
    # validates a proposal bound to those final facts bytes; publish the bundle
    # lock only after it succeeds. This avoids a circular file-hash dependency.
    from atlas_derive_overlap import generate
    write(out, "model-timing.json", payload)
    proposal = generate(model, out / "model-timing.json", out / "overlap-candidate.json",
                        out / "overlap-samples.json")
    payload["overlap_rules_sha256"] = sha256(encoded(overlap_semantics(proposal)))
    seal_derived(payload)
    semantic_hash = verify_derived(payload)
    facts_hash = write(out, "model-timing.json", payload)
    write(out, "source-manifest.json", sources)
    write(out, "numerical-data-manifest.json", data)
    overlap = overlap_bundle(model, out)
    if before != identity(model) or generator_sources != tool_manifest():
        raise ModelTimingValidationError("model or generator tools changed during overlap validation")
    write(out, "facts.lock.json", dict(schema="atlas.model_facts.lock.v1", model=source_identity,
                                     configuration_sha256=model_values["configuration_sha256"],
                                     op_timing_sha256=facts_hash, validation_sha256=report_hash,
                                     semantic_payload_sha256=semantic_hash,
                                     vocabulary_sha256=payload["vocabulary_sha256"],
                                     overlap_sha256=overlap["overlap_sha256"],
                                     overlap_validation_sha256=overlap["overlap_validation_sha256"]))
    write(out, "timing-receipt.json", dict(schema="atlas.model_derivation_receipt.v1",
                                         facts_sha256=facts_hash, semantic_payload_sha256=semantic_hash,
                                         validation_sha256=report_hash, derive_seconds=derive_seconds,
                                         validation_seconds=validate_seconds, **overlap))
    print(json.dumps(dict(passed=True, operations=len(blocks), facts_sha256=facts_hash,
                          derive_seconds=derive_seconds, validation_seconds=validate_seconds,
                          **overlap)), flush=True)
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
