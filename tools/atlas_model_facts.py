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
from atlas_search_policy import POLICY
from atlas_model_capabilities import check_configuration
from atlas_model_semantics import check_model_semantics
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
             "atlas_derive_overlap.py", "atlas_overlap_rules.py", "atlas_vpu_partition.py", "atlas_validate_overlap.py",
             "atlas_model_schema.py", "atlas_model_diagnostics.py", "atlas_derive_arch.py",
             "atlas_derive_vocabulary.py", "atlas_model_capabilities.py", "model-timing.lock.json",
             "model-timing-vocabulary.json", "atlas_derived_contract.py",
             "atlas_derive_scratchpad.py", "atlas_derive_frontend.py", "atlas_validate_mxu_weights.py",
             "atlas_model_semantics.py", "model-semantics.json", "atlas_validate_runtime.py",
             "atlas_derive_replay.py",
             "atlas_model_replay.py", "atlas_validate_replay.py", "atlas_derive_costs.py",
             "atlas_search_policy.py", "model-search-policy.json",
             "atlas_derive_admission.py", "atlas_validate_admission.py",
             "atlas_derive_dma_admission.py", "atlas_validate_dma_admission.py"]
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
    semantics = check_model_semantics(model)
    sys.path.insert(0, str(model))
    import npu_model
    import torch
    from atlas_derive_vpu import derive_vpu
    from atlas_derive_mxu import derive_mxu, derive_weight_timing
    from atlas_validate_mxu_weights import validate_weight_timing
    from atlas_derive_memory import derive_memory
    from atlas_derive_curve import derive_curve
    from atlas_derive_scratchpad import derive_scratchpad
    from atlas_derive_frontend import derive_frontend
    from atlas_derive_replay import derive_replay
    from atlas_validate_replay import validate_replay
    from atlas_derive_costs import derive_costs
    from atlas_validate_curve import validate_curve
    from atlas_validate_runtime import validate_runtime
    from atlas_derive_arch import derive_arch
    from atlas_derive_vocabulary import validate_vocabulary, derive_isa_metadata
    from atlas_derive_admission import derive_admission
    from atlas_validate_admission import validate_admission
    from atlas_derive_dma_admission import derive_dma_admission
    from atlas_validate_dma_admission import validate_dma_admission

    if safe_path(Path(npu_model.__file__).parent.parent) != model:
        raise UnsupportedModelTiming("imported model does not match --npu-model")
    torch.set_num_threads(POLICY["engineering"]["scorer"]["torch_threads"])
    cfg, config = configuration()
    check_configuration(cfg)
    note_unqualified_model(before[0]["source_sha256"], sha256(encoded(config)))
    architecture = derive_arch(cfg)
    vocabulary = validate_vocabulary(cfg)
    isa_metadata = derive_isa_metadata(cfg)
    inputs = [dict(name=name, engine=engine, assembly=assembly) for name, engine, assembly in isa_cases(cfg)]
    start = time.monotonic()
    memory, memory_releases = derive_memory(cfg)
    vpu, vpu_releases = derive_vpu(cfg)
    mxu, mxu_releases = derive_mxu(cfg)
    mxu_weight_timing = derive_weight_timing(cfg)
    dma_timing = derive_curve(cfg)
    scratchpad_contract = derive_scratchpad(cfg)
    frontend_timing = derive_frontend(cfg)
    replay_timing = derive_replay(cfg)
    proposal_costs = derive_costs(cfg, architecture, isa_metadata)
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
    runtime_report = validate_runtime(model)
    runtime_report_hash = write(out, "runtime-validation.json", runtime_report)
    report, _ = validate_derived(cfg, blocks, releases, out)
    report["mxu_weight_validation"] = validate_weight_timing(cfg, mxu_weight_timing)
    report["runtime_validation"] = dict(passed=True, report_sha256=runtime_report_hash)
    dma_report = validate_curve(cfg, dma_timing, out / "dma-validation")
    replay_report = validate_replay(cfg, replay_timing)
    report["replay_validation"] = dict(passed=True, report_sha256=write(out, "replay-validation.json", replay_report),
                                       program_sha256=replay_report["program_sha256"], runs=replay_report["runs"])
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
    report["reviewed_validation_suite"] = POLICY["validation"]["reviewed_suite"]
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
                   replay_timing=replay_timing, proposal_costs=proposal_costs,
                   vocabulary_sha256=sha256(encoded(vocabulary)),
                   validation=dict(passed=True, operation_count=len(blocks), observed_runs=report["observed_runs"],
                                   unobserved_runs=report["unobserved_runs"], gate_report_sha256=report_hash),
                   limits=["Operation timing blocks describe isolated disjoint operands; admission has separate source contracts.",
                           "DMA costs are conditional drained/no-denial service bounds; dynamic completion still requires matching waits.",
                           "Unit releases are isolated lifetime envelopes; engine_admission and overlap rules govern initiation.",
                           "VMEM releases describe physical port windows, not whole-operation reservations.",
                           "Logical MREG release requests take effect at the next begin_cycle.",
                           "ACC row events retain conservative provider lifetime policies.",
                           "Weight windows describe isolated native model uses; they do not admit operation overlap."],
                   lineage=dict(source_manifest_sha256=sha256(encoded(sources)), inputs=inputs,
                                semantic_dependencies=semantics,
                                numerical_data_manifest_sha256=sha256(encoded(data)), observer_sources=observer_sources,
                                generator_sources=generator_sources, contract_lock=LOCK))
    payload["validation"]["mxu_weight_validation"] = report["mxu_weight_validation"]
    # Derive both contracts before any new native validation. Observations may
    # reject a contract but never fit its initiation or resource predicates.
    payload["engine_admission"] = derive_admission(model, payload)
    payload["dma_admission"] = derive_dma_admission(model, payload)
    admission_start = time.monotonic()
    for field, validator in (("engine_admission", validate_admission),
                             ("dma_admission", validate_dma_admission)):
        print("Validating frozen source-derived " + field, flush=True)
        qualified, admission_report = validator(model, payload, payload[field])
        payload[field] = qualified
        report_digest = write(out, field.replace("_", "-") + "-validation.json", admission_report)
        payload[field]["validation"]["report_sha256"] = report_digest
    admission_seconds = time.monotonic() - admission_start
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
    # The operation report precedes admission/overlap qualification. Record the
    # complete reviewed suite only after every gate has actually succeeded.
    suite_hash = write(out, "suite-validation.json", dict(
        schema="atlas.model_validation_suite.v1", passed=True,
        semantic_payload_sha256=semantic_hash,
        policy=POLICY["validation"]["reviewed_suite"],
        reports=dict(operations_and_weights=report_hash, runtime=runtime_report_hash,
                     replay=report["replay_validation"]["report_sha256"],
                     dma=dma_report_hash,
                     engine_admission=payload["engine_admission"]["validation"]["report_sha256"],
                     dma_admission=payload["dma_admission"]["validation"]["report_sha256"],
                     overlap=overlap["overlap_validation_sha256"])))
    write(out, "facts.lock.json", dict(schema="atlas.model_facts.lock.v1", model=source_identity,
                                     configuration_sha256=model_values["configuration_sha256"],
                                     op_timing_sha256=facts_hash, validation_sha256=report_hash,
                                     semantic_payload_sha256=semantic_hash,
                                     suite_validation_sha256=suite_hash,
                                     vocabulary_sha256=payload["vocabulary_sha256"],
                                     overlap_sha256=overlap["overlap_sha256"],
                                     overlap_validation_sha256=overlap["overlap_validation_sha256"]))
    write(out, "timing-receipt.json", dict(schema="atlas.model_derivation_receipt.v1",
                                         facts_sha256=facts_hash, semantic_payload_sha256=semantic_hash,
                                         validation_sha256=report_hash, derive_seconds=derive_seconds,
                                         suite_validation_sha256=suite_hash,
                                         validation_seconds=validate_seconds,
                                         admission_validation_seconds=admission_seconds, **overlap))
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
