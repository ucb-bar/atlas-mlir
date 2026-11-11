"""Graph correspondence and actual instruction timing exports across both modes."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("merlin.frontends.linalg_float_graph")

from atlas_mlir_native.shared_graph import compile_graph, pack_graph_inputs, plan_graph, virtual_graph
from atlas_mlir_native.shared_timing import verify_issue_export
from merlin.frontends.linalg_text import parse_mlir_text

ROOT = Path(__file__).resolve().parents[1]
BINS = Path(os.environ.get("ATLAS_OOT_BIN_DIR", str(ROOT / "build/bin")))


def source(name):
    text = (ROOT / "test/examples" / (name + ".mlir")).read_text()
    return text, parse_mlir_text(text).body.block.first_op.sym_name.data


@pytest.mark.parametrize("name", ["native_bf16_relu_linalg", "native_mxu0_relu_linalg",
                                  "native_mxu0_relu_pack_mxu0_linalg"])
def test_graph_reads_actual_producers_and_all_outputs(name):
    text, entry = source(name)
    request, schedules = plan_graph(text, entry=entry)
    virtual, abi = virtual_graph(request, schedules)
    assert len(abi["outputs"]) == len(request.outputs)
    assert virtual.count('"atlas.virtual_output_bf16"') == len(request.outputs)
    assert len(schedules) == sum(n.op == "ordered_dot" for n in request.nodes)
    if "pack" in name:
        assert tuple(n.op for n in request.nodes) == (
            "input", "input", "input", "ordered_dot", "relu", "cast", "ordered_dot")
        assert request.nodes[-1].inputs == (request.nodes[-2].id, "input2")


@pytest.mark.parametrize("pipeline", ["source-order", "optimized"])
def test_actual_words_have_consumed_annotations_and_mutations_refuse(tmp_path, pipeline):
    if not (BINS / "atlas-opt").exists():
        pytest.skip("build atlas-opt/atlas-emit to exercise actual timing export")
    text, entry = source("native_mxu0_relu_pack_mxu0_linalg")
    directory = tmp_path / pipeline
    manifest = compile_graph(text, entry=entry, destination=directory, atlas_bin_dir=BINS, pipeline=pipeline)
    record = json.loads((directory / "issue-timing.json").read_text())
    program = json.loads((directory / "program.json").read_text())
    assert manifest["shared_issue_timing"]["instructions"] == manifest["words"]
    assert "schedule.overlap" in (directory / "timing-block0.mlir").read_text()
    for mutation in ("bits", "coverage", "spacing", "provider"):
        bad = deepcopy(record)
        if mutation == "bits":
            bad["blocks"][0]["instructions"][0]["word_u32"] ^= 1
        elif mutation == "coverage":
            bad["blocks"][0]["covered_pairs"] -= 1
        elif mutation == "provider":
            bad["provider"] = "unknown"
        else:
            for block in bad["blocks"]:
                for row in block["instructions"]:
                    row["issue"] = 0
        with pytest.raises(ValueError):
            verify_issue_export(bad, program)


def test_runtime_domain_and_extent_guards():
    text, entry = source("native_bf16_relu_linalg")
    request, schedules = plan_graph(text, entry=entry)
    _, abi = virtual_graph(request, schedules)
    assert len(pack_graph_inputs(request, abi, {"input0": bytes(2048)})[0][1]) == 2048
    for payload in (bytes(2047), bytes.fromhex("807f") * 1024, bytes.fromhex("0100") * 1024):
        with pytest.raises(ValueError):
            pack_graph_inputs(request, abi, {"input0": payload})


def test_tail_and_unknown_pack_policy_refuse():
    text, entry = source("native_mxu0_relu_pack_mxu0_linalg")
    for changed in (text.replace("32x32", "31x31"), text.replace(
            "fp8-e4m3-unit-e8m0-seven-codes-v1", "arbitrary-bf16-truncation")):
        with pytest.raises(ValueError):
            plan_graph(changed, entry=entry)
