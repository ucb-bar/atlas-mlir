"""Shared source intake with explicit target precision and legality boundaries."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("merlin.frontends.linalg_tensor_graph")
from atlas_mlir_native.shared_tensor_graph import (
    PRECISION,
    lower_tensor_graph,
    pack_tensor_inputs,
)
from merlin.frontends.linalg_tensor_graph import read_tensor_graph

from tools.check_shared_attention import fixture

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "test/examples/captured_attention_mlp32_linalg.mlir").read_text()


def read(source=SOURCE):
    return read_tensor_graph(source, entry="forward", target_identity="test")


def test_explicit_precision_required():
    for policy in (None, "f32", "automatic"):
        with pytest.raises(ValueError, match="explicit"):
            lower_tensor_graph(read(), precision=policy)


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("retained", [False, True])
def test_actual_edges_schedules_and_every_output(width, retained):
    request = read(fixture(SOURCE, width, retained))
    virtual, abi, schedules = lower_tensor_graph(request, precision=PRECISION)
    assert len(schedules) == 7
    assert len(abi["outputs"]) == (12 if retained else 1)
    assert virtual.count('"atlas.virtual_output_bf16"') == len(abi["outputs"])
    assert 'kind = "row_max"' in virtual and 'kind = "row_sum"' in virtual
    assert 'kind = "recip"' in virtual and 'kind = "exp"' in virtual
    assert any(row["node"] == request.nodes[24].id for row in abi["conversions"])
    for schedule in schedules.values():
        schedule.revalidate()


@pytest.mark.parametrize(
    "source,message",
    [
        (
            SOURCE.replace("32x", "31x").replace(
                "output_shape [32, 1]", "output_shape [31, 1]"
            ),
            "full rank-two",
        ),
        (
            SOURCE.replace("32x", "64x").replace(
                "output_shape [32, 1]", "output_shape [64, 1]"
            ),
            "32-element row",
        ),
        (
            SOURCE.replace("f32", "f16").replace("0xff800000", "0xfc00"),
            "source precision",
        ),
    ],
)
def test_unsupported_geometry_or_precision_refuses(source, message):
    with pytest.raises(ValueError, match=message):
        lower_tensor_graph(read(source), precision=PRECISION)


def test_provenance_is_irrelevant():
    original = read()
    changed = read(SOURCE.replace("softmax_0", "wrong").replace("query", "unknown"))
    va, aa, _ = lower_tensor_graph(original, precision=PRECISION)
    vb, ab, _ = lower_tensor_graph(changed, precision=PRECISION)
    assert va == vb and aa == ab
    assert original.source_identity != changed.source_identity


def test_boundary_packing_rounding_flush_saturation_and_rejections():
    request = read()
    arguments = {
        n.id: np.zeros(n.type.shape, np.float32)
        for n in request.nodes
        if n.op == "input"
    }
    arguments["input0"][0, :8] = [0, -0.0, 0.001, -0.001, 1.0625, 1.1875, 1000, -1000]
    abi = {"inputs": [{"node": "input0", "row": 0, "col": 0, "dtype": "fp8"}]}
    payload = pack_tensor_inputs(request, abi, arguments)[0][1]
    assert list(payload[:8]) == [0, 0, 0, 0, 0x38, 0x3A, 0x7E, 0xFE]
    arguments["input0"] = arguments["input0"].astype(np.float64)
    with pytest.raises(ValueError, match="dtype"):
        pack_tensor_inputs(request, abi, arguments)
    arguments["input0"] = arguments["input0"].astype(np.float32)
    arguments["input0"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pack_tensor_inputs(request, abi, arguments)
    arguments.pop("input0")
    with pytest.raises(ValueError, match="identities"):
        pack_tensor_inputs(request, abi, arguments)


def test_model_decoder_keeps_all_sixteen_vector_immediate_bits():
    model_python = os.environ.get("ATLAS_MODEL_PYTHON")
    model_root = os.environ.get("ATLAS_MODEL_ROOT")
    if not model_python or not model_root:
        pytest.skip("select the model Python and pinned model root")
    script = """
from npu_model.configs.isa_definition import VLI_ALL
from npu_model.isa_types import MatrixReg,Imm16
from test.run_shared_model_case import instructions
for value in (1,0x3f80,0x8000,0xffff):
    word=VLI_ALL(vd=MatrixReg(12),imm=Imm16(value)).to_bytecode()
    assert instructions(word.to_bytes(4,'little'))[0].to_bytecode()==word
"""
    subprocess.run(
        [model_python, "-c", script],
        check=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((model_root, str(ROOT)))},
    )
