"""Exact K=1 host admission proves source structure and preserves +0 addition."""

import ctypes
import shutil
import subprocess
from dataclasses import replace
from math import prod

import numpy as np
import pytest
import test_application_float_arithmetic as fixtures

from atlas_mlir_native.already_quantized import POLICY as ALREADY_QUANTIZED_POLICY
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.host_k1_contraction import emit_c, prepare_source
from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.target_binding import NativeCompilationError


def request(batch=(), m=2, k=1, n=3):
    selected = fixtures.policy()
    rank = len(batch) + 3
    axes = tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank))
    maps = tuple(IndexMap(rank, (*axes[:-3], *rows), (0,) * (rank - 1)) for rows in (
        (axes[-3], axes[-1]), (axes[-1], axes[-2]), (axes[-3], axes[-2])))
    types = [TensorType((*batch, *shape), "f32", "raw-bit-pattern")
             for shape in ((m, k), (k, n), (m, n))]
    return KernelRequest((
        SemanticNode("a", "input", (), types[0], effect="input"),
        SemanticNode("b", "input", (), types[1], effect="input"),
        SemanticNode("seed", "constant", (), types[2], effect="constant"),
        SemanticNode("y", "source_f32_matmul_unqualified", ("a", "b", "seed"), types[2], (
            ("numeric_semantics", "source-f32-mul-add-unqualified-v1"),
            ("reduction_order", "unresolved"), ("scalar_add_order", "accumulator-product")),
            index_maps=(*maps, maps[-1])),
    ), ("y",), ("external",), (("a", "external"), ("b", "external")),
        selected["target_identity"], "diagnostic",
        constants=(ConstantBinding("seed", "f32-le-raw", "00000000" * prod(types[2].shape)),))


def policy():
    return {**fixtures.policy(), "host_float_arithmetic": ["f32_mul"],
            "already_quantized": ALREADY_QUANTIZED_POLICY}


@pytest.mark.parametrize("batch", [(), (2,), (2, 3)])
def test_explicit_exact_multiply_admits_only_proved_k1_as_host(batch):
    source, selected = request(batch), policy()
    adapter = AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"])
    (alternative,) = adapter.plan(source).selected
    assert alternative.placement == "host"
    result = adapter.compile_region(None, source, alternative, "outer_product", None, None, None)
    assert result.programs == ()
    assert result.inputs == ("a", "b") and result.outputs == ("y",)
    receipt = result.description["normalization"]
    assert receipt["source_request_digest"] == source.digest()
    assert receipt["scalar_request_digest"] == prepare_source(source).body.digest()
    assert receipt["precision_change"] == "none" and receipt["k"] == 1
    assert receipt["source_index_maps"] == [value.record() for value in source.nodes[-1].index_maps]
    assert source.nodes[-1].op == "source_f32_matmul_unqualified"


def test_unselected_host_arithmetic_and_k_greater_than_one_stay_rejected():
    selected = policy()
    adapter = AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"])
    with pytest.raises(NativeCompilationError, match="already-quantized intake refuses floating contractions"):
        adapter.plan(request(k=2))
    selected["host_float_arithmetic"] = ["f64_mul"]
    with pytest.raises(NativeCompilationError, match="already-quantized intake refuses floating contractions"):
        AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"]).plan(request())


@pytest.mark.parametrize("mutation", ["offset", "output-map", "batch-map", "type", "attrs", "seed", "negative-zero"])
def test_structural_changes_do_not_get_relabelled(mutation):
    source = request((2,))
    nodes = list(source.nodes)
    result = nodes[-1]
    if mutation in {"offset", "output-map", "batch-map"}:
        maps = list(result.index_maps)
        if mutation == "offset":
            maps[0] = replace(maps[0], offsets=(0, 1, 0))
        elif mutation == "output-map":
            maps[-1] = maps[0]
        else:
            rows = list(maps[1].coefficients)
            rows[0] = (0, 1, 0, 0)
            maps[1] = replace(maps[1], coefficients=tuple(rows))
        nodes[-1] = replace(result, index_maps=tuple(maps))
    elif mutation == "type":
        nodes[0] = replace(nodes[0], type=replace(nodes[0].type, numerical_policy="finite"))
    elif mutation == "attrs":
        nodes[-1] = replace(result, attrs=(*result.attrs, ("allow_fma", True)))
    elif mutation == "seed":
        nodes[2] = replace(nodes[2], attrs=(("unknown", True),))
    else:
        source = replace(source, constants=(replace(source.constants[0], data_hex="00000080" * 12),))
    source = replace(source, nodes=tuple(nodes))
    with pytest.raises(ValueError):
        prepare_source(source)
    selected = policy()
    with pytest.raises(NativeCompilationError, match="already-quantized intake refuses floating contractions"):
        AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"]).plan(source)


@pytest.mark.skipif(shutil.which("cc") is None, reason="host C compiler unavailable")
def test_generated_c_preserves_broadcast_bits_positive_zero_and_atomic_rejection(tmp_path):
    source = request((2,), m=8, n=5)
    record, files = emit_c(source, "outer")
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    library_path = tmp_path / "outer.so"
    compiled = subprocess.run([
        "nice", "-n", "10", shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        "-shared", "-fPIC", str(tmp_path / "outer.c"), "-o", str(library_path),
    ], capture_output=True)
    assert compiled.returncode == 0, compiled.stderr.decode()
    function = ctypes.CDLL(str(library_path)).outer
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    function.restype = ctypes.c_int
    left = np.array([0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff, 0x3f800001, 0xbf800001,
                     0x3f800000, 0xbf800000, 0x00800000, 0x80800000, 3, 0x80000003, 0x40000000, 0xc0000000], dtype="<u4")
    right = np.array([0x3f000000, 0xbf000000, 0x3f800001, 0x40000000, 0,
                      0x3f800000, 0xbf800000, 0x3f000000, 0xbf000000, 0x80000000], dtype="<u4")
    with np.errstate(all="ignore"):
        expected = (left.view("<f4").reshape(2, 8, 1) * right.view("<f4").reshape(2, 1, 5) + np.float32(0)).view("<u4")
    inputs = (ctypes.c_void_p * 2)(left.ctypes.data, right.ctypes.data)
    output = np.full((2, 8, 5), 0xcdcdcdcd, dtype="<u4")
    outputs = (ctypes.c_void_p * 1)(output.ctypes.data)
    assert function(None, inputs, outputs) == 0
    np.testing.assert_array_equal(output, expected)
    assert not np.any(output == 0x80000000)
    # A NaN in the last broadcast column must leave all output bytes untouched.
    right[-1] = 0x7fc00000
    output.fill(0xcdcdcdcd)
    assert function(None, inputs, outputs) == -2
    assert np.all(output == 0xcdcdcdcd)
    right[-1] = 0x7f800000
    left[-1] = 0
    assert function(None, inputs, outputs) == -2
    assert np.all(output == 0xcdcdcdcd)
    assert function(None, inputs, (ctypes.c_void_p * 1)(left.ctypes.data)) == -1
    assert record["scalar_arithmetic"]["numeric_semantics"] == "ieee754-f32-mul-rne-v1"
