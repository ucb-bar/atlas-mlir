"""Explicit span96 admission and gradual-underflow primitive selection."""
from dataclasses import replace
import ctypes
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from atlas_mlir_native.row_softmax import (
    ROWSPAN80_DOMAIN, ROWSPAN96_DOMAIN, prepare_inputs, prepare_row_softmax,
    validate_row_span,
)
from atlas_mlir_native.softmax_precision import input_spec, lower_semantic_softmax
from test_row_span_softmax import payload
from test_tiled_softmax import source_request


def plan(shape=(1, 2), *, masked=False, domain=ROWSPAN96_DOMAIN):
    request = source_request(shape, masked)
    return prepare_row_softmax(replace(request, nodes=(
        replace(request.nodes[0], attrs=(("input_domain", domain),)), *request.nodes[1:])))


@pytest.mark.parametrize("values", [[-128, -32], [-48, 48], [-96, 0], [-95.5, 0], [-0.0, 0.0]])
def test_span96_accepts_exact_edges(values):
    prepare_inputs(plan(), payload(values))


@pytest.mark.parametrize("values", [[-128, -31.5], [-48.25, 48], [-96, 2.0**-119],
                                    [-32.5, -32.25], [48.25, 48], [-129, -32], [0, 2.0**-120]])
def test_span96_refuses_exact_span_maximum_and_score_codes(values):
    with pytest.raises(ValueError, match="domain"):
        prepare_inputs(plan(), payload(values))


def test_span80_admission_and_primitives_are_unchanged():
    old, new = plan(domain=ROWSPAN80_DOMAIN), plan()
    with pytest.raises(ValueError, match="span80"):
        prepare_inputs(old, payload([-96, 0]))
    old_ops = {node.op for node in old.regions[0].request.nodes}
    new_ops = {node.op for node in new.regions[0].request.nodes}
    assert {"exp_bf16", "mul_bf16_zero", "row_sum_bf16"} <= old_ops
    assert {"exp_bf16_nonpositive", "mul_bf16_finite", "row_sum_bf16_nonnegative"} <= new_ops
    wide = plan((1, 33))
    assert "add_bf16_nonnegative" in {node.op for node in wide.regions[0].request.nodes}
    assert old.pad_element == new.pad_element == b"\x00\xc2"
    assert old.policy == new.policy


def test_span96_masked_rows_prove_kept_scores_and_require_finite_padding():
    selected = plan((1, 3), masked=True)
    prepare_inputs(selected, payload([-128, -32, -32]), payload([1, 1, 0]))
    with pytest.raises(ValueError, match="padding"):
        prepare_inputs(selected, payload([-128, -32, 0]), payload([1, 1, 0]))
    with pytest.raises(ValueError, match="nonempty"):
        prepare_inputs(selected, payload([-32]*3), payload([0]*3))


def test_span96_has_distinct_conversion_and_semantic_identity():
    from atlas_mlir_native.binding import TARGET_IDENTITY
    from merlin.tests.ir.test_softmax_source import source_graph
    graph = replace(source_graph((1, 2)), target_identity=TARGET_IDENTITY, lowering_policy="diagnostic")
    old = lower_semantic_softmax(graph, score_domain=ROWSPAN80_DOMAIN)
    new = lower_semantic_softmax(graph, score_domain=ROWSPAN96_DOMAIN)
    assert old.request.digest() != new.request.digest()
    assert old.policy_sha256 != new.policy_sha256
    assert input_spec(ROWSPAN96_DOMAIN).policy == "f32-to-bf16-rne-row-relative-span96-pad32-v1"
    assert input_spec(ROWSPAN96_DOMAIN).admitted_code_ranges == input_spec(ROWSPAN80_DOMAIN).admitted_code_ranges


def test_span96_column_limits_cover_python_lowering_and_source_c():
    from atlas_mlir_native.binding import TARGET_IDENTITY
    from atlas_mlir_native.softmax_c_source import render_source_interface
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
    from merlin.semantic_compiler.target_binding import NativeCompilationError
    from merlin.tests.ir.test_softmax_source import source_graph
    prepare_inputs(plan((1, 1024)), payload([0]*1024))
    with pytest.raises(NativeCompilationError):
        plan((1, 1025))
    with pytest.raises(ValueError, match="1024"):
        validate_row_span(payload([0]*1025), (1, 1025), domain=ROWSPAN96_DOMAIN)
    graph = replace(source_graph((1, 1025)), target_identity=TARGET_IDENTITY, lowering_policy="diagnostic")
    with pytest.raises(LinalgBridgeError, match="1024"):
        lower_semantic_softmax(graph, score_domain=ROWSPAN96_DOMAIN)
    request = source_request((1, 1025))
    request = replace(request, nodes=(replace(request.nodes[0], attrs=(("input_domain", ROWSPAN96_DOMAIN),)),
                                      *request.nodes[1:]))
    with pytest.raises(ValueError, match="1024"):
        render_source_interface(SimpleNamespace(request=request), "wide", 1, score_domain=ROWSPAN96_DOMAIN)


def test_generated_c_exact_span96_check_matches_python(tmp_path):
    from merlin.compile.c_softmax_mask import render_bf16_row_span_preflight
    compiler = shutil.which("cc")
    assert compiler, "host C compiler required for generated preflight verification"
    source = "#include <stdint.h>\n#include <stddef.h>\n" + render_bf16_row_span_preflight(
        1, 2, "test", maximum_lower_bound=-32, maximum_upper_bound=48, maximum_span=96)
    source += "\nint check(const uint8_t *in) { return test_row_span(in,0); }\n"
    (tmp_path / "span.c").write_text(source)
    library = tmp_path / "span.so"
    subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    str(tmp_path / "span.c"), "-o", str(library)], check=True, capture_output=True)
    check = ctypes.CDLL(str(library)).check
    check.argtypes = (ctypes.c_void_p,)
    check.restype = ctypes.c_int
    for values, expected in (([-128, -32], 0), ([-48, 48], 0), ([-96, 0], 0),
                             ([-96, 2.0**-119], 1), ([-128, -31.5], 1), ([-48.25, 48], 1)):
        encoded = payload(values)
        assert check(encoded) == expected
        if expected:
            with pytest.raises(ValueError):
                validate_row_span(encoded, (1, 2), domain=ROWSPAN96_DOMAIN)
        else:
            validate_row_span(encoded, (1, 2), domain=ROWSPAN96_DOMAIN)


def test_nonnegative_sum_profiles_cover_subnormal_tree_and_chop_rounding():
    from atlas_mlir_native.bf16_domains import build_bf16_primitive_request
    from atlas_mlir_native.reference import evaluate_bf16_add_nonnegative, evaluate_bf16_row_sum_nonnegative
    from merlin.semantic_compiler.reference import TensorValue
    node = build_bf16_primitive_request("row_sum_bf16_nonnegative").nodes[-1]
    # Each row has a different edge: exact subnormal sum, transition to normal,
    # normal max, and nearest-even BF16 rounding immediately above 1.
    rows = ((1,)*32, (0x7f,)*32, (0x3f80,)*32,
            (0x3f80, 0x3b80, *([0]*30)), (0x3f00, 0x3f00, 0x3c00, 0x3b80, *([0]*28)))
    expected = (0x20, 0x2fe, 0x4200, 0x3f80, 0x3f82)
    values = tuple(code for row in (*rows, *([rows[0]]*27)) for code in row)
    result = evaluate_bf16_row_sum_nonnegative(node, (TensorValue(node.type, values),))
    assert tuple(result.elements[32*i] for i in range(5)) == expected
    for code in (0x8000, 0x8001, 0x3f81, 0x7f80):
        with pytest.raises(ValueError):
            evaluate_bf16_row_sum_nonnegative(node, (TensorValue(node.type, (code,)*1024),))
    node = build_bf16_primitive_request("add_bf16_nonnegative").nodes[-1]
    for left, right, expected in ((1, 1, 2), (0x7f, 1, 0x80), (0x3f80, 0x3b80, 0x3f80),
                                  (0x3f81, 0x3b80, 0x3f81), (0, 0, 0)):
        args = tuple(TensorValue(node.type, (code,)*1024) for code in (left, right))
        assert set(evaluate_bf16_add_nonnegative(node, args).elements) == {expected}
    with pytest.raises(ValueError, match="overflows"):
        evaluate_bf16_add_nonnegative(node, tuple(TensorValue(node.type, (0x7f7f,)*1024) for _ in range(2)))
    for code in (0x8000, 0xbf80, 0x7f80):
        with pytest.raises(ValueError):
            evaluate_bf16_add_nonnegative(node, (TensorValue(node.type, (code,)*1024),
                                               TensorValue(node.type, (0,)*1024)))


@pytest.mark.parametrize("mask_policy,domain", [
    ("bf16-minfinite-zero-rowspan80-nonempty-v1", ROWSPAN80_DOMAIN),
    ("bf16-minfinite-zero-rowspan96-nonempty-v1", ROWSPAN96_DOMAIN),
])
def test_application_mask_policy_explicitly_selects_domain(mask_policy, domain):
    from atlas_mlir_native.already_quantized import POLICY
    from atlas_mlir_native.application import AtlasApplicationAdapter
    from test_attention_application_policy import base
    selected = base() | {"already_quantized": POLICY, "already_quantized_mask": mask_policy}
    adapter = AtlasApplicationAdapter(selected, original_source_sha256="a"*64)
    assert adapter.already_quantized_score_domain == domain
    assert adapter.already_quantized_mask_sentinel == 0xff7f0000
