"""CPU binary64 placement is explicit and keeps source arithmetic intact."""

from dataclasses import replace

import pytest
import test_application_c_package as fixtures
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.gelu import GELU_POLICY
from atlas_mlir_native.pointwise_math import MATH_POLICY
from atlas_mlir_native.reduction import REDUCTION_POLICY
from atlas_mlir_native.row_softmax import GUARDED_SOFTMAX_POLICY, MASKED_SOFTMAX_POLICY
from atlas_mlir_native.sigmoid import SIGMOID_POLICY
from merlin.compile.c_float_casts import FLOAT_CAST_OPERATIONS
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.tests.runtime.test_float_arithmetic_c import request


def policy():
    return {
        **fixtures.ApplicationLayoutPolicyTest().policy(),
        "schema": "atlas.diagnostic_application_precision.v19",
        "reduction": REDUCTION_POLICY,
        "pointwise_math": MATH_POLICY,
        "sigmoid": SIGMOID_POLICY,
        "masked_softmax": MASKED_SOFTMAX_POLICY,
        "guarded_masked_softmax": GUARDED_SOFTMAX_POLICY,
        "gelu": GELU_POLICY,
        "host_predicates": ["mask"],
        "host_integer_ops": ["i64_add"],
        "host_float_casts": list(FLOAT_CAST_OPERATIONS),
        "host_float_arithmetic": ["f64_mul"],
    }


def test_binary64_multiply_is_selected_before_compilation_and_uses_no_device_program():
    selected = policy()
    source = replace(request((1, 360)), target_identity=selected["target_identity"], lowering_policy="diagnostic")
    adapter = AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"])
    (region,) = adapter.plan(source).selected
    assert region.placement == "host"
    compiled = adapter.compile_region(None, source, region, "host_multiply", None, None, None)
    assert not compiled.programs
    assert compiled.description["arithmetic"] == "exact-source-f64-multiply"
    assert b"__uint128_t" in compiled.files["host_multiply.c"]
    assert compiled.inputs == ("a", "b")
    # No compiler math helper is selected or needed by this policy.
    assert "source_constant_math" not in selected
    old = {k: v for k, v in selected.items() if k != "host_float_arithmetic"}
    old["schema"] = "atlas.diagnostic_application_precision.v17"
    with pytest.raises(NativeCompilationError, match="no compiled application route"):
        AtlasApplicationAdapter(old, original_source_sha256=old["source_sha256"]).plan(source)


@pytest.mark.parametrize("operations", [[], ["f64_mul", "f64_mul"], ["f64_add"], [True]])
def test_unknown_or_duplicate_cpu_arithmetic_selection_fails(operations):
    selected = {**policy(), "host_float_arithmetic": operations}
    with pytest.raises(NativeCompilationError, match="float arithmetic selection"):
        AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"])


@pytest.mark.parametrize("operation", ["bf16_add", "bf16_sub"])
def test_bf16_source_rounding_uses_an_explicit_cpu_region(operation):
    from merlin.tests.runtime.test_bf16_arithmetic_c import request as bf16_request

    selected = policy()
    source = replace(bf16_request((1, 50, 720), operation),
                     target_identity=selected["target_identity"], lowering_policy="diagnostic")
    with pytest.raises(NativeCompilationError, match="no compiled application route"):
        AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"]).plan(source)
    selected["host_float_arithmetic"] = [operation]
    adapter = AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"])
    (region,) = adapter.plan(source).selected
    assert region.placement == "host"
    compiled = adapter.compile_region(None, source, region, "source_arithmetic", None, None, None)
    assert not compiled.programs
    assert compiled.description["numeric_semantics"] == f"ieee754-bf16-{operation[5:]}-rne-v1"
    assert compiled.description["arithmetic"] == "exact-source-" + operation.replace("_", "-")
    assert compiled.inputs == ("a", "b")
