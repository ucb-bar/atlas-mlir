"""Source sine/cosine needs explicit CPU placement and a selected library."""

from dataclasses import replace

import pytest
import test_application_float_arithmetic as fixtures
from atlas_mlir_native.application import AtlasApplicationAdapter

from merlin.compile.c_unary_math import OPERATIONS
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.tests.runtime.test_unary_math_c import policy as math_policy
from merlin.tests.runtime.test_unary_math_c import request


@pytest.mark.parametrize("operation", sorted(OPERATIONS))
def test_selected_cpu_math_preserves_library_and_effect_requirements(operation):
    selected = {
        **fixtures.policy(),
        "schema": "atlas.diagnostic_application_precision.v20",
        "host_unary_math": math_policy(),
    }
    adapter = AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"])
    source = replace(request(operation), target_identity=selected["target_identity"], lowering_policy="diagnostic")
    (alternative,) = adapter.plan(source).selected
    assert alternative.placement == "host"
    result = adapter.compile_region(None, source, alternative, "unary", None, None, None)
    assert result.programs == ()
    assert result.description["host_linkage"]["symbols"] == [OPERATIONS[operation][1]]
    assert result.description["framework_equivalence"] == "not_established"
    assert result.description["host_abi"] == "lp64d"
    with pytest.raises(NativeCompilationError, match="no compiled application route"):
        AtlasApplicationAdapter(fixtures.policy(), original_source_sha256=selected["source_sha256"]).plan(source)


def test_unknown_math_permissions_are_rejected():
    selected = {
        **fixtures.policy(),
        "schema": "atlas.diagnostic_application_precision.v20",
        "host_unary_math": {**math_policy(), "effects": "ignore-everything"},
    }
    with pytest.raises(NativeCompilationError, match="numerical/effect"):
        AtlasApplicationAdapter(selected, original_source_sha256=selected["source_sha256"])
