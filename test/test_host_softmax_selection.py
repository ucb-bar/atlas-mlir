"""The scalar host softmax family runs only where an admitted policy has no device softmax route."""

from dataclasses import replace

import pytest
import test_application_c_package as fixtures
import atlas_mlir_native.application as application
from atlas_mlir_native.application import AtlasApplicationAdapter
from merlin.compiler.lower.c.host_ops.softmax_precision import host_softmax_policy
from merlin.compiler.semantic.linalg.linalg_recognition import LinalgBridgeError
from merlin.compiler.semantic.target_binding import NativeCompilationError

from atlas_test_fixtures.host_softmax import make_request

HOST_SOFTMAX = host_softmax_policy()


def selected(policy, source):
    adapter = AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])
    return adapter, adapter.plan(source).selected


def request(policy, shape, mask_kind="none"):
    extra = {"fill_bits": 0xFF7FFFFF} if mask_kind == "select" else {}
    return replace(make_request(shape, mask_kind, **extra), target_identity=policy["target_identity"],
                   lowering_policy="diagnostic")


def test_device_softmax_route_is_kept_when_host_softmax_is_admitted():
    policy = fixtures.ApplicationLayoutPolicyTest().policy()
    source = request(policy, (2, 3))
    for admitted in (policy, {**policy, "host_softmax": HOST_SOFTMAX}):
        _, regions = selected(admitted, source)
        assert [dict(region.metadata)["kind"] for region in regions] == ["softmax"]


@pytest.mark.parametrize("mask_kind", ["none", "select"])
def test_device_refusal_falls_back_to_one_self_contained_host_region(mask_kind, monkeypatch):
    def refuse(*args, **kwargs):
        raise LinalgBridgeError("device softmax family refuses this region")

    monkeypatch.setattr(application, "_lower_semantic_softmax", refuse)
    policy = fixtures.ApplicationLayoutPolicyTest().policy()
    source = request(policy, (2, 3), mask_kind)
    with pytest.raises(LinalgBridgeError):
        selected(policy, source)
    adapter, (region,) = selected({**policy, "host_softmax": HOST_SOFTMAX}, source)
    assert region.placement == "host"
    assert dict(region.metadata)["kind"] == "host_softmax"
    compiled = adapter.compile_region(None, source, region, "host_softmax", None, None, None)
    assert not compiled.programs
    assert compiled.outputs == ("result",)
    assert "host_linkage" not in compiled.description
    assert b"static float merlin_host_expf(float x)" in compiled.files["host_softmax.c"]


@pytest.mark.parametrize("value", [True, False, "true", 1, {**HOST_SOFTMAX, "order": "other"},
                                   {**HOST_SOFTMAX, "linkage": {}}])
def test_host_softmax_admission_needs_its_declared_order(value):
    policy = {**fixtures.ApplicationLayoutPolicyTest().policy(), "host_softmax": value}
    with pytest.raises(NativeCompilationError, match="its declared order"):
        AtlasApplicationAdapter(policy, original_source_sha256=policy["source_sha256"])


@pytest.mark.parametrize("mask_kind", ["none", "additive", "select"])
def test_exported_host_softmax_executes_and_rejects_aliases(mask_kind, monkeypatch, tmp_path):
    """Compile the published unit, then check its C ABI and pre-write refusal."""
    import ctypes
    import shutil
    import subprocess

    compiler = shutil.which("cc")
    assert compiler, "host softmax execution regression requires a local C compiler"

    def refuse(*args, **kwargs):
        raise LinalgBridgeError("device softmax family refuses this region")

    monkeypatch.setattr(application, "_lower_semantic_softmax", refuse)
    policy = fixtures.ApplicationLayoutPolicyTest().policy()
    source = request(policy, (2, 4), mask_kind)
    adapter, (region,) = selected({**policy, "host_softmax": HOST_SOFTMAX}, source)
    compiled = adapter.compile_region(None, source, region, "softmax_exec", None, None, None)
    for name, data in compiled.files.items():
        (tmp_path / name).write_bytes(data)
    library = tmp_path / "softmax.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-O2", "-fno-fast-math",
                    "-ffp-contract=off", "-fno-tree-vectorize", str(tmp_path / "softmax_exec.c"),
                    "-o", str(library)], check=True, capture_output=True)
    fn = ctypes.CDLL(str(library)).softmax_exec
    pointers = ctypes.POINTER(ctypes.c_void_p)
    fn.argtypes, fn.restype = [ctypes.c_void_p, pointers, pointers], ctypes.c_int
    scores = (ctypes.c_float * 8)(*([0.0] * 8))
    output = (ctypes.c_float * 8)(*([-7.0] * 8))
    mask = ((ctypes.c_ubyte * 8)(*([1, 1, 0, 0] * 2)) if mask_kind == "select"
            else (ctypes.c_float * 8)(*([0.0, 0.0, float("-inf"), float("-inf")] * 2)))
    address = lambda value: ctypes.cast(value, ctypes.c_void_p).value
    inputs = (ctypes.c_void_p * 2)(address(scores), address(mask))
    outputs = (ctypes.c_void_p * 1)(address(output))
    assert fn(None, inputs, outputs) == 0
    expected = [0.25] * 8 if mask_kind == "none" else [0.5, 0.5, 0.0, 0.0] * 2
    assert list(output) == expected
    original = bytes(scores)
    aliases = (ctypes.c_void_p * 1)(address(scores))
    assert fn(None, inputs, aliases) == -1
    assert bytes(scores) == original
    if mask_kind == "select":
        mask[0] = 2
        original = bytes(output)
        assert fn(None, inputs, outputs) == -2
        assert bytes(output) == original
