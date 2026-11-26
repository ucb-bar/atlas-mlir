"""A host body can be shared only when its regenerated C is identical."""
from dataclasses import replace
from types import SimpleNamespace

from atlas_mlir_native.application import AtlasApplicationAdapter
from merlin.compiler.schedule.region_loop import correspondence
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps


def region(prefix):
    tensor = TensorType((1,), "f32", "raw-bit-pattern")
    x, w, y = (prefix + key for key in ("x", "w", "y"))
    return KernelRequest((SemanticNode(x, "input", (), tensor, effect="input"),
                          SemanticNode(w, "input", (), tensor, effect="input"),
                          SemanticNode(y, "f32_add", (x, w), tensor,
                              (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
                              index_maps=pointwise_identity_maps(1, 2))),
                         (y,), ("external",), ((x, "external"), (w, "external")), "synthetic")


def test_shared_host_export_retains_current_source_receipt(tmp_path):
    adapter = object.__new__(AtlasApplicationAdapter)
    alternative = SimpleNamespace(placement="host", metadata=(("kind", "float_arithmetic"),))
    first, second = region("first_"), region("second_")
    package = adapter.export_region(first, alternative, "body", tmp_path, tmp_path)
    bound = adapter.bind_loop_region(package, second, alternative,
                                    correspondence(first.record(), second.record()), tmp_path, tmp_path)
    assert bound is not None
    assert bound.files == package.files
    assert bound.description["source_request_digest"] == second.digest()
    assert bound.description["source_request_digest"] != package.description["source_request_digest"]
    assert bound.inputs == ("second_x", "second_w")


def test_device_body_declines_without_dynamic_resident_address_abi(tmp_path):
    adapter = object.__new__(AtlasApplicationAdapter)
    # In particular, no attempt to export a device region at another region's path.
    package = SimpleNamespace(programs=("segments/body",))
    assert adapter.bind_loop_region(package, None, SimpleNamespace(placement="accelerator"),
                                    {}, tmp_path, tmp_path) is None


def test_changed_generated_host_body_is_not_reused(tmp_path):
    adapter = object.__new__(AtlasApplicationAdapter)
    alternative = SimpleNamespace(placement="host", metadata=(("kind", "float_arithmetic"),))
    source = region("first_")
    package = adapter.export_region(source, alternative, "body", tmp_path, tmp_path)
    package = replace(package, files={**package.files, "unexpected.c": b"int unexpected;"})
    assert adapter.bind_loop_region(package, source, alternative, {}, tmp_path, tmp_path) is None
