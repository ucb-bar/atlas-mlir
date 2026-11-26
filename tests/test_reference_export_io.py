"""Serialized source intake uses raw bits, never framework execution/goldens."""
import builtins
import hashlib
import json
from unittest.mock import patch
import zipfile

import numpy as np
import pytest

from atlas_mlir_native.export_loader import ExportBundle, FixtureCase, TensorRecord
from atlas_mlir_native.reference_eval import evaluate
from atlas_mlir_native.reference_eval.export_io import export_sources


def meta(dtype=13, shape=(2,)):
    strides, stride = [], 1
    for dim in reversed(shape):
        strides.insert(0, {"as_int": stride})
        stride *= dim
    return dict(dtype=dtype, sizes=[{"as_int": d} for d in shape], strides=strides,
                storage_offset={"as_int": 0}, layout=7,
                device={"type": "cpu", "index": None}, requires_grad=False)


def tensor(name):
    return {"as_tensor": {"name": name}}


def argument(name, value, kind=1):
    return dict(name=name, arg=value, kind=kind)


def document():
    graph = dict(inputs=[tensor("p"), tensor("x")], outputs=[tensor("alias")],
                 nodes=[dict(name="alias", target="torch.ops.aten.alias.default",
                             inputs=[argument("self", tensor("x"))], outputs=[tensor("alias")])],
                 tensor_values={name: meta() for name in ("p", "x", "alias")},
                 is_single_tensor_return=False)
    signature = dict(input_specs=[{"parameter": {"arg": {"name": "p"}, "parameter_name": "weight"}},
                                  {"user_input": {"arg": tensor("x")}}],
                     output_specs=[{"user_output": {"arg": tensor("alias")}}])
    return dict(schema_version={"major": 8, "minor": 17},
                graph_module=dict(graph=graph, signature=signature))


class PoisonOutput:
    def __getattribute__(self, name):
        raise AssertionError("fixture output was accessed")


def bundle(tmp_path, doc=None, *, config=None, compression=zipfile.ZIP_STORED, duplicate=False):
    doc = doc or document()
    path = tmp_path / "model.pt2"
    record = dict(path_name="weight_0", use_pickle=False, tensor_meta=meta())
    config = config if config is not None else {"weight": record}
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.writestr("model/models/model.json", json.dumps(doc))
        archive.writestr("model/byteorder", "little")
        archive.writestr("model/data/weights/model_weights_config.json", json.dumps({"config": config}))
        archive.writestr("model/data/constants/model_constants_config.json", '{"config":{}}')
        archive.writestr("model/data/weights/weight_0", bytes.fromhex("0080c17f"))
        if duplicate:
            archive.writestr("model/byteorder", "little")
    fixture_path = tmp_path / "fixture.raw"
    fixture_path.write_bytes(bytes.fromhex("803f0080"))
    fixture = TensorRecord("input", "BF16", (2,), fixture_path, 0, 4)
    inventory = {file.name: dict(bytes=file.stat().st_size,
                                sha256=hashlib.sha256(file.read_bytes()).hexdigest())
                 for file in (path, fixture_path)}
    return ExportBundle(tmp_path, tmp_path, tmp_path, path, fixture_path,
                        (FixtureCase(0, (fixture,), (PoisonOutput(),)),), {}, inventory)


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_raw_source_and_fixture_bits_without_framework_or_goldens(tmp_path, compression):
    source = bundle(tmp_path, compression=compression)
    original = builtins.__import__

    def no_framework(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "tensorflow", "jax"}:
            raise AssertionError("framework import attempted")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=no_framework):
        nodes, constants, inputs = export_sources(source)
        result = evaluate(nodes, constants, inputs)
    assert constants["weight"].raw.tolist() == [0x8000, 0x7fc1]
    assert inputs["x"].raw.tolist() == [0x3f80, 0x8000]
    assert result.outputs[0].raw.tolist() == [0x3f80, 0x8000]
    assert not constants["weight"].raw.flags.writeable
    assert not result.ordinary_framework_kernels_called


def test_binding_and_multiresult_translation(tmp_path):
    doc = document()
    graph = doc["graph_module"]["graph"]
    graph["nodes"] = [dict(name="quantize", target="torch.ops.atlas.quantize_act.default",
                           inputs=[argument("x", tensor("x")),
                                   argument("dtype", {"as_scalar_type": 13}, 2)],
                           outputs=[tensor("codes"), tensor("scale")])]
    graph["tensor_values"].update(codes=meta(1), scale=meta(4, ()))
    graph["outputs"] = [tensor("codes"), tensor("scale")]
    doc["graph_module"]["signature"]["output_specs"] = [
        {"user_output": {"arg": x}} for x in graph["outputs"]]
    nodes, _, _ = export_sources(bundle(tmp_path, doc))
    node = nodes[2]
    assert node["id"] == "quantize"
    assert node["target"] == "atlas.quantize_act.default"
    assert node["args"] == [{"value_id": "value:x"}]
    assert node["kwargs"] == {"dtype": {"kind": "dtype", "value": "bfloat16"}}
    assert [(r["id"], r["dtype"], r["shape"]) for r in node["results"]] == [
        ("value:codes", "uint8", [2]), ("value:scale", "int32", [])]


def test_reauthenticate_before_source_storage_access(tmp_path):
    source = bundle(tmp_path)
    source.graph_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after intake"):
        export_sources(source)


@pytest.mark.parametrize("index", [-1, 1, True])
def test_fixture_index_must_select_authenticated_case(tmp_path, index):
    with pytest.raises(ValueError, match="fixture case index"):
        export_sources(bundle(tmp_path), index)


def test_pickle_storage_refused(tmp_path):
    config = {"weight": dict(path_name="weight_0", use_pickle=True, tensor_meta=meta())}
    with pytest.raises(ValueError, match="pickled source tensors"):
        export_sources(bundle(tmp_path, config=config))


def test_traversal_storage_member_refused_before_read(tmp_path):
    config = {"weight": dict(path_name="../absent", use_pickle=False, tensor_meta=meta())}
    with pytest.raises(ValueError, match="serialized export member"):
        export_sources(bundle(tmp_path, config=config))


def test_schema_revision_and_source_mutations_refused(tmp_path):
    doc = document()
    doc["schema_version"]["minor"] = 18
    with pytest.raises(ValueError, match="expected 8.17"):
        export_sources(bundle(tmp_path, doc))
    doc = document()
    doc["graph_module"]["signature"]["output_specs"] = [
        {"buffer_mutation": {"arg": {"name": "alias"}, "buffer_name": "weight"}}]
    with pytest.raises(ValueError, match="mutations"):
        export_sources(bundle(tmp_path, doc))


def test_dynamic_shape_refusal_names_source_node(tmp_path):
    doc = document()
    doc["graph_module"]["graph"]["tensor_values"]["alias"]["sizes"] = [
        {"as_expr": {"expr_str": "u0", "hint": None}}]
    with pytest.raises(ValueError, match="source node alias.*runtime shape support"):
        export_sources(bundle(tmp_path, doc))


def test_duplicate_members_and_truncated_storage_refused(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate name"):
        source = bundle(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate serialized export archive member"):
        export_sources(source)
    config = {"weight": dict(path_name="weight_0", use_pickle=False, tensor_meta=meta(shape=(3,)))}
    with pytest.raises(ValueError, match="truncated or misaligned"):
        export_sources(bundle(tmp_path, config=config))
