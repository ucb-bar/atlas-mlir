"""Atlas import is storage preserving and never calls eager quantization."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
from xdsl.dialects.builtin import StringAttr

_spec = importlib.util.spec_from_file_location("atlas_import_ops", Path(__file__).parent / "atlas_import_ops.py")
_ops = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _ops
_spec.loader.exec_module(_ops)

from atlas_mlir_native.capture import load_export as load_atlas_export
from atlas_mlir_native.source_ops import QuantizeActOp, QLinearOp, QKMatmulOp, PVMatmulOp
from atlas_mlir_native.source_normalize import AtlasNormalizeError as AtlasImportError
from atlas_mlir_native.source_normalize import normalize, _operand_attrs
from m2m.capture.trace import snapshot_exported_program
from xdsl.dialects.func import CallOp
import hashlib
import json
from m2m.ir.import_fx import FXImporter


def normalize_export(ep):
    """Exercise the generic importer and target normalizer with a source ABI census.

    The fixture census is assembled from schema records and stamped SSA bindings.
    Production exports authenticate these records through atlas_m2m_adapter.
    """
    snapshot_exported_program(ep, stage="original")
    snapshot_exported_program(ep, stage="prepared")
    module = FXImporter(opaque_custom_ops=True).import_graph(ep)
    def abi(value):
        typ = value.type
        dtype = {"i8": "uint8", "i32": "int32", "i64": "int64", "bf16": "bfloat16",
                 "f32": "float32", "f16": "float16"}.get(str(typ.element_type), str(typ.element_type))
        return dict(dtype=dtype, shape=list(typ.get_shape()))
    source, buffers = {}, {}
    for name, tensor in ep.state_dict.items():
        raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        record = dict(dtype=str(tensor.dtype).removeprefix("torch."), shape=list(tensor.shape),
                      bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        source[name] = record
        buffers[name] = dict(source_name=name, **record)
    calls = []
    for op in module.walk():
        if not isinstance(op, CallOp):
            continue
        schema = op.attributes["m2m.schema"].data
        literals = {key.removeprefix("m2m.literal."): json.loads(value.data)
                    for key, value in op.attributes.items() if key.startswith("m2m.literal.")}
        kind = schema.split("::", 1)[1].split("(", 1)[0]
        operator = getattr(torch.ops.atlas, kind).default
        roles = [arg.name for arg in operator._schema.arguments if arg.name not in literals]
        operands = []
        for role, value in zip(roles, op.operands, strict=True):
            attrs, index = _operand_attrs(value)
            target = attrs.get("prov.get_attr")
            operands.append(dict(role=role, node_id=attrs["prov.source_node_ids"].data[0].data,
                                 result_index=index, buffer=target.data if target is not None else None,
                                 **abi(value)))
        calls.append(dict(node_id=op.attributes["prov.source_node_ids"].data[0].data,
                          site_id="fixture", invocation=len(calls), target=str(operator), schema=schema,
                          literals=literals, operands=operands,
                          results=[dict(index=index, **abi(value)) for index, value in enumerate(op.results)]))
    manifest = dict(schema="m2m.quantization_manifest.v2", adapter_id="atlas_fp8",
                    sites=[dict(site_id="fixture", status="quantized", execution_route="atlas.native_fp8")],
                    atlas=dict(schema="atlas.m2m_adapter.v1", storage="uint8", calls=calls,
                               buffers=buffers, source_tensors=source))
    return normalize(module, manifest)


class Linear(torch.nn.Module):
    def __init__(self, code_dtype="uint8", bias=True):
        super().__init__()
        self.code_dtype = code_dtype
        self.register_buffer("codes", torch.tensor([[0, 56, 184], [126, 254, 128]], dtype=torch.uint8))
        self.register_buffer("exponent", torch.tensor([[-13], [7]], dtype=torch.int32))
        self.register_buffer("bias", torch.tensor([1, -1], dtype=torch.bfloat16) if bias else None)

    def forward(self, x):
        c, e = torch.ops.atlas.quantize_act.default(x, "static", "tensor", "ref", -9, self.code_dtype)
        return torch.ops.atlas.qlinear.default(c, e, self.codes, self.exponent, self.bias, "bfloat16", "atlas_e4m3")


def export_linear(**kwargs):
    return torch.export.export(Linear(**kwargs), (torch.ones(2, 3, dtype=torch.bfloat16),))


@pytest.mark.parametrize("bias", [True, False])
def test_linear_typed_values_and_unchanged_lifted_bytes(bias):
    ep = export_linear(bias=bias)
    module = normalize_export(ep)
    module.verify()
    quant = next(op for op in module.walk() if isinstance(op, QuantizeActOp))
    linear = next(op for op in module.walk() if isinstance(op, QLinearOp))
    assert linear.lhs_codes is quant.codes
    assert linear.lhs_exp is quant.exponent
    assert str(quant.codes.type) == "tensor<2x3xi8>"
    assert str(quant.exponent.type) == "tensor<i32>"
    assert str(linear.result.type) == "tensor<2x2xbf16>"
    assert quant.attributes["static_exp"].value.data == -9
    assert linear.attributes["code_dtype"] == StringAttr("uint8")
    assert linear.attributes["exponent_dtype"] == StringAttr("int32")
    assert len(linear.operands) == (5 if bias else 4)
    assert ep.state_dict["codes"].flatten().tolist() == [0, 56, 184, 126, 254, 128]
    assert ep.state_dict["exponent"].flatten().tolist() == [-13, 7]
    assert not any(op.name in ("func.call", "tensor.empty") for op in module.walk())
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser
    from atlas_mlir_native.source_ops import Atlas
    context = Context(allow_unregistered=True)
    for dialect in (Builtin, Func, Atlas):
        context.load_dialect(dialect)
    reparsed = Parser(context, str(module)).parse_module()
    assert reparsed is not None
    reparsed.verify()
    assert any(isinstance(op, QLinearOp) for op in reparsed.walk())


def test_saved_export_and_lifted_storage(tmp_path):
    ep = export_linear()
    path = tmp_path / "native.pt2"
    torch.export.save(ep, path)
    loaded = load_atlas_export(path, ops_module="atlas_import_ops")
    assert torch.equal(loaded.state_dict["codes"], ep.state_dict["codes"])
    assert torch.equal(loaded.state_dict["exponent"], ep.state_dict["exponent"])
    module = normalize_export(loaded)
    module.verify()
    linear = next(op for op in module.walk() if isinstance(op, QLinearOp))
    assert str(linear.rhs_codes.type) == "tensor<2x3xi8>"
    assert str(linear.rhs_exp.type) == "tensor<2x1xi32>"


def test_registration_precedes_load():
    events = []
    with patch("atlas_mlir_native.capture.import_module", side_effect=lambda name: events.append("register")), \
            patch("torch.export.load", side_effect=lambda path: events.append("load")):
        load_atlas_export("unused.pt2")
    assert events == ["register", "load"]


@pytest.mark.parametrize("name,cls", [("qk_matmul", QKMatmulOp), ("pv_matmul", PVMatmulOp)])
def test_attention_typed_storage(name, cls):
    class Attention(torch.nn.Module):
        def forward(self, a, ae, b, be):
            return getattr(torch.ops.atlas, name).default(a, ae, b, be)
    inputs = (torch.zeros(1, 2, 3, dtype=torch.uint8), torch.tensor(-11, dtype=torch.int32),
              torch.zeros(1, 4, 3, dtype=torch.uint8), torch.tensor([[[6]]], dtype=torch.int32))
    module = normalize_export(torch.export.export(Attention(), inputs))
    module.verify()
    op = next(op for op in module.walk() if isinstance(op, cls))
    assert str(op.result.type) == "tensor<1x2x4xf32>"
    assert str(op.lhs_exp.type) == "tensor<i32>"
    assert str(op.rhs_exp.type) == "tensor<1x1x1xi32>"


def test_qconv2d_refused_even_with_opaque_fallback():
    class Conv(torch.nn.Module):
        def forward(self, c, e, w, we):
            return torch.ops.atlas.qconv2d.default(c, e, w, we, None, "bfloat16", "atlas_e4m3", 1, 1, 2)
    inputs = (torch.zeros(2, 3, dtype=torch.uint8), torch.tensor(-11, dtype=torch.int32),
              torch.zeros(4, 3, dtype=torch.uint8), torch.tensor(6, dtype=torch.int32))
    ep = torch.export.export(Conv(), inputs)
    with pytest.raises(AtlasImportError, match="convolution lowering is not qualified"):
        normalize_export(ep)


def test_float8_projection_refused():
    with pytest.raises(AtlasImportError, match="logical uint8 codes"):
        normalize_export(export_linear(code_dtype="float8_e4m3fn"))


def test_folded_pv_shift_refused():
    class Folded(torch.nn.Module):
        def forward(self, a, ae, b, be):
            return torch.ops.atlas.pv_matmul.default(a, ae, b, be, 2)
    inputs = (torch.zeros(1, 2, 3, dtype=torch.uint8), torch.tensor(-11, dtype=torch.int32),
              torch.zeros(1, 4, 3, dtype=torch.uint8), torch.tensor(6, dtype=torch.int32))
    with pytest.raises(AtlasImportError, match="pop_shift must be zero"):
        normalize_export(torch.export.export(Folded(), inputs))


def test_exponent_dtype_must_remain_signed_int32():
    class BadExponent(torch.nn.Module):
        def forward(self, a, ae, b, be):
            return torch.ops.atlas.qk_matmul.default(a, ae, b, be)
    inputs = (torch.zeros(1, 2, 3, dtype=torch.uint8), torch.tensor(-11, dtype=torch.int64),
              torch.zeros(1, 4, 3, dtype=torch.uint8), torch.tensor(6, dtype=torch.int32))
    with pytest.raises(AtlasImportError, match="ordered tensor shape or dtype differs"):
        normalize_export(torch.export.export(BadExponent(), inputs))
