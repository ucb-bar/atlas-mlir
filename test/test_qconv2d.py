"""Unfolded-code ABI normalization and unchanged qlinear admission."""
from dataclasses import replace

import pytest
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, ModuleOp, StringAttr, TensorType, bf16, i8, i32
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.ir import Block, Region

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.source_normalize import AtlasNormalizeError, normalize
from atlas_mlir_native.source_ops import QLinearOp
from atlas_mlir_native.source_translation import translate_prequantized_contraction
from merlin.compiler.semantic.linalg.linalg_bridge import translate_linalg_text
from merlin.compiler.semantic.target_binding import NativeCompilationError


def fixture(batch=2, height=2, width=3, channels=5, reduction=7, bias=False,
            groups=None, weight_reduction=None):
    import json
    shapes = [(batch * height * width, reduction), (),
              (channels, reduction if weight_reduction is None else weight_reduction), (channels, 1)]
    dtypes = ['uint8', 'int32', 'uint8', 'int32']
    elements = [i8, i32, i8, i32]
    roles = ['x_codes', 'x_exp', 'w_codes', 'w_exp']
    if bias:
        shapes.append((channels,)); dtypes.append('bfloat16'); elements.append(bf16); roles.append('bias')
    types = [TensorType(e, s) for e, s in zip(elements, shapes)]
    output = TensorType(bf16, [batch, channels, height, width])
    block = Block(arg_types=types)
    identity = lambda key: {'prov.source_node_ids': ArrayAttr([StringAttr(key)])}
    schema = 'atlas::qconv2d(Tensor x_codes, Tensor x_exp, Tensor w_codes, Tensor w_exp, Tensor? bias, str out_dtype, str weight_format, int batch, int out_h, int out_w) -> Tensor'
    literals = dict(out_dtype='bfloat16', weight_format='atlas_e4m3', batch=batch, out_h=height, out_w=width)
    if groups is not None:
        schema = schema.replace('int out_w)', 'int out_w, int groups)')
        literals['groups'] = groups
    if not bias:
        literals['bias'] = None
    call = CallOp('opaque', block.args, [output])
    call.attributes.update(identity('conv'))
    call.attributes['m2m.schema'] = StringAttr(schema)
    call.attributes.update({'m2m.literal.' + k: StringAttr(json.dumps(v)) for k, v in literals.items()})
    block.add_ops([call, ReturnOp(call.results[0])])
    module = ModuleOp([FuncOp.external('opaque', types, [output]), FuncOp('main', (types, [output]), Region(block),
                      arg_attrs=ArrayAttr([DictionaryAttr(identity(role)) for role in roles]))])
    record = dict(node_id='conv', site_id='site', target='atlas.qconv2d.default', schema=schema, literals=literals,
                  operands=[dict(role=role, node_id=role, result_index=0, buffer=None, dtype=dtype, shape=list(shape))
                            for role, dtype, shape in zip(roles, dtypes, shapes)],
                  results=[dict(index=0, dtype='bfloat16', shape=[batch, channels, height, width])])
    manifest = dict(schema='m2m.quantization_manifest.v2', adapter_id='atlas_fp8',
                    sites=[dict(site_id='site', status='quantized', execution_route='atlas.native_fp8')],
                    atlas=dict(schema='atlas.m2m_adapter.v1', storage='uint8', buffers={}, source_tensors={}, calls=[record]))
    return module, manifest


def translate(module):
    return translate_linalg_text(str(module), entry='main', target_identity=TARGET_IDENTITY, lowering_policy='diagnostic',
        source_operation_translators={'atlas.qlinear': translate_prequantized_contraction}).request


def assert_unfolded_qlinear_and_nchw_layout(geometry):
    module, manifest = fixture(*geometry)
    normalize(module, manifest)
    module.verify()
    linear = next(op for op in module.walk() if isinstance(op, QLinearOp))
    restored = [op for op in module.walk() if op.name in {'tensor.expand_shape', 'tensor.empty', 'linalg.transpose'}]
    assert len(restored) == 3
    assert all(op.attributes['prov.source_node_ids'] == ArrayAttr([StringAttr('conv')])
               and 'prov.origin_node_ids' not in op.attributes for op in restored)
    assert list(linear.result.type.get_shape()) == [geometry[0] * geometry[1] * geometry[2], geometry[3]]
    source = translate(module)
    assert [n.op for n in source.nodes if n.effect == 'pure'] == ['atlas.qlinear', 'reshape', 'transpose']
    contraction = next(n for n in source.nodes if n.op == 'atlas.qlinear')
    region = replace(source, nodes=tuple(n for n in source.nodes if n.effect == 'input') + (contraction,), outputs=(contraction.id,))
    plan = prepare_source(region)
    assert (plan.m, plan.n, plan.k) == (geometry[0] * geometry[1] * geometry[2], geometry[3], geometry[4])
    assert plan.operand_ids == ('arg0', 'arg2')
    assert plan.exponent_ids == ('arg1', 'arg3')


@pytest.mark.parametrize('geometry', [(2, 2, 3, 5, 7), (1, 32, 32, 768, 768)])
def test_unfolded_codes_become_unchanged_qlinear_and_nchw_layout(geometry):
    assert_unfolded_qlinear_and_nchw_layout(geometry)


def test_overlapping_windows_padding1_stride2_k25_structural_layout():
    # The exported prefix owns padding/im2col. This checks only its unfolded
    # ABI geometry and layout lowering, without numerical qualification.
    batch, in_channels, in_h, in_w = 2, 1, 9, 11
    kernel, padding, stride, out_channels = 5, 1, 2, 3
    assert stride < kernel
    out_h = (in_h + 2 * padding - kernel) // stride + 1
    out_w = (in_w + 2 * padding - kernel) // stride + 1
    reduction = in_channels * kernel * kernel
    assert (out_h, out_w, reduction) == (4, 5, 25)
    assert_unfolded_qlinear_and_nchw_layout((batch, out_h, out_w, out_channels, reduction))


@pytest.mark.parametrize('kwargs,message', [
    ({'groups': 2}, 'grouped convolution is unsupported: groups > 1'),
    ({'reduction': 50, 'weight_reduction': 25}, 'grouped reduction mismatch'),
])
def test_grouped_convolution_refuses_without_mutation(kwargs, message):
    module, manifest = fixture(**kwargs)
    before = str(module)
    uses = {v: tuple(v.uses) for op in module.walk() for v in (*op.operands, *op.results)}
    with pytest.raises(AtlasNormalizeError, match=message):
        normalize(module, manifest)
    assert str(module) == before
    assert {v: tuple(v.uses) for v in uses} == uses


def test_bias_reaches_existing_checked_addition_refusal():
    module, manifest = fixture(1, 32, 32, 768, 768, bias=True)
    normalize(module, manifest)
    source = translate(module)
    contraction = next(n for n in source.nodes if n.op == 'atlas.qlinear')
    region = replace(source, nodes=tuple(n for n in source.nodes if n.effect == 'input') + (contraction,), outputs=(contraction.id,))
    with pytest.raises(NativeCompilationError, match='bias requires source preparation'):
        prepare_source(region)


@pytest.mark.parametrize('geometry', [(2, 2, 3, 5, 7), (1, 32, 32, 768, 768)])
def test_bias_splits_after_readout_before_nchw_layout(geometry):
    from merlin.compiler.semantic.source_preparation import (
        split_prequantized_bias, verify_prequantized_bias,
    )

    module, manifest = fixture(*geometry, bias=True)
    normalize(module, manifest)
    source = translate(module)
    original = next(n for n in source.nodes if n.op == 'atlas.qlinear')
    prepared, receipt = split_prequantized_bias(source)
    assert verify_prequantized_bias(prepared, receipt).digest() == source.digest()
    readout = prepared.node(original.id + '.readout')
    assert readout == replace(original, id=readout.id, inputs=original.inputs[:4])
    addition = prepared.node(original.id)
    assert addition.op == 'bf16_add'
    assert addition.inputs == (readout.id, original.id + '.bias')
    assert dict(addition.attrs)['numeric_semantics'] == 'ieee754-bf16-add-rne-v1'
    broadcast = prepared.node(original.id + '.bias')
    assert broadcast.inputs == (original.inputs[4],)
    assert broadcast.index_maps[0].apply((geometry[0] * geometry[1] * geometry[2] - 1,
                                          geometry[3] - 1)) == (geometry[3] - 1,)
    layouts = [n for n in prepared.nodes if n.op in ('reshape', 'transpose')]
    assert layouts[0].inputs == (addition.id,)
    assert layouts[1].inputs == (layouts[0].id,)
    region = replace(prepared, nodes=tuple(n for n in prepared.nodes if n.id in readout.inputs)
                     + (readout,), outputs=(readout.id,),
                     input_storages=tuple((key, storage) for key, storage in prepared.input_storages
                                          if key in readout.inputs))
    plan = prepare_source(region)
    assert plan.operand_ids == ('arg0', 'arg2')
    assert plan.exponent_ids == ('arg1', 'arg3')
    assert (plan.m, plan.n, plan.k) == (geometry[0] * geometry[1] * geometry[2],
                                       geometry[3], geometry[4])


@pytest.mark.parametrize('field,value', [('batch', True), ('out_h', 0), ('out_w', -1), ('unknown', 1)])
def test_invalid_literals_refuse_without_mutation(field, value):
    import json
    module, manifest = fixture()
    manifest['atlas']['calls'][0]['literals'][field] = value
    call = next(op for op in module.walk() if isinstance(op, CallOp))
    call.attributes['m2m.literal.' + field] = StringAttr(json.dumps(value))
    before = str(module)
    uses = {v: tuple(v.uses) for op in module.walk() for v in (*op.operands, *op.results)}
    with pytest.raises(AtlasNormalizeError):
        normalize(module, manifest)
    assert str(module) == before
    assert {v: tuple(v.uses) for v in uses} == uses


def test_late_manifest_failure_releases_all_staged_uses():
    module, manifest = fixture()
    extra = dict(manifest['atlas']['calls'][0], node_id='missing')
    manifest['atlas']['calls'].append(extra)
    before = str(module)
    uses = {v: tuple(v.uses) for op in module.walk() for v in (*op.operands, *op.results)}
    with pytest.raises(AtlasNormalizeError, match='unmatched native calls'):
        normalize(module, manifest)
    assert str(module) == before
    assert {v: tuple(v.uses) for v in uses} == uses


def test_inconsistent_spatial_shape_refuses_without_mutation():
    import json
    module, manifest = fixture()
    manifest['atlas']['calls'][0]['literals']['out_w'] = 4
    call = next(op for op in module.walk() if isinstance(op, CallOp))
    call.attributes['m2m.literal.out_w'] = StringAttr(json.dumps(4))
    before = str(module)
    uses = {v: tuple(v.uses) for op in module.walk() for v in (*op.operands, *op.results)}
    with pytest.raises(AtlasNormalizeError, match='unfolded rows'):
        normalize(module, manifest)
    assert str(module) == before
    assert {v: tuple(v.uses) for v in uses} == uses
