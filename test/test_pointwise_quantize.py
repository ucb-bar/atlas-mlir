"""Fused pointwise static PACK retains source rounding, tails and ABI guards."""
from dataclasses import replace

import pytest

from atlas_mlir_native import pointwise_quantize as fused
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.pointwise import BF16_POINTWISE_POLICY
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps, IndexMap
from test_already_quantized import quantizer_source
from test_pointwise_narrowing import adapter as base_adapter, case as narrowed_case
from test_bf16_pointwise import request as bf16_case
from test_pointwise_boundary_casts import case as unary_case

PATTERNS = ('pointwise_narrowing', 'pointwise_input_widening', 'pointwise_quantize')


def adapter():
    result = base_adapter()
    result.bf16_pointwise = BF16_POINTWISE_POLICY
    return result


def case(shape=(3, 35), kind='pointwise', exponent=-7, broadcast=False):
    if kind == 'pointwise':
        source = bf16_case(shape)
    elif kind == 'pointwise_narrowing':
        source = narrowed_case(shape=shape)
    else:
        source = unary_case(shape, input_dtype='bf16')
    if broadcast:
        a, b, op, *rest = source.nodes
        b = replace(b, type=replace(b.type, shape=(shape[-1],)))
        mapping = IndexMap(len(shape), (tuple([0]*(len(shape)-1) + [1]),), (0,))
        op = replace(op, index_maps=(op.index_maps[0], mapping, op.index_maps[-1]))
        source = replace(source, nodes=(a, b, op, *rest))
    value = source.node(source.outputs[0])
    quantizer = quantizer_source(fixed=exponent)
    codes, scale = quantizer.nodes[-2:]
    codes = replace(codes, id='packed', inputs=(value.id,), type=replace(codes.type, shape=shape))
    scale = replace(scale, inputs=(value.id,))
    return replace(source, nodes=(*source.nodes, codes, scale), outputs=(codes.id, scale.id),
                   output_storages=('external',)*2)


@pytest.mark.parametrize('shape', [(1, 7), (33, 65), (2, 3, 35)])
@pytest.mark.parametrize('kind', ['pointwise', 'pointwise_narrowing', 'pointwise_input_widening'])
def test_selector_fuses_only_private_bf16_result(shape, kind):
    source = case(shape, kind)
    group = fused.prepare(source, adapter(), kind)
    selected, = AdjacentDeviceAdapter(adapter(), patterns=PATTERNS).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'pointwise_quantize'
    assert group.exponent == -7
    assert group.input_ids == tuple(n.id for n in source.nodes if n.effect == 'input')


def test_shared_result_dynamic_scale_and_changed_rounding_refuse():
    source = case(kind='pointwise_narrowing')
    producer = source.nodes[-3]
    wrapper = AdjacentDeviceAdapter(adapter(), patterns=PATTERNS)
    shared = replace(source, outputs=(*source.outputs, producer.id), output_storages=('external',)*3)
    extra = SemanticNode('extra', 'copy', (producer.id,), producer.type, index_maps=pointwise_identity_maps(2, 1))
    reader = replace(source, nodes=(*source.nodes, extra), outputs=(*source.outputs, extra.id),
                     output_storages=('external',)*3)
    for request in (shared, reader):
        assert all(dict(r.metadata)['kind'] != 'pointwise_quantize' for r in wrapper.plan(request).selected)
    for key, value in [('mode', 'dynamic'), ('static_exp', True), ('static_exp', 1 << 31), ('rule', 'unknown')]:
        nodes = tuple(replace(n, attrs=tuple(sorted(dict(n.attrs, **{key: value}).items())))
                      if n.op.startswith('atlas.quantize_act.') else n for n in source.nodes)
        with pytest.raises(ValueError):
            fused.prepare(replace(source, nodes=nodes), adapter(), 'pointwise_narrowing')
    changed = replace(producer, attrs=())
    with pytest.raises(ValueError):
        fused.prepare(replace(source, nodes=(*source.nodes[:-3], changed, *source.nodes[-2:])),
                      adapter(), 'pointwise_narrowing')
    base = adapter()
    base.activation_quantizer_route = 'host'
    assert all(dict(r.metadata)['kind'] != 'pointwise_quantize'
               for r in AdjacentDeviceAdapter(base, patterns=PATTERNS).plan(source).selected)


def test_streamed_frames_refuse_the_single_dispatch_fusion():
    from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
    with codegen_options(CodegenOptions(io_memory_bytes=65536)):
        with pytest.raises(ValueError, match='single resident frame'):
            fused.prepare(case((513, 513)), adapter(), 'pointwise')
