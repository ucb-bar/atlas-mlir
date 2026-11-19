"""Private static quantization preserves the selected softmax codes and guards."""
from dataclasses import replace

import numpy as np
import pytest

from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native import softmax_quantize as fused
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from test_already_quantized import quantizer_source
from test_pointwise_narrowing import adapter
from test_softmax_narrowing import case as softmax_case

PATTERNS = ('softmax_narrowing', 'softmax_input_widening', 'softmax_quantize')


def case(shape=(33, 65), kind='softmax', exponent=-8, input_dtype='f32'):
    from test_softmax_input_widening import case as widened_case

    source = widened_case(shape, kind) if input_dtype == 'bf16' else softmax_case(shape, kind)
    probability = source.node(source.outputs[0])
    quantizer = quantizer_source(fixed=exponent, count=int(np.prod(shape)))
    codes, scale = quantizer.nodes[-2:]
    codes = replace(codes, inputs=(probability.id,), type=replace(codes.type, shape=shape))
    scale = replace(scale, inputs=(probability.id,))
    return replace(source, nodes=(*source.nodes, codes, scale), outputs=(codes.id, scale.id),
                   output_storages=('external', 'external'))


@pytest.mark.parametrize('input_dtype', ['f32', 'bf16'])
@pytest.mark.parametrize('kind', ['softmax', 'masked_softmax', 'guarded_softmax'])
@pytest.mark.parametrize('shape', [(1, 7), (33, 65), (2, 3, 35)])
def test_selection_retains_the_original_softmax_arithmetic(kind, shape, input_dtype):
    source = case(shape, kind, input_dtype=input_dtype)
    base = adapter()
    group = fused.prepare(source, base, kind, input_dtype=input_dtype)
    selected, = AdjacentDeviceAdapter(base, patterns=PATTERNS).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'softmax_quantize'
    assert group.softmax.precision.source_region.shape == shape
    assert group.exponent == -8


def test_shared_results_and_dynamic_quantizers_keep_separate_regions():
    source = case()
    base = adapter()
    wrapper = AdjacentDeviceAdapter(base, patterns=PATTERNS)
    probability = source.nodes[-3]
    shared = replace(source, outputs=(*source.outputs, probability.id), output_storages=('external',)*3)
    copy = SemanticNode('extra', 'copy', (probability.id,), probability.type,
                        index_maps=pointwise_identity_maps(2, 1))
    reader = replace(source, nodes=(*source.nodes, copy), outputs=(*source.outputs, copy.id),
                     output_storages=('external',)*3)
    for request in (shared, reader):
        assert all(dict(row.metadata)['kind'] != 'softmax_quantize' for row in wrapper.plan(request).selected)
    for key, value in [('mode', 'dynamic'), ('static_exp', True), ('rule', 'unknown')]:
        nodes = tuple(replace(n, attrs=tuple(sorted(dict(n.attrs, **{key: value}).items())))
                      if n.op.startswith('atlas.quantize_act.') else n for n in source.nodes)
        with pytest.raises(ValueError):
            fused.prepare(replace(source, nodes=nodes), base, 'softmax')
    base.activation_quantizer_route = 'host'
    assert all(dict(row.metadata)['kind'] != 'softmax_quantize' for row in wrapper.plan(source).selected)


def test_only_final_stores_are_packed_and_ambiguous_bodies_refuse():
    from atlas_mlir_native.row_output_quantize import final_stores
    from merlin.frontends.linalg_mlir import parse_mlir_text

    text = '''module {
      func.func @row() -> !atlas.virtual_state {
        %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
        %p = arith.constant 8192 : i32
        %n = arith.constant 2048 : i32
        %x = "atlas.virtual_vpu_splat"() {bits = 0 : i32} : () -> !atlas.virtual_bf16
        %a = "atlas.virtual_dma_store_bf16"(%s,%x,%p,%n) : (!atlas.virtual_state,!atlas.virtual_bf16,i32,i32) -> !atlas.virtual_state
        %b,%y = "atlas.virtual_dma_load_bf16"(%a,%p,%n) : (!atlas.virtual_state,i32,i32) -> (!atlas.virtual_state,!atlas.virtual_bf16)
        %c = "atlas.virtual_dma_store_bf16"(%b,%y,%p,%n) : (!atlas.virtual_state,!atlas.virtual_bf16,i32,i32) -> !atlas.virtual_state
        return %c : !atlas.virtual_state
      }
    }'''
    function = parse_mlir_text(text).body.block.first_op
    selected, = final_stores(function, 8192, 2048)
    assert selected.results[0] == function.body.block.last_op.arguments[0]
    # A final reader needs BF16, so removing/changing its preceding store is unsafe.
    reader = text.replace('return %c', '''%d,%z = "atlas.virtual_dma_load_bf16"(%c,%p,%n) : (!atlas.virtual_state,i32,i32) -> (!atlas.virtual_state,!atlas.virtual_bf16)
        return %d''')
    for changed in (reader, text.replace('constant 2048', 'constant 1024')):
        with pytest.raises(ValueError):
            final_stores(parse_mlir_text(changed).body.block.first_op, 8192, 2048)
    # An unconditional cycle cannot prove completion or a final publication.
    cyclic = text.replace('return %c : !atlas.virtual_state', 'cf.br ^again\n      ^again:\n        cf.br ^again')
    with pytest.raises(ValueError, match='cycle'):
        final_stores(parse_mlir_text(cyclic).body.block.first_op, 8192, 2048)
