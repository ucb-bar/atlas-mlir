"""Private activation casts preserve selected arithmetic, ownership and guards."""
from dataclasses import replace

import pytest

from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.pointwise_narrowing import prepare
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from test_gelu_precision import source_request
from test_pointwise_narrowing import adapter


def case(shape, kind='gelu', *, input_dtype='f32', output_dtype='bf16', reverse=False):
    source = source_request(shape)
    if kind in ('sigmoid', 'silu'):
        from atlas_mlir_native.sigmoid import SOURCE_ATTRS
        source = replace(source, nodes=(source.nodes[0], replace(source.nodes[1], attrs=SOURCE_ATTRS)))
    if kind == 'silu':
        x, sigmoid = source.nodes
        product = SemanticNode('product', 'f32_mul', (sigmoid.id, x.id) if reverse else (x.id, sigmoid.id),
            x.type, (('numeric_semantics', 'ieee754-f32-mul-rne-v1'),),
            index_maps=pointwise_identity_maps(len(shape), 2))
        source = replace(source, nodes=(*source.nodes, product), outputs=(product.id,))
    value = source.nodes[-1]
    cast = SemanticNode('narrowed', 'f32_to_bf16', (value.id,), replace(value.type, dtype='bf16'),
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),), index_maps=pointwise_identity_maps(len(shape), 1))
    if output_dtype == 'bf16':
        source = replace(source, nodes=(*source.nodes, cast), outputs=(cast.id,))
    if input_dtype == 'bf16':
        x = source.nodes[0]
        operand = replace(x, id='codes', type=replace(x.type, dtype='bf16'))
        widened = replace(x, op='bf16_to_f32', inputs=(operand.id,), effect='pure',
            attrs=(('numeric_semantics', 'bf16-to-f32-exact-v1'),),
            index_maps=pointwise_identity_maps(len(shape), 1))
        source = replace(source, nodes=(operand, widened, *source.nodes[1:]), input_storages=((operand.id, 'external'),))
    return source


@pytest.mark.parametrize('shape', [(1, 7), (33, 65), (2, 3, 35)])
def test_selection_keeps_original_gelu_and_requires_qualified_policy(shape):
    from atlas_mlir_native.gelu import prepare_source
    from merlin.compile.semantic_regions import extract_semantic_region
    source = case(shape)
    base = adapter()
    group = prepare(source, base)
    assert group.precision == prepare_source(extract_semantic_region(source, ('y',)))
    selected, = AdjacentDeviceAdapter(base, patterns=('pointwise_narrowing',)).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'pointwise_narrowing'
    base.gelu = False
    with pytest.raises(ValueError, match='non-NaN'):
        prepare(source, base)
    escaped = replace(source, outputs=('y', 'narrowed'), output_storages=('external',)*2)
    with pytest.raises(ValueError):
        prepare(escaped, adapter())
    changed = replace(source.nodes[1], attrs=tuple((key, value.replace('3d372713', '3d372714'))
                      for key, value in source.nodes[1].attrs))
    with pytest.raises(ValueError):
        prepare(replace(source, nodes=(source.nodes[0], changed, source.nodes[2])), adapter())


@pytest.mark.parametrize('kind', ['gelu', 'sigmoid', 'silu'])
@pytest.mark.parametrize('output_dtype', ['bf16', 'f32'])
def test_raw_input_selection_and_private_boundaries(kind, output_dtype):
    from atlas_mlir_native.pointwise_input_widening import prepare as prepare_inputs
    source = case((2,3,35), kind, input_dtype='bf16', output_dtype=output_dtype)
    group = prepare_inputs(source, adapter(), output_dtype=output_dtype)
    assert group.input_id == 'codes' and group.output_dtype == output_dtype
    selected, = AdjacentDeviceAdapter(adapter(), patterns=('pointwise_narrowing', 'pointwise_input_widening')).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'pointwise_input_widening'
    escaped = replace(source, outputs=('x', *source.outputs), output_storages=('external',)*2)
    with pytest.raises(ValueError):
        prepare_inputs(escaped, adapter(), output_dtype=output_dtype)


@pytest.mark.parametrize('reverse', [False, True])
def test_selected_silu_product_order_and_cast_ownership(reverse):
    source = case((33,35), 'silu', reverse=reverse)
    group = prepare(source, adapter())
    assert group.precision.source.nodes[-1].inputs == source.nodes[-2].inputs
    selected, = AdjacentDeviceAdapter(adapter(), patterns=('pointwise_narrowing',)).plan(source).selected
    assert len(selected.nodes) == 3
    base = adapter()
    base.sigmoid = False
    with pytest.raises(ValueError, match='non-NaN'):
        prepare(source, base)


@pytest.mark.parametrize('change', [
    {'max_normal_exponent_code': 200},
    {'flush_below_normal': True},
    {'canonicalize_negative_zero': True},
    {'pre_encode_scale_exponent': 1},
    {'column_scale_exponents': ((0, 35, 1),)},
])
def test_raw_input_refuses_encoders_that_change_values(monkeypatch, change):
    from atlas_mlir_native.gelu import GeluPrecision, input_spec
    from atlas_mlir_native.pointwise_input_widening import prepare as prepare_inputs

    spec = replace(input_spec(), **change)
    monkeypatch.setattr(GeluPrecision, 'operand_specs', property(lambda self: (spec,)))
    source = case((2, 3, 35), input_dtype='bf16')
    with pytest.raises(ValueError, match='clamping|numerical conversion'):
        prepare_inputs(source, adapter(), output_dtype='bf16')
