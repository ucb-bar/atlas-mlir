"""Softmax publication keeps the selected arithmetic, guards and private output."""
from dataclasses import replace

import pytest

from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.softmax_narrowing import prepare
from merlin.semantic_compiler.model import IndexMap, SemanticNode, TensorType, pointwise_identity_maps
from merlin.tests.ir.test_guarded_softmax_source import guarded_graph
from merlin.tests.ir.test_softmax_source import source_graph
from test_pointwise_narrowing import adapter


def case(shape, kind='softmax'):
    source = guarded_graph(shape) if kind == 'guarded_softmax' else source_graph(shape)
    if kind == 'masked_softmax':
        scores = source.nodes[0]
        mask = SemanticNode('mask', 'input', (), TensorType((shape[-1],), 'f32', 'raw-bit-pattern'), effect='input')
        identity = pointwise_identity_maps(len(shape), 1)[0]
        masked = SemanticNode('masked', 'f32_add', (scores.id, mask.id), scores.type,
            (('numeric_semantics', 'ieee754-f32-add-rne-v1'),),
            index_maps=(identity, IndexMap(len(shape), (identity.coefficients[-1],), (0,)), identity))
        nodes = (scores, mask, masked, *(replace(node, inputs=tuple(masked.id if key == scores.id else key
                 for key in node.inputs)) for node in source.nodes[1:]))
        source = replace(source, nodes=nodes, input_storages=(*source.input_storages, ('mask', 'external')))
    result = source.node(source.outputs[0])
    cast = SemanticNode('narrowed', 'f32_to_bf16', (result.id,), replace(result.type, dtype='bf16'),
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),), index_maps=pointwise_identity_maps(len(shape), 1))
    return replace(source, nodes=(*source.nodes, cast), outputs=(cast.id,),
                   target_identity=TARGET_IDENTITY, lowering_policy='diagnostic')


@pytest.mark.parametrize('shape', [(1, 7), (33, 65), (2, 3, 35)])
@pytest.mark.parametrize('kind', ['softmax', 'masked_softmax', 'guarded_softmax'])
def test_selector_keeps_the_original_softmax_request(shape, kind):
    from atlas_mlir_native.softmax_precision import lower_semantic_softmax
    source = case(shape, kind)
    base = adapter()
    group = prepare(source, base, kind)
    expected = lower_semantic_softmax(group.producer, allow_masked=kind != 'softmax',
        allow_guarded=kind == 'guarded_softmax', score_domain=base.already_quantized_score_domain)
    assert group.precision == expected
    selected, = AdjacentDeviceAdapter(base, patterns=('softmax_narrowing',)).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'softmax_narrowing'
    assert len(base.plan(source).selected) == 2
    escaped = replace(source, outputs=(source.nodes[-1].inputs[0], source.outputs[0]),
                      output_storages=('external', 'external'))
    assert len(AdjacentDeviceAdapter(base, patterns=('softmax_narrowing',)).plan(escaped).selected) == 2


def test_changed_cast_and_unknown_domain_are_refused():
    source = case((2, 35))
    base = adapter()
    cast = source.nodes[-1]
    for changed in (replace(cast, attrs=()), replace(cast, index_maps=()),
                    replace(cast, inputs=('scores',))):
        with pytest.raises(ValueError):
            prepare(replace(source, nodes=(*source.nodes[:-1], changed)), base, 'softmax')
    base.already_quantized_score_domain = 'unknown'
    with pytest.raises(ValueError, match='bounded score policy'):
        prepare(source, base, 'softmax')
