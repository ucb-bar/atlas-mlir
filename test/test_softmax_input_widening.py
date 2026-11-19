"""Direct BF16 score transport preserves selected softmax arithmetic and guards."""
from dataclasses import replace

import pytest

from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.softmax_input_widening import prepare
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from test_pointwise_narrowing import adapter
from test_softmax_narrowing import case as narrowed_case


def case(shape, kind='softmax', output_dtype='bf16'):
    source = narrowed_case(shape, kind)
    if output_dtype == 'f32':
        source = replace(source, nodes=source.nodes[:-1], outputs=source.nodes[-1].inputs)
    scores = source.nodes[0]
    operand = SemanticNode('bf16_scores', 'input', (), replace(scores.type, dtype='bf16'), effect='input')
    widened = replace(scores, op='bf16_to_f32', inputs=(operand.id,), effect='pure',
        attrs=(('numeric_semantics', 'bf16-to-f32-exact-v1'),),
        index_maps=pointwise_identity_maps(len(shape), 1))
    return replace(source, nodes=(operand, widened, *source.nodes[1:]),
        input_storages=tuple((operand.id if key == scores.id else key, storage)
                             for key, storage in source.input_storages))


@pytest.mark.parametrize('shape', [(1, 7), (33, 65), (2, 3, 35)])
@pytest.mark.parametrize('kind', ['softmax', 'masked_softmax', 'guarded_softmax'])
@pytest.mark.parametrize('output_dtype', ['f32', 'bf16'])
def test_selector_binds_original_score_codes_and_output(shape, kind, output_dtype):
    source = case(shape, kind, output_dtype)
    base = adapter()
    group = prepare(source, base, kind, output_dtype=output_dtype)
    assert group.input_id == 'bf16_scores' and group.output_dtype == output_dtype
    assert group.precision.source_region.scores == 'scores'
    selected, = AdjacentDeviceAdapter(base, patterns=('softmax_narrowing', 'softmax_input_widening')).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'softmax_input_widening'
    assert len(selected.nodes) == len([node for node in source.nodes if node.effect == 'pure'])


def test_refuses_escaping_widening_and_changed_cast_semantics():
    source = case((2, 35))
    base = adapter()
    for changed in (replace(source, outputs=('scores', *source.outputs), output_storages=('external', 'external')),
                    replace(source, nodes=(source.nodes[0], replace(source.nodes[1], attrs=()), *source.nodes[2:]))):
        with pytest.raises(ValueError):
            prepare(changed, base, 'softmax', output_dtype='bf16')
