"""Folded input indexing preserves raw packing and refusals."""
from dataclasses import replace

import pytest
from atlas_mlir_native.already_quantized import prepare_source
from merlin.semantic_compiler.model import (
    IndexMap,
    SemanticNode,
    pointwise_identity_maps,
)
from test_already_quantized import source_request


def fixture(operand='w', m=3, n=35, k=33, batch=(2,)):
    source = source_request(m, n, k, batch)
    logical = source.node(operand)
    shape = logical.type.shape
    rank = len(shape)
    axes = (*range(rank - 2), rank - 1, rank - 2)
    physical = replace(logical, id=operand + '_physical', type=replace(logical.type,
                       shape=(*shape[:-2], shape[-1], shape[-2])))
    view = SemanticNode(operand, 'transpose', (physical.id,), logical.type,
        index_maps=(IndexMap(rank, tuple(tuple(int(j == axis) for j in range(rank)) for axis in axes),
                            (0,) * rank), pointwise_identity_maps(rank, 1)[0]))
    inputs = tuple(physical if node.id == operand else node for node in source.nodes if node.effect != 'pure')
    operations = tuple(node for node in source.nodes if node.effect == 'pure')
    folded = replace(source, nodes=(*inputs, view, *operations),
        input_storages=tuple((physical.id if key == operand else key, storage) for key, storage in source.input_storages))
    return source, folded


def test_source_intake_rejects_escaping_non_code_and_non_matrix_permutations():
    _, exponent_view = fixture('sw')
    with pytest.raises(ValueError, match='code operand'):
        prepare_source(exponent_view)
    _, source = fixture()
    with pytest.raises(ValueError, match='remain private'):
        prepare_source(replace(source, outputs=(*source.outputs, 'w'), output_storages=('external',) * 2))
    view = source.node('w')
    for changed in (replace(view, attrs=(('unknown', 'meaning'),)),
                    replace(view, op='gather', index_maps=(replace(view.index_maps[0], offsets=(1, 0)), view.index_maps[1]))):
        with pytest.raises(ValueError, match='batch axis'):
            prepare_source(replace(source, nodes=tuple(changed if n.id == 'w' else n for n in source.nodes)))
    # Moving a batch axis is a complete permutation but this target packer
    # deliberately admits only the last two matrix axes.
    _, source = fixture('a', m=2, k=2, batch=(2,))
    view = source.node('a')
    mapping = IndexMap(3, ((0, 1, 0), (1, 0, 0), (0, 0, 1)), (0, 0, 0))
    changed = replace(view, index_maps=(mapping, view.index_maps[1]))
    with pytest.raises(ValueError, match='batch axis'):
        prepare_source(replace(source, nodes=tuple(changed if n.id == 'a' else n for n in source.nodes)))
