"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from dataclasses import replace
from math import prod
from merlin.compiler.semantic.model import ConstantBinding, IndexMap, SemanticNode, TensorType, pointwise_identity_maps
from atlas_test_fixtures.softmax_source import source_graph

def graph(shape, channels=64, views=False):
    source = source_graph(shape)
    batch = shape[:-2]
    matrix_shape = (*batch, shape[-1], channels)
    output_type = TensorType((*batch, shape[-2], channels), 'f32', 'raw-bit-pattern')
    value = SemanticNode('values', 'input', (), TensorType(matrix_shape, 'f32', 'raw-bit-pattern'), effect='input')
    seed = SemanticNode('value_seed', 'constant', (), output_type, effect='constant')
    lhs, extra = ('result', [])
    if views:
        extra.append(SemanticNode('copy_probability', 'gather', (lhs,), source.node(lhs).type, index_maps=pointwise_identity_maps(len(shape), 1)))
        extra.append(SemanticNode('flat_probability', 'reshape', ('copy_probability',), TensorType((prod(shape),), 'f32', 'raw-bit-pattern')))
        extra.append(SemanticNode('matrix_probability', 'reshape', ('flat_probability',), source.node(lhs).type))
        lhs = 'matrix_probability'
    rank = len(shape)
    axes = tuple((tuple((int(i == j) for j in range(rank + 1))) for i in range(rank + 1)))
    maps = tuple((IndexMap(rank + 1, (*axes[:rank - 2], *pair), (0,) * rank) for pair in ((axes[-3], axes[-1]), (axes[-1], axes[-2]), (axes[-3], axes[-2]))))
    result = SemanticNode('context', 'source_f32_matmul_unqualified', (lhs, 'values', 'value_seed'), output_type, (('reduction_order', 'unresolved'), ('numeric_semantics', 'source-f32-mul-add-unqualified-v1')), index_maps=(*maps, maps[-1]))
    return replace(source, nodes=(*source.nodes, value, seed, *extra, result), outputs=(result.id,), input_storages=(*source.input_storages, (value.id, 'external')), constants=(*source.constants, ConstantBinding(seed.id, 'f32-le-raw', '00000000' * prod(output_type.shape))))
