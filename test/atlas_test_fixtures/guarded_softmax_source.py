"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from dataclasses import replace
from math import prod
from merlin.compiler.semantic.model import ConstantBinding, IndexMap, SemanticNode, TensorType, pointwise_identity_maps
from atlas_test_fixtures.softmax_source import source_graph

def guarded_graph(shape=(2, 3, 5), *, shared_maximum=False):
    source = source_graph(shape)
    rank = len(shape)
    full = source.node('scores').type
    column = TensorType((*shape[:-1], 1), 'f32', 'raw-bit-pattern')
    identity = pointwise_identity_maps(rank, 1)[0]
    scalar_map = IndexMap(rank, (), ())
    row_map = IndexMap(rank, (*identity.coefficients[:-1], (0,) * rank), (0,) * rank)
    mask = SemanticNode('excluded', 'input', (), TensorType(shape[-2:], 'i1', 'boolean'), effect='input')
    negative = SemanticNode('negative_inf', 'constant', (), TensorType((), 'f32', 'raw-bit-pattern'), effect='constant')
    zero = replace(negative, id='zero')
    guard_limit = replace(negative, id='guard_limit', type=column)
    mask_view = SemanticNode('mask_view', 'gather', (mask.id,), TensorType(shape, 'i1', 'boolean'), index_maps=(IndexMap(rank, identity.coefficients[-2:], (0, 0)), identity))
    fill = SemanticNode('fill', 'gather', (negative.id,), full, index_maps=(scalar_map, identity))
    masked = SemanticNode('masked', 'select', (mask_view.id, fill.id, 'scores'), full, index_maps=(identity,) * 4)
    core = tuple((replace(n, inputs=tuple(('masked' if k == 'scores' else k for k in n.inputs))) for n in source.nodes[1:]))
    maximum = next((n for n in core if n.id == 'maximum'))
    duplicate = replace(maximum, id='guard_maximum')
    parent = maximum.id if shared_maximum else duplicate.id
    view = SemanticNode('guard_view', 'reshape', (parent,), column)
    compare = SemanticNode('guard_equal', 'mask', (view.id, guard_limit.id), TensorType(column.shape, 'i1', 'boolean'), (('operation', 'arith.cmpf'), ('predicate', 'oeq')), index_maps=pointwise_identity_maps(rank, 2))
    condition = SemanticNode('guard_condition', 'gather', (compare.id,), TensorType(shape, 'i1', 'boolean'), index_maps=(row_map, identity))
    zero_view = SemanticNode('zero_view', 'gather', (zero.id,), full, index_maps=(scalar_map, identity))
    result = SemanticNode('guarded', 'select', (condition.id, zero_view.id, 'result'), full, index_maps=(identity,) * 4)
    return replace(source, nodes=(source.nodes[0], mask, negative, mask_view, fill, masked, *core, zero, guard_limit, *((duplicate,) if not shared_maximum else ()), view, compare, condition, zero_view, result), outputs=(result.id,), input_storages=(*source.input_storages, (mask.id, 'external')), constants=(*source.constants, ConstantBinding(negative.id, 'f32-le-raw', '000080ff'), ConstantBinding(zero.id, 'f32-le-raw', '00000000'), ConstantBinding(guard_limit.id, 'f32-le-raw', '000080ff' * prod(shape[:-1]))))
