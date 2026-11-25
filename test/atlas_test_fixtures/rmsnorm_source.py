"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from math import prod
import struct
from merlin.compiler.semantic.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

def source_graph(shape=(1, 2048), *, reciprocal_mean=False, residual=True):
    rank = len(shape)
    full, rows, column, gamma_type = [TensorType(s, 'f32', 'raw-bit-pattern') for s in (shape, shape[:-1], (*shape[:-1], 1), (shape[-1],))]
    identity = pointwise_identity_maps(rank, 1)[0]
    prefix = IndexMap(rank, identity.coefficients[:-1], (0,) * (rank - 1))
    scalar = IndexMap(rank, (), ())
    broadcast = IndexMap(rank, (*identity.coefficients[:-1], (0,) * rank), (0,) * rank)
    gamma_map = IndexMap(rank, (identity.coefficients[-1],), (0,))
    constant_type = TensorType((), 'f32', 'raw-bit-pattern')
    attrs = lambda kind: (('numeric_semantics', f'ieee754-f32-{kind}-rne-v1'),)
    mean_kind = 'mul' if reciprocal_mean else 'div'
    nodes = [SemanticNode('H', 'input', (), full, effect='input'), SemanticNode('gamma', 'input', (), gamma_type, effect='input'), SemanticNode('seed', 'constant', (), rows, effect='constant'), SemanticNode('factor', 'constant', (), constant_type, effect='constant'), SemanticNode('eps', 'constant', (), constant_type, effect='constant'), SemanticNode('square', 'f32_mul', ('H', 'H'), full, attrs('mul'), index_maps=pointwise_identity_maps(rank, 2)), SemanticNode('sum', 'source_f32_reduce_add_unqualified', ('square', 'seed'), rows, (('axis', rank - 1), ('reduction_order', 'unresolved'), ('numeric_semantics', 'source-f32-add-unqualified-v1'), ('scalar_add_order', 'input-accumulator')), index_maps=(identity, prefix, prefix)), SemanticNode('sum_view', 'reshape', ('sum',), column), SemanticNode('mean', 'f32_' + mean_kind, ('sum_view', 'factor'), column, attrs(mean_kind), index_maps=(identity, scalar, identity)), SemanticNode('shifted', 'f32_add', ('mean', 'eps'), column, attrs('add'), index_maps=(identity, scalar, identity)), SemanticNode('inverse', 'source_f32_rsqrt_unqualified', ('shifted',), column, (('numeric_semantics', 'source-f32-rsqrt-unqualified-v1'),), index_maps=pointwise_identity_maps(rank, 1)), SemanticNode('normalized', 'f32_mul', ('H', 'inverse'), full, attrs('mul'), index_maps=(identity, broadcast, identity)), SemanticNode('result', 'f32_mul', ('gamma', 'normalized'), full, attrs('mul'), index_maps=(gamma_map, identity, identity))]
    if residual:
        nodes.append(SemanticNode('residual', 'f32_add', ('H', 'H'), full, attrs('add'), index_maps=pointwise_identity_maps(rank, 2)))
    return KernelRequest(tuple(nodes), ('result',), ('external',), (('H', 'external'), ('gamma', 'external')), 'synthetic', 'diagnostic', constants=(ConstantBinding('seed', 'f32-le-raw', '00000000' * prod(shape[:-1])), ConstantBinding('factor', 'f32-le-raw', struct.pack('<f', 1 / shape[-1] if reciprocal_mean else shape[-1]).hex()), ConstantBinding('eps', 'f32-le-raw', 'acc52737')))
