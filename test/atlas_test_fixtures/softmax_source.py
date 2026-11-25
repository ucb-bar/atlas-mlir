"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from math import prod
from merlin.compiler.semantic.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

def source_graph(shape):
    rank = len(shape)
    full, row, column = (TensorType(extent, 'f32', 'raw-bit-pattern') for extent in (shape, shape[:-1], (*shape[:-1], 1)))
    identity = pointwise_identity_maps(rank, 1)[0]
    prefix = IndexMap(rank, identity.coefficients[:-1], (0,) * (rank - 1))
    reduce_maps = (identity, prefix, prefix)
    broadcast = (identity, IndexMap(rank, (*identity.coefficients[:-1], (0,) * rank), (0,) * rank), identity)
    max_attrs = (('axis', rank - 1), ('reduction_order', 'unresolved'), ('numeric_semantics', 'source-f32-maximum-unqualified-v1'), ('scalar_operand_order', 'input-accumulator'))
    sum_attrs = (('axis', rank - 1), ('reduction_order', 'unresolved'), ('numeric_semantics', 'source-f32-add-unqualified-v1'), ('scalar_add_order', 'input-accumulator'))
    nodes = (SemanticNode('scores', 'input', (), full, effect='input'), SemanticNode('min_seed', 'constant', (), row, effect='constant'), SemanticNode('maximum', 'source_f32_reduce_max_unqualified', ('scores', 'min_seed'), row, max_attrs, index_maps=reduce_maps), SemanticNode('max_view', 'reshape', ('maximum',), column), SemanticNode('shifted', 'f32_sub', ('scores', 'max_view'), full, (('numeric_semantics', 'ieee754-f32-sub-rne-v1'),), index_maps=broadcast), SemanticNode('exponential', 'source_f32_exp_unqualified', ('shifted',), full, (('numeric_semantics', 'source-f32-exp-unqualified-v1'),), index_maps=pointwise_identity_maps(rank, 1)), SemanticNode('zero_seed', 'constant', (), row, effect='constant'), SemanticNode('total', 'source_f32_reduce_add_unqualified', ('exponential', 'zero_seed'), row, sum_attrs, index_maps=reduce_maps), SemanticNode('sum_view', 'reshape', ('total',), column), SemanticNode('result', 'f32_div', ('exponential', 'sum_view'), full, (('numeric_semantics', 'ieee754-f32-div-rne-v1'),), index_maps=broadcast))
    return KernelRequest(nodes, ('result',), ('external',), (('scores', 'external'),), 'synthetic', constants=(ConstantBinding('min_seed', 'f32-le-raw', '000080ff' * prod(shape[:-1])), ConstantBinding('zero_seed', 'f32-le-raw', '00000000' * prod(shape[:-1]))))
