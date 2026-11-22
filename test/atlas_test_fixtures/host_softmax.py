"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from math import prod
from merlin.compiler.semantic.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

def make_request(shape, mask_kind='none', mask_shape=(), mask_map=None, fill_bits=None, keep_when=True):
    """Build real unresolved source IR for admission and qualification fixtures."""
    shape = tuple(shape)
    rank = len(shape)
    if rank < 2:
        raise ValueError('source recognizer requires rank >= 2; use (1, width) for rows')
    full = TensorType(shape, 'f32', 'raw-bit-pattern')
    row = TensorType(shape[:-1], 'f32', 'raw-bit-pattern')
    column = TensorType((*shape[:-1], 1), 'f32', 'raw-bit-pattern')
    identity = pointwise_identity_maps(rank, 1)[0]
    prefix = IndexMap(rank, identity.coefficients[:-1], (0,) * (rank - 1))
    reduce_maps = (identity, prefix, prefix)
    broadcast = (identity, IndexMap(rank, (*identity.coefficients[:-1], (0,) * rank), (0,) * rank), identity)
    nodes = [SemanticNode('scores', 'input', (), full, effect='input')]
    inputs = [('scores', 'external')]
    constants = [ConstantBinding('min_seed', 'f32-le-raw', '000080ff' * prod(shape[:-1])), ConstantBinding('zero_seed', 'f32-le-raw', '00000000' * prod(shape[:-1]))]
    source_id = 'scores'
    if mask_kind != 'none':
        mask_shape = tuple(mask_shape) if mask_shape else shape
        if mask_map is None:
            mask_map = IndexMap(rank, identity.coefficients[-len(mask_shape):], (0,) * len(mask_shape))
        mask_type = TensorType(mask_shape, 'i1', 'boolean') if mask_kind == 'select' else TensorType(mask_shape, 'f32', 'raw-bit-pattern')
        nodes.append(SemanticNode('mask', 'input', (), mask_type, effect='input'))
        inputs.append(('mask', 'external'))
        if mask_kind == 'additive':
            nodes.append(SemanticNode('masked', 'f32_add', ('scores', 'mask'), full, (('numeric_semantics', 'ieee754-f32-add-rne-v1'),), index_maps=(identity, mask_map, identity)))
        elif mask_kind == 'select':
            if type(fill_bits) is not int or not 0 <= fill_bits <= 4294967295:
                raise ValueError('select fixture requires uint32 fill_bits')
            nodes.append(SemanticNode('fill', 'constant', (), TensorType((), 'f32', 'raw-bit-pattern'), effect='constant'))
            constants.append(ConstantBinding('fill', 'f32-le-raw', fill_bits.to_bytes(4, 'little').hex()))
            scalar_map = IndexMap(rank, (), ())
            branches = ('scores', 'fill') if keep_when else ('fill', 'scores')
            maps = (identity, scalar_map) if keep_when else (scalar_map, identity)
            nodes.append(SemanticNode('masked', 'select', ('mask', *branches), full, index_maps=(mask_map, *maps, identity)))
        else:
            raise ValueError('unsupported fixture mask kind')
        source_id = 'masked'
    nodes.extend((SemanticNode('min_seed', 'constant', (), row, effect='constant'), SemanticNode('maximum', 'source_f32_reduce_max_unqualified', (source_id, 'min_seed'), row, (('axis', rank - 1), ('reduction_order', 'unresolved'), ('numeric_semantics', 'source-f32-maximum-unqualified-v1'), ('scalar_operand_order', 'input-accumulator')), index_maps=reduce_maps), SemanticNode('max_view', 'reshape', ('maximum',), column), SemanticNode('shifted', 'f32_sub', (source_id, 'max_view'), full, (('numeric_semantics', 'ieee754-f32-sub-rne-v1'),), index_maps=broadcast), SemanticNode('exponential', 'source_f32_exp_unqualified', ('shifted',), full, (('numeric_semantics', 'source-f32-exp-unqualified-v1'),), index_maps=pointwise_identity_maps(rank, 1)), SemanticNode('zero_seed', 'constant', (), row, effect='constant'), SemanticNode('total', 'source_f32_reduce_add_unqualified', ('exponential', 'zero_seed'), row, (('axis', rank - 1), ('reduction_order', 'unresolved'), ('numeric_semantics', 'source-f32-add-unqualified-v1'), ('scalar_add_order', 'input-accumulator')), index_maps=reduce_maps), SemanticNode('sum_view', 'reshape', ('total',), column), SemanticNode('result', 'f32_div', ('exponential', 'sum_view'), full, (('numeric_semantics', 'ieee754-f32-div-rne-v1'),), index_maps=broadcast)))
    return KernelRequest(tuple(nodes), ('result',), ('external',), tuple(inputs), 'host-softmax-test', constants=tuple(constants))
