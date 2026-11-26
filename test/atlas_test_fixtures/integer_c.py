"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from dataclasses import replace
from merlin.compiler.semantic.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

def request_for(operation, shape=(2, 3, 5), axis=1, boolean=False):
    integer = TensorType(shape, 'i64', 'signed-i64')
    source = SemanticNode('x', 'input', (), TensorType(shape, 'i1', 'boolean') if boolean else integer, effect='input')
    operands = [source]
    output = integer
    if operation == 'i64_reduce_add':
        output = replace(integer, shape=shape[:axis] + shape[axis + 1:])
        operands.append(SemanticNode('seed', 'input', (), output, effect='input'))
        projection = IndexMap(len(shape), tuple((tuple((int(i == j) for j in range(len(shape)))) for i in range(len(shape)) if i != axis)), (0,) * (len(shape) - 1))
        maps = (pointwise_identity_maps(len(shape), 1)[0], projection, projection)
        attrs = {'axis': axis, 'minimum_index_bits': max(2, max(shape).bit_length() + 1), 'numeric_semantics': 'i64-add-modulo-2pow64-v1'}
    elif operation in {'i64_min_value', 'i64_first_min_index', 'i64_cumsum'}:
        if operation != 'i64_cumsum':
            output = replace(integer, shape=shape[:axis] + shape[axis + 1:])
        maps = ()
        attrs = {'axis': axis, 'minimum_index_bits': max(2, max(shape).bit_length() + 1), 'numeric_semantics': 'signed-i64-prefix-sum-wrap-v1' if operation == 'i64_cumsum' else 'signed-i64-min-first-index-v1'}
    else:
        if operation.startswith('signed_'):
            output = TensorType(shape, operation[-3:], 'raw-bit-pattern')
            policy = f'signed-i64-to-{output.dtype}-rne-v1'
        else:
            operands.append(SemanticNode('y', 'input', (), integer, effect='input'))
            policy = f'signed-i64-{operation[4:]}-wrap-v1'
        maps = pointwise_identity_maps(len(shape), len(operands))
        attrs = {'numeric_semantics': policy}
    node = SemanticNode('out', operation, tuple((n.id for n in operands)), output, tuple(sorted(attrs.items())), index_maps=maps)
    return KernelRequest((*operands, node), ('out',), ('external',), tuple(((n.id, 'external') for n in operands)), 'synthetic')
