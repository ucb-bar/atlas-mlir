"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from merlin.compiler.semantic.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

def request_for(kind):
    truth = kind == 'nonzero'
    integer = kind in {'eq', 'ne', 'slt', 'sle', 'sgt', 'sge'}
    output = TensorType((3, 4), 'i64' if kind == 'extui' else 'i1', 'signed-i64' if kind == 'extui' else 'boolean')
    dtype, policy = ('f32', 'raw-bit-pattern') if truth else ('i64', 'signed-i64') if integer else ('i1', 'boolean')
    operands = tuple((SemanticNode(f'x{i}', 'input', (), TensorType(shape, dtype, policy), effect='input') for i, shape in enumerate(((3, 4),) if truth else ((3, 1),) if kind == 'extui' else ((3, 1), (1, 4)))))
    maps = pointwise_identity_maps(2, 1) if truth else (IndexMap(2, ((1, 0), (0, 0)), (0, 0)), *((IndexMap(2, ((0, 0), (0, 1)), (0, 0)),) if len(operands) == 2 else ()), pointwise_identity_maps(2, 1)[0])
    operation = 'arith.cmpi' if integer else 'arith.' + kind
    attrs = (('numeric_semantics', 'f32-une-zero-v1'),) if truth else (('operation', operation), ('predicate', kind if integer else None))
    node = SemanticNode('out', 'f32_nonzero' if truth else 'mask', tuple((n.id for n in operands)), output, attrs, index_maps=maps)
    return KernelRequest((*operands, node), ('out',), ('external',), tuple(((n.id, 'external') for n in operands)), 'synthetic')
