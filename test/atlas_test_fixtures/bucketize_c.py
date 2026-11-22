"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from merlin.compiler.semantic.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType

def request_for(right, count=31, shape=(17, 31), *, limits=None):
    rank = len(shape)
    value = SemanticNode('values', 'input', (), TensorType(shape, 'f32', 'raw-bit-pattern'), effect='input')
    boundary = SemanticNode('limits', 'input' if limits is None else 'constant', (), TensorType((count,), 'f32', 'raw-bit-pattern'), effect='input' if limits is None else 'constant')
    projection = IndexMap(rank + 1, tuple((tuple((int(i == j) for j in range(rank + 1))) for i in range(rank))), (0,) * rank)
    node = SemanticNode('out', 'bucketize', (value.id, boundary.id), TensorType(shape, 'i64', 'signed-i64'), (('numeric_semantics', 'f32-unordered-boundary-count-i64-v1'), ('right', right), ('boundary_count', count)), index_maps=(projection, IndexMap(rank + 1, ((0,) * rank + (1,),), (0,)), projection))
    return KernelRequest((value, boundary, node), (node.id,), ('external',), tuple(((n.id, 'external') for n in (value, boundary) if n.effect == 'input')), 'synthetic', constants=() if limits is None else (ConstantBinding(boundary.id, 'f32-le-raw', limits.tobytes().hex()),))
