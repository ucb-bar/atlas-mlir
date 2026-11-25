"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from math import prod
from merlin.compiler.semantic.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType

def source_graph(batch):
    rank = len(batch) + 3
    axes = tuple((tuple((int(i == j) for j in range(rank))) for i in range(rank)))
    prefix = axes[:-3]
    maps = tuple((IndexMap(rank, (*prefix, *rows), (0,) * (rank - 1)) for rows in ((axes[-3], axes[-1]), (axes[-1], axes[-2]), (axes[-3], axes[-2]))))
    types = [TensorType((*batch, *shape), 'f32', 'raw-bit-pattern') for shape in ((2, 3), (3, 4), (2, 4))]
    nodes = (SemanticNode('a', 'input', (), types[0], effect='input'), SemanticNode('b', 'input', (), types[1], effect='input'), SemanticNode('seed', 'constant', (), types[2], effect='constant'), SemanticNode('y', 'source_f32_matmul_unqualified', ('a', 'b', 'seed'), types[2], (('numeric_semantics', 'source-f32-mul-add-unqualified-v1'), ('reduction_order', 'unresolved'), ('scalar_add_order', 'accumulator-product')), index_maps=(*maps, maps[-1])))
    return KernelRequest(nodes, ('y',), ('external',), (('a', 'external'), ('b', 'external')), 'synthetic', 'diagnostic', constants=(ConstantBinding('seed', 'f32-le-raw', '00000000' * (prod(batch) * 8)),))
