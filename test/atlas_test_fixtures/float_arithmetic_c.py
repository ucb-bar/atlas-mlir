"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from dataclasses import replace
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

def request(shape, shared=False, dtype='f64'):
    type_ = TensorType(shape, dtype, 'raw-bit-pattern')
    a = SemanticNode('a', 'input', (), type_, effect='input')
    b = replace(a, id='b')
    node = SemanticNode('out', dtype + '_mul', ('a', 'a' if shared else 'b'), type_, (('numeric_semantics', f'ieee754-{dtype}-mul-rne-v1'),), index_maps=pointwise_identity_maps(len(shape), 2))
    return KernelRequest((a, node) if shared else (a, b, node), ('out',), ('external',), (('a', 'external'),) if shared else (('a', 'external'), ('b', 'external')), 'synthetic')
