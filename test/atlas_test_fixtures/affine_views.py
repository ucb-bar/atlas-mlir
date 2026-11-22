"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from dataclasses import replace
from merlin.compiler.semantic.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

def source(operation='mul', *, transpose=False, shape=(3, 7)):
    small = (shape[0],) if transpose else (shape[1],)
    bf16 = TensorType(shape, 'bf16', 'raw-bit-pattern')
    f32 = replace(bf16, dtype='f32')
    weight = SemanticNode('weight', 'input', (), replace(bf16, shape=small), effect='input')
    x = SemanticNode('x', 'input', (), f32, effect='input')
    read = IndexMap(2, ((1, 0),) if transpose else ((0, 1),), (0,))
    view = SemanticNode('view', 'gather', (weight.id,), bf16, index_maps=(read, pointwise_identity_maps(2, 1)[-1]))
    cast = SemanticNode('cast', 'bf16_to_f32', (view.id,), f32, (('numeric_semantics', 'bf16-to-f32-exact-v1'),), index_maps=pointwise_identity_maps(2, 1))
    out = SemanticNode('out', 'f32_' + operation, (cast.id, x.id), f32, (('numeric_semantics', 'ieee754-f32-' + operation + '-rne-v1'),), index_maps=pointwise_identity_maps(2, 2))
    return KernelRequest((weight, x, view, cast, out), (out.id,), ('external',), ((weight.id, 'external'), (x.id, 'external')), 'synthetic')
