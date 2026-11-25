"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
_TYPES = {'bf16_to_f32': ('bf16', 'f32', 2, 4, 'bf16-to-f32-exact-v1'), 'f32_to_bf16': ('f32', 'bf16', 4, 2, 'f32-to-bf16-rne-v1'), 'f32_to_f64': ('f32', 'f64', 4, 8, 'f32-to-f64-exact-v1'), 'f64_to_f32': ('f64', 'f32', 8, 4, 'f64-to-f32-rne-v1')}

def request_for(operation, shape=(2, 3)):
    source, destination, _, _, policy = _TYPES[operation]
    arg = SemanticNode('x', 'input', (), TensorType(shape, source, 'raw-bit-pattern'), effect='input')
    node = SemanticNode('out', operation, (arg.id,), TensorType(shape, destination, 'raw-bit-pattern'), (('numeric_semantics', policy),), index_maps=pointwise_identity_maps(len(shape), 1))
    return KernelRequest((arg, node), (node.id,), ('external',), ((arg.id, 'external'),), 'synthetic')
