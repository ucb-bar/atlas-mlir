"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from dataclasses import replace
from merlin.compiler.semantic.model import TensorType
from atlas_test_fixtures.float_arithmetic_c import request as f64_request

def request(shape, operation, shared=False):
    source = f64_request(shape, shared)
    return replace(source, nodes=tuple((replace(node, type=TensorType(shape, 'bf16', 'raw-bit-pattern'), **{'op': operation, 'attrs': (('numeric_semantics', f'ieee754-bf16-{operation[5:]}-rne-v1'),)} if node.effect == 'pure' else {}) for node in source.nodes)))
