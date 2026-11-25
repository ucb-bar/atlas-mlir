"""Source builders retained locally for Atlas integration tests.

Adapted from the corresponding legacy Merlin tests; no test collectors.
"""
from merlin.runtime import host_math_supplier as host_math
from merlin.compiler.lower.c.unary_math import OPERATIONS
from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

def supplier(operation):
    return host_math.symbol(OPERATIONS[operation][1])

def policy(digest='0' * 64):
    return {'schema': 'merlin.compiled_unary_math.v1', 'policy': 'selected-host-math-rne-finite-v1', 'isa': 'rv64gc', 'abi': 'lp64d', 'effects': 'enabled-fpu-rne-unobserved-flags-and-errno-v1', 'operations': sorted(OPERATIONS), 'linkage': {'schema': 'merlin.host_linkage_contract.v1', 'supplier': 'selected_math_archive', 'archive_sha256': digest, 'symbols': sorted(['merlin_host_check_fp_env', *map(supplier, OPERATIONS)])}}

def request(operation, shape=(2, 3)):
    dtype, symbol = OPERATIONS[operation]
    type_ = TensorType(shape, dtype, 'raw-bit-pattern')
    arg = SemanticNode('arg', 'input', (), type_, effect='input')
    out = SemanticNode('out', operation, (arg.id,), type_, (('numeric_semantics', f"source-{dtype}-{symbol.rstrip('f')}-unqualified-v1"),), index_maps=pointwise_identity_maps(len(shape), 1))
    return KernelRequest((arg, out), (out.id,), ('external',), ((arg.id, 'external'),), 'synthetic')
