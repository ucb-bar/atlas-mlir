"""Keep a selected BF16 SiLU expansion inside one native pointwise loop.

This recognizes a closed source graph, including its casts and literal one.
The target already implements division as reciprocal followed by multiplication;
no source IEEE division or unselected precision policy is reassociated here.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.model import (
    IndexMap,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)

from .binding import BF16_SIGMOID_DOMAIN, TARGET_IDENTITY
from .sigmoid import SILU_ATTRS, input_spec


@dataclass(frozen=True)
class ExpandedSilu:
    source: KernelRequest
    request: KernelRequest
    input_key: str
    proof: dict

    @property
    def operand_specs(self):
        return (input_spec(),)

    def record(self):
        return {'schema': 'atlas.expanded_silu.v1',
            'source_request_digest': self.source.digest(), 'request_digest': self.request.digest(),
            'inputs': [self.input_key], 'input_conversions': [input_spec().record()],
            'output_conversion': 'bf16-bit-preserving', 'proof': self.proof}


def prepare(source, adapter):
    from .adjacent_pointwise import intermediate_domain_proof
    from .exact_float_device import prepare_plan as prepare_exact
    from .pointwise import plan_abi, prepare_source
    from .source_exp import POLICY

    pure = [node for node in source.nodes if node.effect == 'pure']
    if (len(pure) != 7 or not adapter.pointwise or not adapter.pointwise_math
            or not adapter.sigmoid or adapter.source_exp_policy != POLICY
            or adapter.qualified_rne_add is not None):
        raise ValueError('expanded SiLU needs its complete seven-operation selected policy')
    widen, ones, negative, exponential, add, divide, narrow = pure
    if tuple(node.op for node in pure) != (
            'bf16_to_f32', 'gather', 'f32_neg', 'source_f32_exp_unqualified',
            'f32_add', 'f32_div', 'f32_to_bf16'):
        raise ValueError('expanded SiLU operation order differs')
    if (source.target_identity != TARGET_IDENTITY or source.lowering_policy != 'diagnostic'
            or source.physical_extents or source.boundary_layouts or source.output_layouts
            or source.outputs != (narrow.id,) or source.output_storages != ('external',)
            or len(widen.inputs) != 1 or len(ones.inputs) != 1
            or negative.inputs != (widen.id,) or exponential.inputs != (negative.id,)
            or add.inputs not in ((ones.id, exponential.id), (exponential.id, ones.id))
            or divide.inputs != (widen.id, add.id) or narrow.inputs != (divide.id,)):
        raise ValueError('expanded SiLU needs private intermediates and the original edges')
    operand, literal = source.node(widen.inputs[0]), source.node(ones.inputs[0])
    shape = narrow.type.shape
    rank = len(shape)
    bf16 = TensorType(shape, 'bf16', 'raw-bit-pattern')
    f32 = replace(bf16, dtype='f32')
    if (len(source.nodes) != 9 or operand.type != bf16 or narrow.type != bf16
            or operand.op != 'input' or operand.effect != 'input'
            or operand.inputs or operand.attrs or operand.index_maps
            or source.input_storages != ((operand.id, 'external'),)
            or any(node.type != f32 for node in pure[:-1])
            or literal.op != 'constant' or literal.effect != 'constant'
            or literal.type != TensorType((), 'f32', 'raw-bit-pattern')
            or literal.inputs or literal.attrs or literal.index_maps or ones.attrs
            or ones.index_maps != (IndexMap(rank, (), ()), pointwise_identity_maps(rank, 1)[-1])
            or len(source.constants) != 1 or source.constants[0].node_id != literal.id
            or source.constants[0].encoding != 'f32-le-raw'
            or bytes.fromhex(source.constants[0].data_hex) != b'\x00\x00\x80\x3f'):
        raise ValueError('expanded SiLU requires dense equal shapes and the bound scalar one')
    # Reuse the ordinary recognizers for every cast, sign flip and arithmetic
    # operation. They reject changed rounding, maps, types and source attributes.
    for node in (widen, negative, narrow):
        prepare_exact(extract_semantic_region(source, (node.id,)))
    precisions = tuple(prepare_source(extract_semantic_region(source, (node.id,)),
        allow_math=True, bf16_policy=adapter.bf16_pointwise,
        exp_policy=adapter.source_exp_policy) for node in (exponential, add, divide))
    if (tuple(p.request.nodes[-1].op for p in precisions)
            != ('exp_bf16', 'add_bf16_rne', 'divide_bf16_lut')
            or any(node.index_maps != pointwise_identity_maps(rank, len(node.inputs))
                   for node in (exponential, add, divide))):
        raise ValueError('expanded SiLU selected arithmetic or dense maps differ')
    arg = SemanticNode('arg0', 'input', (), bf16, (BF16_SIGMOID_DOMAIN,), 'input')
    result = SemanticNode('result', 'silu_bf16_lut', (arg.id,), bf16, SILU_ATTRS,
        index_maps=pointwise_identity_maps(rank, 1))
    request = KernelRequest((arg, result), (result.id,), ('external',),
        ((arg.id, 'external'),), TARGET_IDENTITY, 'diagnostic', source.digest())
    plan_abi(request)
    proof = {'intermediate_domains': intermediate_domain_proof(),
        'input_domain': 'intersection of original EXP(-x) and division-numerator encoders',
        'signed_zero': 'zero subtraction and sign flip differ only at zero; EXP(+0)=EXP(-0)=1; final product retains x sign',
        'arithmetic': 'same EXP LUT, FP32-RNE add then BF16 chop, reciprocal LUT, BF16-RNE product',
        'casts': 'BF16 widening/reencoding is identity; final finite BF16 widening/narrowing is identity',
        'literal': 'bound scalar FP32 one is embedded in the native body',
        'rejection': 'validate every public input before launch; all removed internal encoders are proved admitted'}
    return ExpandedSilu(source, request, operand.id, proof)


def alternatives(adapter, source, choices):
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

    choices = list(choices)
    singles = sorted((row for row in choices if len(row.nodes) == 1), key=lambda row: row.nodes)
    pure = [node for node in source.nodes if node.effect == 'pure']
    adapter._expanded_silu_groups = {}
    for start in range(len(singles) - 6):
        children = singles[start:start + 7]
        nodes = tuple(row.nodes[0] for row in children)
        if (any(b != a + 1 for a, b in pairwise(nodes))
                or any(row.placement != 'accelerator' for row in children)):
            continue
        try:
            region = extract_semantic_region(source, tuple(pure[index].id for index in nodes))
            group = prepare(region, adapter)
        except ValueError:
            continue
        ports = {port.buffer: port for row in children for port in (*row.inputs, *row.outputs)}
        inputs = tuple(node.id for node in region.nodes if node.effect in ('input', 'constant'))
        adapter._expanded_silu_groups[region.digest()] = group
        choices.append(replace(children[0], id='expanded_silu_' + children[0].id, nodes=nodes,
            implementation='source_expanded_silu', inputs=tuple(ports[key] for key in inputs),
            outputs=tuple(ports[key] for key in region.outputs), demands=(), occupancy=(),
            cycles=CycleInterval.unknown('one selected SiLU loop with private casts removed'),
            metadata=(('kind', 'expanded_silu'),)))
    return tuple(choices)


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .pointwise import compile_pointwise, plan_abi

    abi = plan_abi(group.request)
    compile_pointwise(snapshot, group.request, fixed_inputs=abi['fixed_inputs'],
        fixed_outputs=tuple(abi['fixed_outputs']), target_source=target_source,
        destination=destination, limits=limits)


def export_group(group, compiled, name, *, target_source):
    from merlin.compile.semantic_application import CompiledCRegion

    from .c_runtime import RUNTIME_SHARED_FILES
    from .pointwise_c import prepare_precision_c_package

    compiled = Path(compiled)
    record, files = prepare_precision_c_package(group, compiled, name,
        target_source=Path(target_source), output_dtype='bf16')
    record.update(kind='expanded_silu', source_request_digest=group.source.digest(),
        context_type=name + '_context', source_context_type=name + '_source_context',
        binding_function=name + '_bind', source_binding_function=name + '_source_bind',
        launches_per_invocation=record.get('streaming', {}).get('launch_count', 1),
        native_manifest_sha256=sha256((compiled / 'manifest.json').read_bytes()).hexdigest())
    return CompiledCRegion(name + '_source_dispatch', name + '.h',
        tuple(record['input_node_ids']), group.source.outputs, files, RUNTIME_SHARED_FILES,
        ('segments/' + name,), record)
