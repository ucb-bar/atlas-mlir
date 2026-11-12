"""Publish a selected BF16 pointwise result through a private exact cast.

The selected pointwise operation and all input encoders stay unchanged. Only
the host publication omits its exact widening and the following device cast.
This is a target-representation proof, not reassociation of source FP32 math.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.model import pointwise_identity_maps


@dataclass(frozen=True)
class Narrowing:
    source: object
    precision: object


def prepare(source, adapter):
    from .pointwise import PointwisePrecision, prepare_source
    from .pointwise_math import MathPointwisePrecision

    pure = [node for node in source.nodes if node.effect == 'pure']
    if len(pure) != 2:
        raise ValueError('pointwise narrowing needs one computation and one cast')
    result, cast = pure
    if (result.type.dtype != 'f32' or cast.op != 'f32_to_bf16'
            or cast.inputs != (result.id,) or source.outputs != (cast.id,)
            or cast.type != replace(result.type, dtype='bf16')
            or dict(cast.attrs) != {'numeric_semantics': 'f32-to-bf16-rne-v1'}
            or cast.index_maps != pointwise_identity_maps(len(cast.type.shape), 1)):
        raise ValueError('pointwise narrowing needs a private result and exact dense cast')
    precision = prepare_source(extract_semantic_region(source, (result.id,)),
        allow_math=True, bf16_policy=adapter.bf16_pointwise,
        rne_add_policy=adapter.qualified_rne_add if result.op == 'f32_add' else None,
        exp_policy=adapter.source_exp_policy)
    # These bounded operators cannot produce NaN: sums/products have finite
    # normal-or-zero inputs of magnitude at most 2**63; LUT division/rsqrt also
    # require a positive denominator of magnitude at least 2**-63. Their selected
    # arithmetic produces BF16, and exact widening followed by RNE is identity
    # for every non-NaN BF16 encoding (including infinities and signed zero).
    if (type(precision) not in (PointwisePrecision, MathPointwisePrecision)
            or precision.request.nodes[-1].op not in {
                'mul_bf16_zero', 'add_bf16_rne', 'sub_bf16_rne',
                'divide_bf16_lut', 'rsqrt_bf16_lut'}):
        raise ValueError('pointwise narrowing has no non-NaN result proof for this policy')
    inputs = tuple(node for node in source.nodes if node.effect in ('input', 'constant'))
    original_inputs = tuple(node for node in precision.source.nodes if node.effect in ('input', 'constant'))
    if inputs != original_inputs:
        raise ValueError('pointwise narrowing changed the original operand boundary')
    return Narrowing(source, precision)


def alternatives(adapter, source, choices):
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

    from .pointwise import plan_abi

    choices = list(choices)
    singles = sorted((row for row in choices if len(row.nodes) == 1), key=lambda row: row.nodes)
    pure = [node for node in source.nodes if node.effect == 'pure']
    adapter._pointwise_narrowing_groups = {}
    for first, last in pairwise(singles):
        if (first.placement != 'accelerator' or last.placement != 'accelerator'
                or last.nodes[0] != first.nodes[0] + 1
                or dict(first.metadata).get('kind') not in ('pointwise', 'pointwise_math')
                or dict(last.metadata).get('kind') != 'exact_float_device'):
            continue
        try:
            region = extract_semantic_region(source, tuple(pure[row.nodes[0]].id for row in (first, last)))
            group = prepare(region, adapter)
            plan_abi(group.precision.request)
        except ValueError:
            continue
        ports = {port.buffer: port for row in (first, last) for port in (*row.inputs, *row.outputs)}
        inputs = tuple(node.id for node in region.nodes if node.effect in ('input', 'constant'))
        adapter._pointwise_narrowing_groups[region.digest()] = group
        choices.append(replace(first, id='pointwise_narrowing_' + first.id,
            nodes=(*first.nodes, *last.nodes), implementation='source_pointwise_narrowing',
            inputs=tuple(ports[key] for key in inputs), outputs=tuple(ports[key] for key in region.outputs),
            cycles=CycleInterval.unknown('unchanged pointwise program; exact cast omitted'),
            metadata=(('kind', 'pointwise_narrowing'),)))
    return tuple(choices)


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .pointwise import compile_pointwise, plan_abi

    request = group.precision.request
    abi = plan_abi(request)
    compile_pointwise(snapshot, request, fixed_inputs=abi['fixed_inputs'],
        fixed_outputs=tuple(abi['fixed_outputs']), target_source=target_source,
        destination=destination, limits=limits)


def export_group(group, compiled, name, *, target_source):
    from merlin.compile.semantic_application import CompiledCRegion

    from .c_runtime import RUNTIME_SHARED_FILES
    from .pointwise_c import prepare_precision_c_package

    compiled = Path(compiled)
    record, files = prepare_precision_c_package(group.precision, compiled, name,
        target_source=Path(target_source), output_dtype='bf16')
    record.update(kind='pointwise_narrowing', output_node_ids=list(group.source.outputs),
        source_request_digest=group.source.digest(),
        narrowing_proof='non-NaN BF16 result; exact widening followed by RNE is identity',
        context_type=name + '_context', source_context_type=name + '_source_context',
        binding_function=name + '_bind', source_binding_function=name + '_source_bind',
        launches_per_invocation=record.get('streaming', {}).get('launch_count', 1),
        native_manifest_sha256=sha256((compiled / 'manifest.json').read_bytes()).hexdigest())
    return CompiledCRegion(name + '_source_dispatch', name + '.h',
        tuple(record['input_node_ids']), group.source.outputs, files, RUNTIME_SHARED_FILES,
        ('segments/' + name,), record)
