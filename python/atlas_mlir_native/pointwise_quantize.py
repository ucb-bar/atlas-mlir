"""Pack private pointwise BF16 results before they leave the device.

Merlin owns region extraction and escaping-value checks. Atlas retains the
producer's selected numerical policy, rounding and input guards, then applies
its ordinary static raw PACK to the completed tile. Dynamic quantizers and
shared BF16 results retain their separate implementations.
"""
from dataclasses import dataclass, replace
import json
from pathlib import Path

from merlin.compiler.partition.semantic_regions import extract_semantic_region


@dataclass(frozen=True)
class QuantizedPointwise:
    source: object
    precision: object
    exponent: int
    input_ids: tuple[str, ...]
    input_dtype: str | None


def prepare(source, adapter, kind):
    from .static_quantize import prepare_source as quantizer_source
    from .pointwise import prepare_source, prepare_execution_plan, plan_abi
    from .pointwise_narrowing import prepare as narrowing
    from .pointwise_input_widening import prepare as widening

    pure = [node for node in source.nodes if node.effect == 'pure']
    if len(pure) < 3 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError('pointwise quantization requires a dense private producer')
    quantizer = extract_semantic_region(source, tuple(node.id for node in pure[-2:]))
    x, codes, exponent, fixed = quantizer_source(quantizer)
    producer = extract_semantic_region(source, tuple(node.id for node in pure[:-2]))
    if (x.type.dtype != 'bf16' or producer.outputs != (x.id,)
            or source.outputs != (codes.id, exponent.id)):
        raise ValueError('pointwise result must be private to both static quantizer projections')
    inputs = tuple(node.id for node in producer.nodes if node.effect in ('input', 'constant'))
    input_dtype = None
    if kind == 'pointwise':
        precision = prepare_source(producer, bf16_policy=adapter.bf16_pointwise)
    elif kind == 'pointwise_narrowing':
        precision = narrowing(producer, adapter).precision
    elif kind == 'pointwise_input_widening':
        previous = widening(producer, adapter, output_dtype='bf16')
        precision, inputs, input_dtype = previous.precision, (previous.input_id,), 'bf16'
    else:
        raise ValueError('unsupported pointwise quantization producer')
    if prepare_execution_plan(precision.request)[2]:
        raise ValueError('packed pointwise output requires a single resident frame')
    plan_abi(precision.request)
    return QuantizedPointwise(source, precision, fixed, inputs, input_dtype)


def alternatives(adapter, source, choices):
    from merlin.compiler.partition.global_plan import CycleInterval

    result, groups = list(choices), {}
    adapter._pointwise_quantize_groups = groups
    pure = [node for node in source.nodes if node.effect == 'pure']
    quantizers = {min(row.nodes): row for row in choices
                  if dict(row.metadata).get('kind') == 'activation_quantize_device'
                  or dict(row.metadata).get('kind') == 'static_quantize'
                  and adapter.activation_quantizer_route == 'device'}
    for first in choices:
        kind = dict(first.metadata).get('kind')
        if (first.placement != 'accelerator'
                or kind not in ('pointwise', 'pointwise_narrowing', 'pointwise_input_widening')):
            continue
        last = quantizers.get(max(first.nodes) + 1)
        if last is None:
            continue
        nodes = (*first.nodes, *last.nodes)
        try:
            region = extract_semantic_region(source, tuple(pure[index].id for index in nodes))
            group = prepare(region, adapter, kind)
        except ValueError:
            continue
        ports = {port.buffer: port for row in (first, last) for port in (*row.inputs, *row.outputs)}
        groups[region.digest()] = group
        result.append(replace(first, id='pointwise_quantize_' + first.id, nodes=nodes,
            implementation='source_pointwise_quantize', inputs=tuple(ports[key] for key in group.input_ids),
            outputs=tuple(ports[key] for key in region.outputs),
            cycles=CycleInterval.unknown('static PACK follows the unchanged BF16 pointwise result in VMEM'),
            metadata=(('kind', 'pointwise_quantize'),)))
    return tuple(result)


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .pointwise import compile_pointwise, plan_abi

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'source-request.json').write_text(json.dumps(group.source.record(), sort_keys=True) + '\n')
    request = group.precision.request
    abi = plan_abi(request)
    compile_pointwise(snapshot, request, fixed_inputs=abi['fixed_inputs'],
        fixed_outputs=tuple(abi['fixed_outputs']), target_source=target_source,
        destination=destination / 'pointwise', limits=limits, static_output_exponent=group.exponent)


def export_group(group, compiled, name, *, target_source):
    from merlin.compiler.build.regions import CompiledCRegion
    from .c_runtime import RUNTIME_SHARED_FILES
    from .pointwise_c import prepare_precision_c_package

    compiled = Path(compiled)
    if (compiled.is_symlink() or (compiled / 'source-request.json').is_symlink()
            or json.loads((compiled / 'source-request.json').read_text()) != group.source.record()):
        raise ValueError('quantized pointwise publication differs from its source')
    record, files = prepare_precision_c_package(group.precision, compiled / 'pointwise', name,
        target_source=Path(target_source), input_dtype=group.input_dtype, output_dtype='i8',
        static_output_exponent=group.exponent)
    record.update(kind='pointwise_quantize', input_node_ids=list(group.input_ids),
        output_node_ids=list(group.source.outputs), source_request_digest=group.source.digest(),
        quantization_proof='unchanged rounded BF16 result; ordinary static raw PACK; authored exponent',
        context_type=name + '_context', source_context_type=name + '_source_context',
        binding_function=name + '_bind', source_binding_function=name + '_source_bind',
        launches_per_invocation=1, static_exponent=group.exponent)
    return CompiledCRegion(name + '_source_dispatch', name + '.h', group.input_ids,
        group.source.outputs, files, RUNTIME_SHARED_FILES, ('segments/' + name + '/pointwise',), record)
