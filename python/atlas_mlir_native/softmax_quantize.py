"""Pack private BF16 softmax results in their producing row program.

Merlin extracts the dense region and its escaping values. Atlas proves the
selected softmax arithmetic and uses its ordinary exact static PACK lowering.
Dynamic scales and shared probability tensors retain the separate regions.
"""
from dataclasses import dataclass, replace
import json
from pathlib import Path

from merlin.compiler.partition.semantic_regions import extract_semantic_region


@dataclass(frozen=True)
class QuantizedSoftmax:
    source: object
    softmax: object
    quantizer: object
    exponent: int
    input_dtype: str


def prepare(source, adapter, kind, *, input_dtype='f32'):
    from .static_quantize import prepare_source
    from .softmax_narrowing import prepare as narrowing
    from .softmax_input_widening import prepare as widening

    pure = [node for node in source.nodes if node.effect == 'pure']
    if len(pure) < 3 or source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError('softmax quantization requires a dense private producer')
    quantizer = extract_semantic_region(source, tuple(node.id for node in pure[-2:]))
    x, codes, exponent, fixed = prepare_source(quantizer)
    producer = extract_semantic_region(source, tuple(node.id for node in pure[:-2]))
    if (x.type.dtype != 'bf16' or producer.outputs != (x.id,)
            or source.outputs != (codes.id, exponent.id)):
        raise ValueError('softmax probability must be private to both static quantizer projections')
    if input_dtype == 'bf16':
        softmax = widening(producer, adapter, kind, output_dtype='bf16')
    elif input_dtype == 'f32':
        softmax = narrowing(producer, adapter, kind)
    else:
        raise ValueError('unsupported softmax score transport')
    return QuantizedSoftmax(source, softmax, quantizer, fixed, input_dtype)


def alternatives(adapter, source, choices):
    from merlin.compiler.partition.global_plan import CycleInterval

    result, groups = list(choices), {}
    adapter._softmax_quantize_groups = groups
    pure = [node for node in source.nodes if node.effect == 'pure']
    quantizers = {min(row.nodes): row for row in choices
                  if dict(row.metadata).get('kind') == 'activation_quantize_device'
                  or dict(row.metadata).get('kind') == 'static_quantize'
                  and adapter.activation_quantizer_route == 'device'}
    for first in choices:
        flavor = dict(first.metadata).get('kind')
        if first.placement != 'accelerator' or flavor not in ('softmax_narrowing', 'softmax_input_widening'):
            continue
        last = quantizers.get(max(first.nodes) + 1)
        if last is None:
            continue
        previous = extract_semantic_region(source, tuple(pure[index].id for index in first.nodes))
        prior = getattr(adapter, '_' + flavor + '_groups')[previous.digest()]
        if flavor == 'softmax_input_widening' and prior.output_dtype != 'bf16':
            continue
        nodes = (*first.nodes, *last.nodes)
        try:
            region = extract_semantic_region(source, tuple(pure[index].id for index in nodes))
            group = prepare(region, adapter, prior.kind,
                            input_dtype='bf16' if flavor == 'softmax_input_widening' else 'f32')
        except ValueError:
            continue
        ports = {port.buffer: port for row in (first, last) for port in (*row.inputs, *row.outputs)}
        inputs = tuple(node.id for node in region.nodes if node.effect in ('input', 'constant'))
        groups[region.digest()] = group
        result.append(replace(first, id='softmax_quantize_' + first.id, nodes=nodes,
            implementation='source_softmax_quantize', inputs=tuple(ports[key] for key in inputs),
            outputs=tuple(ports[key] for key in region.outputs),
            cycles=CycleInterval.unknown('static PACK follows the unchanged BF16 softmax result in VMEM'),
            metadata=(('kind', 'softmax_quantize'),)))
    return tuple(result)


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .row_loop import compile_row_loop

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'source-request.json').write_text(json.dumps(group.source.record(), sort_keys=True) + '\n')
    compile_row_loop(snapshot, group.softmax.precision.request, target_source=target_source,
        destination=destination / 'row', limits=limits, static_output_exponent=group.exponent)


def export_group(group, compiled, name, *, target_source):
    from merlin.compiler.build.regions import CompiledCRegion

    from .c_runtime import RUNTIME_SHARED_FILES
    from .softmax_c import _prepare_c_package

    compiled = Path(compiled)
    if (compiled.is_symlink() or (compiled / 'source-request.json').is_symlink()
            or json.loads((compiled / 'source-request.json').read_text()) != group.source.record()):
        raise ValueError('quantized softmax publication differs from its source')
    softmax = group.softmax
    record, files = _prepare_c_package(softmax.precision.request, compiled / 'row', name,
        target_source=Path(target_source), runtime_linkage='separate',
        semantic_source=softmax.producer, row_loop_compilation=compiled / 'row',
        allow_masked=softmax.kind != 'softmax', allow_guarded=softmax.kind == 'guarded_softmax',
        finite_sentinel_bits=softmax.finite_sentinel_bits, score_domain=softmax.score_domain,
        input_dtype=group.input_dtype, output_dtype='i8', static_output_exponent=group.exponent)
    region = softmax.precision.source_region
    scores = softmax.input_id if group.input_dtype == 'bf16' else region.scores
    inputs = (scores, region.mask) if softmax.kind != 'softmax' else (scores,)
    description = dict(kind='softmax_quantize', context_type=name + '_context',
        source_context_type=name + '_source_context', binding_function=name + '_bind',
        source_binding_function=name + '_source_bind', workspace_bytes=record['workspace_bytes'],
        source_workspace_bytes=record['source_interface']['workspace_bytes'],
        precision_plan=softmax.precision.record(), native_export=record,
        source_request_digest=group.source.digest(), launches_per_invocation=record['launches_per_invocation'],
        static_exponent=group.exponent,
        quantization_proof='unchanged rounded BF16 probability; ordinary static raw PACK; authored exponent')
    return CompiledCRegion(name + '_source_dispatch', name + '.h', inputs,
        group.source.outputs, files, RUNTIME_SHARED_FILES, ('segments/' + name + '/row',), description)
