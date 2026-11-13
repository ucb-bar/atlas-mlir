"""Consume original BF16 scores instead of materializing an exact f32 widening."""
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.compile.input_widening import split_private_bf16_widening
from merlin.compile.semantic_regions import extract_semantic_region

from .softmax_narrowing import KINDS


@dataclass(frozen=True)
class Widening:
    source: object
    producer: object
    precision: object
    kind: str
    input_id: str
    output_dtype: str
    score_domain: str
    finite_sentinel_bits: int | None


def prepare(source, adapter, kind, *, output_dtype='f32'):
    from .row_softmax import SOFTMAX_DOMAIN, ROWSPAN80_DOMAIN
    from .softmax_precision import lower_semantic_softmax
    from .softmax_narrowing import prepare as prepare_narrowing

    if kind not in KINDS or adapter.already_quantized_score_domain not in (SOFTMAX_DOMAIN, ROWSPAN80_DOMAIN):
        raise ValueError('softmax input widening needs a supported bounded score policy')
    split = split_private_bf16_widening(source)
    if output_dtype == 'bf16':
        narrowed = prepare_narrowing(split.consumer, adapter, kind)
        producer, precision = narrowed.producer, narrowed.precision
    elif output_dtype == 'f32':
        producer = split.consumer
        precision = lower_semantic_softmax(producer, allow_masked=kind != 'softmax',
            allow_guarded=kind == 'guarded_softmax', score_domain=adapter.already_quantized_score_domain)
    else:
        raise ValueError('unsupported softmax publication dtype')
    if precision.source_region.scores != split.widened_id:
        raise ValueError('the private widening must supply the softmax score boundary')
    # The original transport exactly widens BF16 then rounds it back to BF16
    # before checking these finite code ranges. Direct code validation is the
    # same operation, including signed zeros and rejection before device I/O.
    return Widening(source, producer, precision, kind, split.input_id, output_dtype,
        adapter.already_quantized_score_domain,
        adapter.already_quantized_mask_sentinel if kind == 'masked_softmax' else None)


def alternatives(adapter, source, choices):
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

    choices = list(choices)
    pure = [node for node in source.nodes if node.effect == 'pure']
    casts = {row.nodes[0]: row for row in choices if len(row.nodes) == 1
             and row.placement == 'accelerator' and dict(row.metadata).get('kind') == 'exact_float_device'}
    adapter._softmax_input_widening_groups = {}
    for last in tuple(choices):
        kind = dict(last.metadata).get('kind')
        if last.placement != 'accelerator' or kind not in (*KINDS, 'softmax_narrowing'):
            continue
        first = casts.get(min(last.nodes) - 1)
        if first is None:
            continue
        output_dtype = 'f32'
        if kind == 'softmax_narrowing':
            previous = extract_semantic_region(source, tuple(pure[index].id for index in last.nodes))
            kind = adapter._softmax_narrowing_groups[previous.digest()].kind
            output_dtype = 'bf16'
        try:
            region = extract_semantic_region(source, tuple(pure[index].id for index in (*first.nodes, *last.nodes)))
            group = prepare(region, adapter, kind, output_dtype=output_dtype)
        except ValueError:
            continue
        ports = {port.buffer: port for row in (first, last) for port in (*row.inputs, *row.outputs)}
        inputs = tuple(node.id for node in region.nodes if node.effect in ('input', 'constant'))
        adapter._softmax_input_widening_groups[region.digest()] = group
        choices.append(replace(last, id='softmax_input_widening_' + last.id,
            nodes=(*first.nodes, *last.nodes), implementation='source_softmax_input_widening',
            inputs=tuple(ports[key] for key in inputs), outputs=tuple(ports[key] for key in region.outputs),
            cycles=CycleInterval.unknown('unchanged row-softmax program; private exact widening omitted'),
            metadata=(('kind', 'softmax_input_widening'),)))
    return tuple(choices)


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .row_loop import compile_row_loop

    compile_row_loop(snapshot, group.precision.request, target_source=target_source,
                     destination=destination, limits=limits)


def export_group(group, compiled, name, *, target_source):
    from merlin.compile.semantic_application import CompiledCRegion
    from .c_runtime import RUNTIME_SHARED_FILES
    from .softmax_c import _prepare_c_package

    compiled = Path(compiled)
    record, files = _prepare_c_package(group.precision.request, compiled, name,
        target_source=Path(target_source), runtime_linkage='separate',
        semantic_source=group.producer, row_loop_compilation=compiled,
        allow_masked=group.kind != 'softmax', allow_guarded=group.kind == 'guarded_softmax',
        finite_sentinel_bits=group.finite_sentinel_bits, score_domain=group.score_domain,
        input_dtype='bf16', output_dtype=group.output_dtype)
    region = group.precision.source_region
    inputs = (group.input_id, region.mask) if group.kind != 'softmax' else (group.input_id,)
    description = dict(kind='softmax_input_widening', context_type=name + '_context',
        source_context_type=name + '_source_context', binding_function=name + '_bind',
        source_binding_function=name + '_source_bind', workspace_bytes=record['workspace_bytes'],
        source_workspace_bytes=record['source_interface']['workspace_bytes'],
        precision_plan=group.precision.record(), native_export=record,
        source_request_digest=group.source.digest(), launches_per_invocation=record['launches_per_invocation'],
        widening_proof='finite BF16 score codes; exact widening then RNE preserves every admitted code')
    return CompiledCRegion(name + '_source_dispatch', name + '.h', inputs,
        group.source.outputs, files, RUNTIME_SHARED_FILES, ('segments/' + name,), description)
