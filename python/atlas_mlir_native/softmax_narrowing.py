"""Publish selected BF16 softmax probabilities through a private exact cast."""
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.compile.result_narrowing import split_private_bf16_cast
from merlin.compile.semantic_regions import extract_semantic_region


KINDS = ('softmax', 'masked_softmax', 'guarded_softmax')


@dataclass(frozen=True)
class Narrowing:
    source: object
    producer: object
    precision: object
    kind: str
    score_domain: str
    finite_sentinel_bits: int | None


def prepare(source, adapter, kind):
    from .row_softmax import SOFTMAX_DOMAIN, ROWSPAN80_DOMAIN
    from .softmax_precision import lower_semantic_softmax

    if kind not in KINDS or adapter.already_quantized_score_domain not in (SOFTMAX_DOMAIN, ROWSPAN80_DOMAIN):
        raise ValueError('softmax narrowing needs a supported bounded score policy')
    producer = split_private_bf16_cast(source)
    precision = lower_semantic_softmax(producer, allow_masked=kind != 'softmax',
        allow_guarded=kind == 'guarded_softmax', score_domain=adapter.already_quantized_score_domain)
    # These policies admit finite scores and form exp(score - row_max), with a
    # positive finite sum in every nonempty row. The guarded policy publishes
    # zero for an empty row. Their selected BF16 result is therefore non-NaN;
    # exact widening followed by the source's RNE cast preserves every code.
    return Narrowing(source, producer, precision, kind, adapter.already_quantized_score_domain,
        adapter.already_quantized_mask_sentinel if kind == 'masked_softmax' else None)


def alternatives(adapter, source, choices):
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

    choices = list(choices)
    pure = [node for node in source.nodes if node.effect == 'pure']
    casts = {row.nodes[0]: row for row in choices if len(row.nodes) == 1
             and row.placement == 'accelerator' and dict(row.metadata).get('kind') == 'exact_float_device'}
    adapter._softmax_narrowing_groups = {}
    for first in tuple(choices):
        kind = dict(first.metadata).get('kind')
        if first.placement != 'accelerator' or kind not in KINDS:
            continue
        last = casts.get(max(first.nodes) + 1)
        if last is None:
            continue
        try:
            region = extract_semantic_region(source, tuple(pure[index].id for index in (*first.nodes, *last.nodes)))
            group = prepare(region, adapter, kind)
        except ValueError:
            continue
        ports = {port.buffer: port for row in (first, last) for port in (*row.inputs, *row.outputs)}
        inputs = tuple(node.id for node in region.nodes if node.effect in ('input', 'constant'))
        adapter._softmax_narrowing_groups[region.digest()] = group
        choices.append(replace(first, id='softmax_narrowing_' + first.id,
            nodes=(*first.nodes, *last.nodes), implementation='source_softmax_narrowing',
            inputs=tuple(ports[key] for key in inputs), outputs=tuple(ports[key] for key in region.outputs),
            cycles=CycleInterval.unknown('unchanged row-softmax program; private exact cast omitted'),
            metadata=(('kind', 'softmax_narrowing'),)))
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
        output_dtype='bf16')
    region = group.precision.source_region
    inputs = (region.scores, region.mask) if group.kind != 'softmax' else (region.scores,)
    description = dict(kind='softmax_narrowing', context_type=name + '_context',
        source_context_type=name + '_source_context', binding_function=name + '_bind',
        source_binding_function=name + '_source_bind', workspace_bytes=record['workspace_bytes'],
        source_workspace_bytes=record['source_interface']['workspace_bytes'],
        precision_plan=group.precision.record(), native_export=record,
        source_request_digest=group.source.digest(), launches_per_invocation=record['launches_per_invocation'],
        narrowing_proof='non-NaN BF16 probabilities; exact widening followed by RNE is identity')
    return CompiledCRegion(name + '_source_dispatch', name + '.h', inputs,
        group.source.outputs, files, RUNTIME_SHARED_FILES, ('segments/' + name,), description)
