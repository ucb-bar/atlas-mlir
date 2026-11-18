"""Publish selected BF16 softmax probabilities through a private exact cast."""
from dataclasses import dataclass
from pathlib import Path

from merlin.compile.result_narrowing import split_private_bf16_cast

from .composition_adapter import c_region, device_casts, fuse, kind_of, neighbor_pairs, source_names


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
    return fuse(adapter, source, choices, '_softmax_narrowing_groups', 'softmax_narrowing',
                neighbor_pairs(choices, device_casts(choices), lambda row: kind_of(row) in KINDS),
                prepare=lambda region, rows: prepare(region, adapter, kind_of(rows[0])),
                cycles='unchanged row-softmax program; private exact cast omitted')


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .row_loop import compile_row_loop

    compile_row_loop(snapshot, group.precision.request, target_source=target_source,
                     destination=destination, limits=limits)


def export_softmax(softmax, compiled, name, *, target_source, outputs, programs, scores=None, input_dtype='f32',
                   output_dtype, static_output_exponent=None, **fields):
    """Export a compiled row softmax as one source-dispatch C region described by `fields`."""
    from .softmax_c import _prepare_c_package

    compiled = Path(compiled)
    record, files = _prepare_c_package(softmax.precision.request, compiled, name,
        target_source=Path(target_source), runtime_linkage='separate',
        semantic_source=softmax.producer, row_loop_compilation=compiled,
        allow_masked=softmax.kind != 'softmax', allow_guarded=softmax.kind == 'guarded_softmax',
        finite_sentinel_bits=softmax.finite_sentinel_bits, score_domain=softmax.score_domain,
        input_dtype=input_dtype, output_dtype=output_dtype, static_output_exponent=static_output_exponent)
    region = softmax.precision.source_region
    scores = region.scores if scores is None else scores
    description = dict(fields, **source_names(name), workspace_bytes=record['workspace_bytes'],
        source_workspace_bytes=record['source_interface']['workspace_bytes'],
        precision_plan=softmax.precision.record(), native_export=record,
        launches_per_invocation=record['launches_per_invocation'])
    return c_region(name, (scores, region.mask) if softmax.kind != 'softmax' else (scores,), outputs, files, programs,
                    description)


def export_group(group, compiled, name, *, target_source):
    return export_softmax(group, compiled, name, target_source=target_source, output_dtype='bf16',
        outputs=group.source.outputs, programs=('segments/' + name,), kind='softmax_narrowing',
        source_request_digest=group.source.digest(),
        narrowing_proof='non-NaN BF16 probabilities; exact widening followed by RNE is identity')
