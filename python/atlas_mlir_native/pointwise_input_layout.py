"""Absorb private full-coverage reads into pointwise operand packing.

Merlin proves affine composition and ownership. Atlas separately rechecks the
selected pointwise arithmetic, every input code and the resulting tile frame.
Full source coverage preserves the old validation domain, including broadcasts.
"""
from dataclasses import dataclass, replace

from merlin.compile.operand_layouts import fold_private_full_read, private_operand_layouts

from .composition_adapter import check_source_request, fold_operand_views, write_source_request
from . import pointwise_quantize


@dataclass(frozen=True)
class InputLayout:
    source: object
    consumer: object
    kind: str
    group: object
    view_id: str


def prepare(source, adapter, kind):
    from .pointwise import prepare_source, plan_abi

    pure = [node for node in source.nodes if node.effect == 'pure']
    views = [node for node in pure if node.op in ('gather', 'transpose')]
    if len(views) != 1 or kind not in ('pointwise', 'pointwise_quantize'):
        raise ValueError('pointwise input layout requires one private affine read')
    view, = views
    operations = [node for node in pure if node != view]
    if len(operations) != (3 if kind == 'pointwise_quantize' else 1):
        raise ValueError('pointwise input layout requires one arithmetic producer')
    consumer = fold_private_full_read(source, view.id, operations[0].id)
    if kind == 'pointwise_quantize':
        group = pointwise_quantize.prepare(consumer, adapter, 'pointwise')
    else:
        group = prepare_source(consumer, bf16_policy=adapter.bf16_pointwise,
            exp_policy=adapter.source_exp_policy,
            rne_add_policy=adapter.qualified_rne_add
                if operations[0].op in ('f32_add', 'bf16_add') else None)
        plan_abi(group.request)
    return InputLayout(source, consumer, kind, group, view.id)


def alternatives(adapter, source, choices):
    return fold_operand_views(adapter, '_pointwise_input_layout_groups', 'pointwise_input_layout',
        private_operand_layouts(source, choices), choices, prepare=lambda region, kind: prepare(region, adapter, kind),
        accepts=lambda kind: kind in ('pointwise', 'pointwise_quantize'),
        cycles='pointwise operand packing absorbs a private full affine read')


def _verify_source(group):
    readers = [node for node in group.source.nodes if group.view_id in node.inputs]
    if len(readers) != 1:
        raise ValueError('pointwise input layout lost its private reader')
    replayed = fold_private_full_read(group.source, group.view_id, readers[0].id)
    if replayed != group.consumer or group.group.source != replayed:
        raise ValueError('pointwise input layout differs from composed source indexing')


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .pointwise_narrowing import compile_request

    _verify_source(group)
    destination = write_source_request(destination, group.source)
    if group.kind == 'pointwise_quantize':
        pointwise_quantize.compile_group(group.group, snapshot, target_source=target_source,
                                         destination=destination / 'consumer', limits=limits)
    else:
        compile_request(snapshot, group.group.request, target_source=target_source,
                        destination=destination / 'consumer', limits=limits)


def export_group(group, compiled, name, *, target_source):
    from .pointwise_narrowing import export_precision

    _verify_source(group)
    compiled = check_source_request(compiled, group.source, 'pointwise input layout publication differs from its source')
    if group.kind == 'pointwise_quantize':
        result = pointwise_quantize.export_group(group.group, compiled / 'consumer', name, target_source=target_source)
        programs = ('segments/' + name + '/consumer/pointwise',)
    else:
        result = export_precision(group.group, compiled / 'consumer', name, target_source=target_source,
                                  outputs=group.source.outputs, programs=())
        programs = ('segments/' + name + '/consumer',)
    return replace(result, programs=programs, description=dict(result.description,
        kind='pointwise_input_layout', source_request_digest=group.source.digest(),
        folded_view=group.view_id, consumer_kind=group.kind,
        layout_proof='private full-source affine read; composed indexing; identical input validation domain'))
