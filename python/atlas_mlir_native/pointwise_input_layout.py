"""Absorb private full-coverage reads into pointwise operand packing.

Merlin proves affine composition and ownership. Atlas separately rechecks the
selected pointwise arithmetic, every input code and the resulting tile frame.
Full source coverage preserves the old validation domain, including broadcasts.
"""
from dataclasses import dataclass, replace
import json
from pathlib import Path

from merlin.compiler.transforms.operand_layouts import fold_private_full_read, private_operand_layouts


@dataclass(frozen=True)
class InputLayout:
    source: object
    consumer: object
    kind: str
    group: object
    view_id: str


def prepare(source, adapter, kind):
    from .pointwise import prepare_source, plan_abi
    from .pointwise_quantize import prepare as quantized

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
        group = quantized(consumer, adapter, 'pointwise')
    else:
        group = prepare_source(consumer, bf16_policy=adapter.bf16_pointwise,
            exp_policy=adapter.source_exp_policy,
            rne_add_policy=adapter.qualified_rne_add
                if operations[0].op in ('f32_add', 'bf16_add') else None)
        plan_abi(group.request)
    return InputLayout(source, consumer, kind, group, view.id)


def alternatives(adapter, source, choices):
    from merlin.compiler.partition.global_plan import CycleInterval

    result, groups = list(choices), {}
    adapter._pointwise_input_layout_groups = groups
    for candidate in private_operand_layouts(source, choices):
        consumer = candidate.consumer
        kind = dict(consumer.metadata).get('kind')
        if consumer.placement != 'accelerator' or kind not in ('pointwise', 'pointwise_quantize'):
            continue
        try:
            group = prepare(candidate.region, adapter, kind)
        except ValueError:
            continue
        groups[group.source.digest()] = group
        nodes = tuple(sorted((*candidate.producer.nodes, *consumer.nodes)))
        result.append(replace(consumer, id='pointwise_input_layout_' + '_'.join(map(str, nodes)),
            nodes=nodes, implementation='source_pointwise_input_layout',
            inputs=candidate.inputs, outputs=candidate.outputs,
            cycles=CycleInterval.unknown('pointwise operand packing absorbs a private full affine read'),
            metadata=(('kind', 'pointwise_input_layout'),)))
    return tuple(result)


def _verify_source(group):
    readers = [node for node in group.source.nodes if group.view_id in node.inputs]
    if len(readers) != 1:
        raise ValueError('pointwise input layout lost its private reader')
    replayed = fold_private_full_read(group.source, group.view_id, readers[0].id)
    if replayed != group.consumer or group.group.source != replayed:
        raise ValueError('pointwise input layout differs from composed source indexing')


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .pointwise import compile_pointwise, plan_abi
    from .pointwise_quantize import compile_group as compile_quantized

    _verify_source(group)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'source-request.json').write_text(json.dumps(group.source.record(), sort_keys=True)+'\n')
    if group.kind == 'pointwise_quantize':
        compile_quantized(group.group, snapshot, target_source=target_source,
                          destination=destination/'consumer', limits=limits)
    else:
        request = group.group.request
        abi = plan_abi(request)
        compile_pointwise(snapshot, request, fixed_inputs=abi['fixed_inputs'],
            fixed_outputs=tuple(abi['fixed_outputs']), target_source=target_source,
            destination=destination/'consumer', limits=limits)


def export_group(group, compiled, name, *, target_source):
    from merlin.compiler.build.regions import CompiledCRegion
    from .c_runtime import RUNTIME_SHARED_FILES
    from .pointwise_c import prepare_precision_c_package
    from .pointwise_quantize import export_group as export_quantized

    _verify_source(group)
    compiled = Path(compiled)
    if (compiled.is_symlink() or (compiled/'source-request.json').is_symlink()
            or json.loads((compiled/'source-request.json').read_text()) != group.source.record()):
        raise ValueError('pointwise input layout publication differs from its source')
    if group.kind == 'pointwise_quantize':
        result = export_quantized(group.group, compiled/'consumer', name, target_source=target_source)
        programs = ('segments/' + name + '/consumer/pointwise',)
    else:
        record, files = prepare_precision_c_package(group.group, compiled/'consumer', name,
                                                    target_source=Path(target_source))
        record.update(context_type=name+'_context', source_context_type=name+'_source_context',
            binding_function=name+'_bind', source_binding_function=name+'_source_bind',
            launches_per_invocation=record.get('streaming', {}).get('launch_count', 1))
        result = CompiledCRegion(name+'_source_dispatch', name+'.h', tuple(record['input_node_ids']),
            group.source.outputs, files, RUNTIME_SHARED_FILES, (), record)
        programs = ('segments/' + name + '/consumer',)
    return replace(result, programs=programs, description=dict(result.description,
        kind='pointwise_input_layout', source_request_digest=group.source.digest(),
        folded_view=group.view_id, consumer_kind=group.kind,
        layout_proof='private full-source affine read; composed indexing; identical input validation domain'))
