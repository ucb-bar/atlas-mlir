"""Absorb a private dense read into exact-float input byte packing.

Merlin owns the graph and affine-index proofs. Atlas retains the original
arithmetic, domain checks, slice schedule and atomic output publication.
"""
from dataclasses import dataclass, replace
import json
from pathlib import Path

from merlin.compile.layout_c import render_indexed_slice_copy
from merlin.compile.operand_layouts import private_operand_layouts
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


@dataclass(frozen=True)
class InputLayout:
    source: object
    arithmetic: object
    physical: object
    logical: object


def prepare_source(source):
    from .exact_float_application import prepare_plan

    if source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError('float input layout requires dense logical boundaries')
    pure = [node for node in source.nodes if node.effect == 'pure']
    views = [node for node in pure if node.op in {'gather', 'transpose'}]
    if len(views) != 1 or len(pure) != 2:
        raise ValueError('float input layout needs one view and one exact operation')
    view = views[0]
    if len(view.inputs) != 1 or view.id in source.outputs:
        raise ValueError('float input layout must remain private')
    physical = source.node(view.inputs[0])
    if (physical.effect != 'input' or physical.op != 'input' or physical.inputs
            or physical.attrs or physical.index_maps):
        raise ValueError('float input layout requires an external dense operand')
    # Includes the signed affine-index and host size_t overflow checks.
    render_indexed_slice_copy(physical, view, 'check_input_layout')
    arithmetic = extract_semantic_region(source, tuple(node.id for node in pure if node != view))
    plan = prepare_plan(arithmetic).plan
    operand_ids = tuple(node.id for node in plan.operands)
    if view.id not in operand_ids or physical.id in operand_ids:
        raise ValueError('float input layout requires one distinct packed input port')
    inputs = tuple(physical.id if key == view.id else key for key in operand_ids)
    if (source.outputs != arithmetic.outputs or source.output_storages != arithmetic.output_storages or source.constants
            or {node.id for node in source.nodes if node.effect != 'pure'} != set(inputs)
            or dict(source.input_storages) != dict.fromkeys(inputs, 'external')):
        raise ValueError('float input layout has an unrelated or escaping boundary')
    return InputLayout(source, arithmetic, physical, view)


def alternatives(adapter, source, choices):
    result, groups = list(choices), {}
    adapter._float_input_layout_groups = groups
    for candidate in private_operand_layouts(source, choices):
        consumer = candidate.consumer
        if (consumer.placement != 'accelerator'
                or dict(consumer.metadata).get('kind') != 'exact_float_device'):
            continue
        try:
            group = prepare_source(candidate.region)
        except ValueError:
            continue
        groups[group.source.digest()] = group
        nodes = tuple(sorted((*candidate.producer.nodes, *consumer.nodes)))
        result.append(replace(consumer, id='float_input_layout_' + '_'.join(map(str, nodes)),
            nodes=nodes, implementation='source_float_input_layout',
            inputs=candidate.inputs, outputs=candidate.outputs,
            cycles=CycleInterval.unknown('unchanged exact-float images with layout absorbed into input packing'),
            metadata=(('kind', 'float_input_layout'),)))
    return tuple(result)


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .exact_float_application import compile_source

    group = prepare_source(group.source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'source-request.json').write_text(json.dumps(group.source.record(), sort_keys=True, indent=2) + '\n')
    compile_source(snapshot, group.arithmetic, target_source=target_source,
                   destination=destination / 'float', limits=limits)


def export_group(group, compiled, name, *, target_source):
    from .exact_float_application import prepare_c_package

    group = prepare_source(group.source)
    compiled = Path(compiled)
    if (compiled.is_symlink() or {path.name for path in compiled.iterdir()} != {'source-request.json', 'float'}
            or (compiled / 'source-request.json').is_symlink()
            or json.loads((compiled / 'source-request.json').read_text()) != group.source.record()):
        raise ValueError('float input layout publication differs from its source')
    result = prepare_c_package(group.arithmetic, compiled / 'float', name,
        target_source=target_source, input_layout=(group.physical, group.logical))
    inputs = tuple(group.physical.id if key == group.logical.id else key for key in result.inputs)
    description = dict(result.description, kind='float_input_layout',
        input_node_ids=list(inputs), source_request_digest=group.source.digest(),
        input_layout=dict(source=group.physical.id, logical=group.logical.id,
            source_shape=list(group.physical.type.shape), logical_shape=list(group.logical.type.shape),
            read_map=group.logical.index_maps[0].record(),
            transport='byte-preserving slices in original exact-float input packing'))
    return replace(result, inputs=inputs, description=description,
        programs=tuple(f'segments/{name}/float/segments/body_{index:03d}'
                       for index, _ in enumerate(description['float_schedule']['bodies'])))
