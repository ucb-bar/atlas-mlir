"""Fold a private input permutation into exact quantizer byte packing.

Merlin proves ownership and renders dense permutation slices. Atlas keeps the
authored quantizer groups, native images and output order unchanged.
"""
from dataclasses import dataclass, replace
import json
from pathlib import Path

from merlin.compile.layout_c import permutation_axes
from merlin.compile.operand_permutations import private_operand_permutations
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


@dataclass(frozen=True)
class InputPermutation:
    source: object
    quantizer: object
    physical: object
    logical: object


def prepare_source(source):
    from .activation_quantize_application import prepare_plan

    if source.physical_extents or source.boundary_layouts or source.output_layouts:
        raise ValueError('quantizer permutation requires dense logical boundaries')
    pure = [node for node in source.nodes if node.effect == 'pure']
    views = [node for node in pure if node.op in {'gather', 'transpose'}]
    if len(views) != 1 or len(pure) != 3:
        raise ValueError('quantizer permutation needs one view and both quantizer projections')
    view = views[0]
    if len(view.inputs) != 1 or view.id in source.outputs:
        raise ValueError('quantizer permutation must remain private')
    physical = source.node(view.inputs[0])
    if physical.effect != 'input' or permutation_axes(physical, view) is None:
        raise ValueError('quantizer permutation must preserve every input byte')
    quantizer = extract_semantic_region(source, tuple(node.id for node in pure if node != view))
    plan = prepare_plan(quantizer)
    if (plan.x.id != view.id or tuple(node.id for node in source.nodes if node.effect != 'pure') != (physical.id,)
            or source.outputs != quantizer.outputs or source.constants
            or source.input_storages != ((physical.id, 'external'),)):
        raise ValueError('quantizer permutation has an unrelated or escaping boundary')
    return InputPermutation(source, quantizer, physical, view)


def alternatives(adapter, source, choices):
    result, groups = list(choices), {}
    adapter._quantizer_input_permutation_groups = groups
    for candidate in private_operand_permutations(source, choices):
        consumer = candidate.consumer
        if (consumer.placement != 'accelerator'
                or dict(consumer.metadata).get('kind') != 'activation_quantize_device'):
            continue
        try:
            group = prepare_source(candidate.region)
        except ValueError:
            continue
        groups[group.source.digest()] = group
        nodes = tuple(sorted((*candidate.producer.nodes, *consumer.nodes)))
        result.append(replace(consumer, id='quantizer_input_permutation_' + '_'.join(map(str, nodes)),
            nodes=nodes, implementation='source_quantizer_input_permutation',
            inputs=candidate.inputs, outputs=candidate.outputs,
            cycles=CycleInterval.unknown('unchanged quantizer images with permutation folded into input packing'),
            metadata=(('kind', 'quantizer_input_permutation'),)))
    return tuple(result)


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .activation_quantize_application import compile_source

    group = prepare_source(group.source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'source-request.json').write_text(json.dumps(group.source.record(), sort_keys=True, indent=2) + '\n')
    compile_source(snapshot, group.quantizer, target_source=target_source,
                   destination=destination / 'quantizer', limits=limits)


def export_group(group, compiled, name, *, target_source):
    from .activation_quantize_application import prepare_c_package

    group = prepare_source(group.source)
    compiled = Path(compiled)
    if (compiled.is_symlink() or {path.name for path in compiled.iterdir()} != {'source-request.json', 'quantizer'}
            or (compiled / 'source-request.json').is_symlink()
            or json.loads((compiled / 'source-request.json').read_text()) != group.source.record()):
        raise ValueError('quantizer permutation publication differs from its source')
    result = prepare_c_package(group.quantizer, compiled / 'quantizer', name,
        target_source=target_source, input_permutation=(group.physical, group.logical))
    description = dict(result.description, kind='quantizer_input_permutation',
        input_node_ids=[group.physical.id], source_request_digest=group.source.digest(),
        input_permutation=dict(source=group.physical.id, logical=group.logical.id,
            source_shape=list(group.physical.type.shape), logical_shape=list(group.logical.type.shape),
            axes=list(permutation_axes(group.physical, group.logical)),
            transport='byte-preserving slices in original quantizer input packing',
            reduction_groups='unchanged logical groups'))
    return replace(result, inputs=(group.physical.id,), description=description,
        programs=tuple(f'segments/{name}/quantizer/segments/body_{index:03d}'
                              for index, _ in enumerate(description['quantizer_schedule']['bodies'])))
