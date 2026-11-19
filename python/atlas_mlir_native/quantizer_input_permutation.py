"""Fold a private input permutation into exact quantizer byte packing.

Merlin proves ownership and renders dense permutation slices. Atlas keeps the
authored quantizer groups, native images and output order unchanged.
"""
from dataclasses import dataclass, replace

from merlin.compile.layout_c import permutation_axes
from merlin.compile.operand_permutations import private_operand_permutations
from merlin.compile.semantic_regions import extract_semantic_region

from .composition_adapter import check_source_request, fold_operand_views, write_source_request


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
    return fold_operand_views(adapter, '_quantizer_input_permutation_groups', 'quantizer_input_permutation',
        private_operand_permutations(source, choices), choices, accepts=lambda kind: kind == 'activation_quantize_device',
        prepare=lambda region, _: prepare_source(region),
        cycles='unchanged quantizer images with permutation folded into input packing')


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .activation_quantize_application import compile_source

    group = prepare_source(group.source)
    destination = write_source_request(destination, group.source, indent=2)
    compile_source(snapshot, group.quantizer, target_source=target_source,
                   destination=destination / 'quantizer', limits=limits)


def export_group(group, compiled, name, *, target_source):
    from .activation_quantize_application import prepare_c_package

    group = prepare_source(group.source)
    compiled = check_source_request(compiled, group.source, 'quantizer permutation publication differs from its source',
                                    {'source-request.json', 'quantizer'})
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
