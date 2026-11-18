"""Read a private matrix transpose during the existing FP8 tile packing.

Merlin proves complete permutation semantics and ownership. Atlas restricts the
mapping to unchanged batch axes and its counted unchanged-code MXU route.
"""
from dataclasses import replace
from functools import partial

from merlin.compile.layout_c import permutation_axes
from merlin.compile.operand_permutations import private_operand_permutations
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.target_binding import NativeCompilationError

from .composition_adapter import fold_operand_views
from .quantized_epilogue import compile_group, export_group as export_counted


def prepare_source(source):
    from .already_quantized import prepare_source as prepare_contraction

    pure = [node for node in source.nodes if node.effect == 'pure']
    views = [node for node in pure if node.op in {'gather', 'transpose'}]
    if len(views) != 1:
        raise ValueError('quantized operand permutation needs one private matrix transpose')
    view = views[0]
    if len(view.inputs) != 1 or view.id in source.outputs:
        raise ValueError('quantized operand permutation must have one input and remain private')
    base = source.node(view.inputs[0])
    axes = permutation_axes(base, view)
    rank = len(view.type.shape)
    if (rank < 2 or axes != (*range(rank - 2), rank - 1, rank - 2)
            or base.effect not in {'input', 'constant'}):
        raise ValueError('quantized operand permutation must preserve every batch axis')
    logical = extract_semantic_region(source, tuple(node.id for node in pure if node != view))
    plan = prepare_contraction(logical)
    if (plan.operand_ids.count(view.id) != 1
            or sum(node.inputs.count(view.id) for node in pure) != 1):
        raise ValueError('quantized operand permutation must feed exactly one code operand')
    inputs = tuple(node.id for node in source.nodes if node.effect in {'input', 'constant'})
    expected = {base.id if key == view.id else key for key in plan.input_ids}
    if set(inputs) != expected or len(inputs) != len(plan.input_ids):
        raise ValueError('quantized operand permutation needs distinct unchanged boundary ports')
    return replace(plan, source=source, input_ids=inputs,
                   operand_permutation=(view.id, base.id, axes))


def alternatives(adapter, source, choices):
    from .quantized_epilogue import counted_scratch

    return fold_operand_views(adapter, '_quantized_operand_permutation_groups', 'quantized_operand_permutation',
        private_operand_permutations(source, choices), choices,
        accepts=lambda kind: kind in {'already_quantized', 'quantized_narrowing', 'quantized_epilogue'},
        prepare=lambda region, _: prepare_source(region), errors=(ValueError, NativeCompilationError),
        admit=counted_scratch, cycles='counted MXU with folded operand permutation; unmeasured')


export_group = partial(export_counted, kind='quantized_operand_permutation')
