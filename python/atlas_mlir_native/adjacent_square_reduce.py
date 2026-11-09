"""Keep an adjacent square and trailing seeded reduction in one device image.

The removed reduction encoder is moved to an exactly equivalent admission
test on the square input. No arithmetic or reduction ordering is changed.
"""
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
from math import prod
from pathlib import Path

from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.model import pointwise_identity_maps

from .region_composition import (
    allocate_adjacent_arena,
    link_adjacent_stages,
    publish_adjacent_group,
    verify_adjacent_group,
    export_adjacent_package,
)
from .binding import DRAM_BASE, TILE_BYTES

PREIMAGE_POLICY = 'f32-to-bf16-square-reduction-exact-preimage-v1'


@lru_cache(maxsize=1)
def intermediate_domain_proof():
    """Enumerate every BF16 code, retaining exactly the old two guards.

    f32 input conversion is unchanged RNE. Thus membership of its resulting
    BF16 code in this preimage is equivalent to the original input guard and
    the reduction guard on the selected BF16 square. Both signed input zeros
    square to positive zero, so reduction's negative-zero canonicalization
    is redundant at this particular internal boundary.
    """
    from .pointwise import input_spec
    from .reduction import input_spec as reduction_input_spec
    from . import reference

    before, after = input_spec(), reduction_input_spec()
    admitted = lambda spec, code: any(low <= code <= high for low, high in spec.admitted_code_ranges)
    ranges, original, accepted, mapping = [], 0, 0, bytearray()
    for code in range(65536):
        if not admitted(before, code):
            continue
        original += 1
        squared = 0 if code & 0x7fff == 0 else reference._round_bf16_product(code, code)
        keep = admitted(after, squared)
        mapping.extend(code.to_bytes(2, 'little') + squared.to_bytes(2, 'little') + bytes([keep]))
        if not keep:
            continue
        accepted += 1
        if ranges and code == ranges[-1][1] + 1:
            ranges[-1][1] = code
        else:
            ranges.append([code, code])
    return dict(policy=PREIMAGE_POLICY, checked_codes=65536,
                original_admitted_codes=original, admitted_preimage_codes=accepted,
                admitted_code_ranges=ranges, mapping_sha256=sha256(mapping).hexdigest(),
                arithmetic_reference_sha256=sha256(Path(reference.__file__).read_bytes()).hexdigest(),
                input_conversion='unchanged f32-to-bf16 RNE; exact preimage of both original encoders',
                internal_conversion='finite BF16-to-f32 exact then BF16 RNE preserves every square result code',
                zero_proof='square(+0)=square(-0)=+0; reduction canonicalization is redundant',
                tail_proof='identical row/column tiles; square padded +0 remains reduction padded +0',
                operation_order='complete square loop, then unchanged seeded left-to-right reduction loop')


def input_spec():
    from .pointwise import input_spec as original_spec
    proof = intermediate_domain_proof()
    return replace(original_spec(), policy=PREIMAGE_POLICY,
                   admitted_code_ranges=tuple(tuple(row) for row in proof['admitted_code_ranges']))


def prepare(region, adapter=None):
    from .pointwise import prepare_source as square_precision, prepare_execution_plan
    from .reduction import prepare_source as reduction_precision, prepare_plan as reduction_plan

    pure = [node for node in region.nodes if node.effect == 'pure']
    if len(pure) != 2:
        raise ValueError('square/reduction enlargement requires exactly two operations')
    square, reduced = pure
    if (square.op != 'f32_mul' or len(square.inputs) != 2 or square.inputs[0] != square.inputs[1]
            or square.index_maps != pointwise_identity_maps(len(square.type.shape), 2)
            or not square.type.shape or square.type.dtype != 'f32'
            or region.node(square.inputs[0]).type != square.type
            or len(reduced.inputs) != 2 or reduced.inputs[0] != square.id or region.outputs != (reduced.id,)):
        raise ValueError('enlargement requires a dense self-product with only its reduction escaping')
    first = square_precision(extract_semantic_region(region, (square.id,)))
    second = reduction_precision(extract_semantic_region(region, (reduced.id,)))
    _, first_plan, streaming = prepare_execution_plan(first.request)
    second_plan = reduction_plan(second.request)
    # Large source rows keep the existing separately selected square and
    # reduction regions. Joining their complete loops exceeds the unchanged
    # initialization proof budget; no reduction axis or arithmetic is split.
    if first_plan.packed_bytes > 524288:
        raise ValueError('square/reduction enlargement exceeds the bounded packed square size')
    if (streaming or first_plan.compact_broadcasts
            or second_plan.kept_axes + second_plan.axes != tuple(range(len(square.type.shape)))
            or first_plan.shape != second_plan.matrix_shape
            or first_plan.tile_shape != second_plan.tile_shape
            or first_plan.packed_bytes != second_plan.packed_sizes[0]):
        raise ValueError('square output and reduction input must have identical ordered physical tiles')
    inputs = tuple(node.id for node in region.nodes if node.effect in ('input', 'constant'))
    data, seed = square.inputs[0], reduced.inputs[1]
    if len(inputs) != 2 or set(inputs) != {data, seed}:
        raise ValueError('square/reduction needs distinct complete data and seed boundaries')
    sizes = {data: first_plan.input_packed_bytes[0], seed: second_plan.packed_sizes[1],
             square.id: first_plan.packed_bytes, reduced.id: second_plan.packed_sizes[2]}
    addresses, arena_bytes = allocate_adjacent_arena(sizes, (*inputs, square.id, reduced.id))
    return dict(region=region, precisions=(first, second), plans=(first_plan, second_plan),
                inputs=inputs, outputs=(square.id, reduced.id), data=data, seed=seed,
                addresses=addresses, sizes=sizes, arena_bytes=arena_bytes,
                proof=intermediate_domain_proof())


def alternatives(adapter, source, choices):
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

    choices = list(choices)
    ordered = sorted((row for row in choices if len(row.nodes) == 1), key=lambda row: row.nodes)
    pure = [node for node in source.nodes if node.effect == 'pure']
    adapter._square_groups = {}
    for first, last in zip(ordered, ordered[1:]):
        if (first.placement != 'accelerator' or last.placement != 'accelerator'
                or last.nodes[0] != first.nodes[0] + 1
                or dict(first.metadata).get('kind') != 'pointwise'
                or dict(last.metadata).get('kind') != 'reduction'):
            continue
        square, reduced = pure[first.nodes[0]], pure[last.nodes[0]]
        if not reduced.inputs or reduced.inputs[0] != square.id:
            continue
        try:
            region = extract_semantic_region(source, (square.id, reduced.id))
            group = prepare(region, adapter)
        except ValueError:
            continue
        ports = {port.buffer: port for row in (first, last) for port in (*row.inputs, *row.outputs)}
        adapter._square_groups[region.digest()] = group
        choices.append(replace(first, id='square_reduce_' + first.id,
            nodes=(*first.nodes, *last.nodes), implementation='source_adjacent_square_reduce',
            inputs=tuple(ports[key] for key in group['inputs']),
            outputs=tuple(ports[key] for key in region.outputs), demands=(), occupancy=(),
            scratch_bytes=group['arena_bytes'],
            cycles=CycleInterval.unknown('checked square/reduction enlargement; no measured cost'),
            metadata=(('kind', 'adjacent_square_reduce'),)))
    return tuple(choices)


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .pointwise import compile_pointwise, selected_body_descriptor, virtual_program as square_virtual
    from .reduction import compile_reduction, selected_body_descriptors, virtual_program as reduce_virtual
    from .region_composition import DeviceBuffer, VirtualStage

    destination = Path(destination)
    destination.mkdir()
    stages, selections = [], []
    square_id, reduced_id = group['outputs']
    for index, (precision, plan) in enumerate(zip(group['precisions'], group['plans'], strict=True)):
        keys = (group['data'],) if index == 0 else (square_id, group['seed'])
        addresses = tuple(group['addresses'][key] for key in keys)
        output = group['addresses'][square_id if index == 0 else reduced_id]
        member = destination / 'members' / str(index)
        member.parent.mkdir(exist_ok=True)
        compile_member = compile_pointwise if index == 0 else compile_reduction
        compile_member(snapshot, precision.request,
            fixed_inputs={node.id: (address - DRAM_BASE) // TILE_BYTES
                          for node, address in zip(precision.request.nodes[:-1], addresses, strict=True)},
            fixed_outputs=((output - DRAM_BASE) // TILE_BYTES,), target_source=target_source,
            destination=member, limits=limits)
        if index == 0:
            descriptor = selected_body_descriptor(plan, member / 'body')
            virtual = square_virtual(plan, addresses, output, descriptor)
        else:
            selected_body_descriptors(plan, member / 'body')
            virtual = reduce_virtual(plan, addresses, output)
        stages.append(VirtualStage(str(index), virtual, keys, precision.source.outputs))
        selections.append(dict(request_digest=precision.request.digest(),
                               manifest_sha256=sha256((member / 'manifest.json').read_bytes()).hexdigest()))
    buffers = tuple(DeviceBuffer(key, address, group['sizes'][key],
                    'row_column_zero_bf16_pair_tiles' if key in (group['seed'], reduced_id)
                    else 'mn_tiles_bf16_pair_halves') for key, address in group['addresses'].items())
    link_adjacent_stages(stages, buffers=buffers, initialized=group['inputs'], outputs=group['region'].outputs,
                         destination=destination, target_source=target_source)
    execution = json.loads((destination / 'execution_plan.json').read_text())
    execution.update(input_addresses=[group['addresses'][key] for key in group['inputs']],
                     input_buffer_bytes=[group['sizes'][key] for key in group['inputs']],
                     output_addresses=[group['addresses'][square_id]],
                     output_bytes=sum(group['sizes'][key] for key in group['outputs']))
    publish_adjacent_group(group, destination, kind='adjacent_square_reduce', execution=execution,
        enlargement=dict(proof=group['proof'], members=selections,
            precision_plan=[precision.record() for precision in group['precisions']],
            source_request_digest=group['region'].digest()))


def export_group(group, compiled, name, *, target_source):
    from merlin.compile.c_conversions import render_f32_array_c
    from merlin.compile.c_pointwise import render_pointwise_c
    from merlin.compile.c_reduction import render_reduction_c
    from .reduction import input_spec as seed_spec

    region, compiled = group['region'], Path(compiled)
    execution = verify_adjacent_group(group, compiled, precision_plan=[p.record() for p in group['precisions']])
    first, second = group['plans']
    data_prefix, reduce_prefix = name + '_square', name + '_reduction'
    # These are the original byte-only packers. Inline linkage permits unused
    # companion helpers without weakening warnings on the exported runtime.
    helpers = [render_pointwise_c(first, data_prefix).replace('static void ', 'static inline void '),
               render_reduction_c(second, reduce_prefix).replace('static void ', 'static inline void ')]
    for index, key in enumerate(group['inputs']):
        spec = input_spec() if key == group['data'] else seed_spec()
        helpers.append(render_f32_array_c(spec, prod(region.node(key).type.shape), name + f'_encode_{index}'))
        packer = data_prefix + '_pack_0' if key == group['data'] else reduce_prefix + '_pack_1'
        helpers.append(f'static void {name}_pack_{index}(uint8_t *out,const uint8_t *in) {{ {packer}(out,in); }}')
    helpers.append(f'''static void {name}_assemble(uint8_t *out,const uint8_t *arena) {{
  {reduce_prefix}_assemble(out,arena+{group['sizes'][group['outputs'][0]]}u);
}}''')
    return export_adjacent_package(group, compiled, name, kind='adjacent_square_reduce', target_source=target_source,
        execution=execution, input_sources=group['inputs'], helpers=helpers,
        packed_sizes=(*(group['sizes'][key] for key in group['inputs']),sum(group['sizes'][key] for key in group['outputs'])),
        metadata=dict(precision_plan=[precision.record() for precision in group['precisions']],
            outer_input_conversions={key: (input_spec() if key == group['data'] else seed_spec()).record()
                                     for key in group['inputs']}))
