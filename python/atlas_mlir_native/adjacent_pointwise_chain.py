"""Compose ordinary three-operation pointwise chains with device boundaries."""
from dataclasses import replace
import json
from math import prod
import os
from pathlib import Path

from merlin.compile.semantic_regions import extract_semantic_region
from .adjacent_pointwise import MAX_ADJACENT_POINTWISE_TILE_VISITS

from .region_composition import (
    allocate_adjacent_arena,
    compile_pointwise_member,
    compile_adjacent_virtual,
    publish_adjacent_group,
    verify_adjacent_group,
    export_adjacent_package,
)


def _identity_view(plan, operand, producer):
    from merlin.semantic_compiler.model import pointwise_identity_maps
    return (producer.type.shape == plan.logical_shape
            and plan.input_map(operand) == pointwise_identity_maps(len(plan.logical_shape),1)[0]
            and plan.input_shape(operand) == plan.shape)


def _column_view(plan, operand, producer):
    rank=len(plan.logical_shape)
    mapping=plan.input_map(operand)
    expected=tuple(tuple(int(row==column and row!=rank-1) for column in range(rank)) for row in range(rank))
    return (rank >= 2 and producer.type.shape == (*plan.logical_shape[:-1],1)
            and mapping.coefficients == expected and mapping.offsets == (0,)*rank
            and plan.shape[1] >= 32 and plan.input_shape(operand) == (plan.shape[0],32))


def prepare(region, adapter):
    from .pointwise import prepare_source, prepare_execution_plan
    original = region
    pure=[node for node in region.nodes if node.effect=='pure']
    narrowing = None
    if len(pure) == 4:
        from .pointwise_narrowing import prepare as prepare_narrowing
        # Reuse the ordinary non-NaN result proof and exact private-cast check.
        narrowing = prepare_narrowing(extract_semantic_region(region, (pure[-2].id, pure[-1].id)), adapter)
        if region.outputs != (pure[-1].id,):
            raise ValueError('pointwise chain narrowing requires only its final cast to escape')
        region = extract_semantic_region(region, tuple(node.id for node in pure[:-1]))
        pure = pure[:-1]
    if (len(pure)!=3 or region.outputs!=(pure[-1].id,) or any(node.type.dtype!='f32' for node in region.nodes)):
        raise ValueError('pointwise chain needs three f32 operations with only its final result escaping')
    if any(pure[i].id not in pure[i+1].inputs for i in range(2)) or pure[0].id in pure[2].inputs:
        raise ValueError('pointwise chain needs two consecutive single internal edges')
    precisions=[];plans=[]
    for node in pure:
        precision=prepare_source(extract_semantic_region(region,(node.id,)),allow_math=True,
            bf16_policy=adapter.bf16_pointwise,
            rne_add_policy=adapter.qualified_rne_add if node.op in {'f32_add','bf16_add'} else None,
            exp_policy=adapter.source_exp_policy)
        _,plan,streaming=prepare_execution_plan(precision.request)
        if streaming:
            raise ValueError('pointwise chain cannot contain host-streamed children')
        precisions.append(precision);plans.append(plan)
    # Declining the specialized chain must not select an equivalently large
    # generic image with extra device boundary scans. Keep complete singles.
    tile_visits=sum(prod(plan.tile_counts) for plan in plans)
    if tile_visits > MAX_ADJACENT_POINTWISE_TILE_VISITS:
        raise ValueError('pointwise chain enlargement exceeds bounded complete tile work: '
                         f'{tile_visits} > {MAX_ADJACENT_POINTWISE_TILE_VISITS}')
    edges=[]
    for index in range(2):
        producer=pure[index];consumer=precisions[index+1]
        keys=consumer.record()['inputs']
        if keys.count(producer.id)!=1:
            raise ValueError('pointwise chain needs one physical view of each intermediate')
        operand=keys.index(producer.id)
        plan=plans[index+1]
        if _identity_view(plan,operand,producer):
            kind='identity'
        else:
            plan=replace(plan,compact_broadcasts=True)
            if not _column_view(plan,operand,producer):
                raise ValueError('pointwise chain intermediate is not an identity or compact column view')
            plans[index+1]=plan;kind='broadcast_column'
        rows,columns=plans[index].shape
        if plan.input_packed_bytes[operand] != plans[index].packed_bytes:
            raise ValueError('pointwise chain transition changes its packed page count')
        edges.append(dict(producer=index,consumer=index+1,operand=operand,kind=kind,rows=rows,columns=columns,
                          spec=consumer.operand_specs[operand],pad=int.from_bytes(plan.input_pad(operand),'little')))
    inputs=tuple(node.id for node in region.nodes if node.effect in {'input','constant'})
    views=[];bindings=[]
    for index,(precision,plan) in enumerate(zip(precisions,plans,strict=True)):
        keys=precision.record()['inputs'];stage=[]
        for operand,key in enumerate(keys):
            if key in inputs:
                name=f'input_view_{index}_{operand}'
                views.append(dict(name=name,source=key,stage=index,operand=operand,
                                  size=plan.input_packed_bytes[operand],spec=precision.operand_specs[operand]))
            else:
                edge=next((edge for edge in edges if edge['consumer']==index and edge['operand']==operand),None)
                if edge is None:
                    raise ValueError('pointwise chain has an unaccounted internal source view')
                name=f'boundary_{index-1}'
            stage.append(name)
        bindings.append(tuple(stage))
    if set(view['source'] for view in views)!=set(inputs):
        raise ValueError('pointwise chain has unaccounted public source ports')
    mutable=tuple(f'output_{i}' for i in range(3))+('boundary_0','boundary_1','status')
    sizes={view['name']:view['size'] for view in views}
    sizes.update({f'output_{i}':plan.packed_bytes for i,plan in enumerate(plans)})
    sizes.update({f'boundary_{i}':plans[edge['consumer']].input_packed_bytes[edge['operand']] for i,edge in enumerate(edges)})
    sizes['status']=2048
    addresses, arena_bytes = allocate_adjacent_arena(sizes, (*(view['name'] for view in views),*mutable))
    proof=dict(boundaries=[dict(producer=pure[edge['producer']].id,consumer=pure[edge['consumer']].id,
                operand=edge['operand'],kind=edge['kind'],source_matrix=[edge['rows'],edge['columns']],
                consumer_conversion=edge['spec'].record(),pad_code=edge['pad']) for edge in edges],
        rejection='invalid logical codes become admitted consumer padding and set shared invocation status; C checks status before output publication',
        operation_order='complete each unchanged selected pointwise child in source order; preserve both intermediate encoder checks on device')
    if narrowing is not None:
        proof['output_narrowing'] = 'non-NaN BF16 result; exact widening followed by RNE is identity'
    return dict(region=original,precisions=tuple(precisions),plans=tuple(plans),inputs=inputs,views=tuple(views),
                bindings=tuple(bindings),edges=tuple(edges),mutable=mutable,sizes=sizes,addresses=addresses,
                arena_bytes=arena_bytes,proof=proof)


def alternatives(adapter, source, choices):
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval
    choices=list(choices)
    excluded={row.nodes for row in choices if dict(row.metadata).get('kind')=='adjacent_pointwise'}
    singles=sorted((row for row in choices if len(row.nodes)==1),key=lambda row:row.nodes)
    pure=[node for node in source.nodes if node.effect=='pure']
    adapter._pointwise_chain_groups={}
    candidates = [singles[start:start + width] for width in (3, 4)
                  for start in range(len(singles) - width + 1)]
    for children in candidates:
        nodes=tuple(row.nodes[0] for row in children)
        if (nodes[:3] in excluded or any(b != a + 1 for a, b in zip(nodes, nodes[1:]))
                or any(row.placement!='accelerator' for row in children)
                or any(dict(row.metadata).get('kind') not in {'pointwise','pointwise_math'} for row in children[:3])
                or len(children) == 4 and dict(children[-1].metadata).get('kind') != 'exact_float_device'):
            continue
        try:
            region=extract_semantic_region(source,tuple(pure[node].id for node in nodes))
            group=prepare(region,adapter)
        except ValueError:
            continue
        ports={port.buffer:port for row in children for port in (*row.inputs,*row.outputs)}
        adapter._pointwise_chain_groups[region.digest()]=group
        prefix = 'pointwise_chain_narrowing_' if len(children) == 4 else 'pointwise_chain_'
        choices.append(replace(children[0],id=prefix+children[0].id,nodes=nodes,
            implementation='source_adjacent_pointwise_chain',inputs=tuple(ports[key] for key in group['inputs']),
            outputs=tuple(ports[key] for key in region.outputs),demands=(),occupancy=(),scratch_bytes=group['arena_bytes'],
            cycles=CycleInterval.unknown('device-checked pointwise chain; no measured cost'),metadata=(('kind','adjacent_pointwise_chain'),)))
    return tuple(choices)


def virtual_group(group, descriptors, *, tools):
    from .device_boundary import virtual_transition
    from .pointwise import virtual_program
    from .region_composition import rematerialize_scalar_constants
    from .resident_graph import compose_fragments
    fragments=[]
    base=group['addresses'][group['mutable'][0]];size=sum(group['sizes'][key] for key in group['mutable'])
    for index,(plan,descriptor) in enumerate(zip(group['plans'],descriptors,strict=True)):
        addresses=[group['addresses'][key] for key in group['bindings'][index]]
        fragments.append(virtual_program(plan,addresses,group['addresses'][f'output_{index}'],descriptor))
        if index<2:
            edge=group['edges'][index];spec=edge['spec']
            fragments.append(virtual_transition(source_address=group['addresses'][f'output_{index}'],
                destination_address=group['addresses'][f'boundary_{index}'],status_address=group['addresses']['status'],
                rows=edge['rows'],columns=edge['columns'],kind=edge['kind'],admitted_code_ranges=spec.admitted_code_ranges,
                pad=edge['pad'],canonicalize_negative_zero=spec.canonicalize_negative_zero,output_base=base,output_bytes=size,
                initialize_status=index==0))
    virtual=compose_fragments(fragments,tools=tools,boundaries_lowered=True)
    return rematerialize_scalar_constants(virtual)


def execution_plan(group):
    return dict(schema='atlas.adjacent_pointwise_chain_execution.v1',launches_per_invocation=1,
        input_addresses=[group['addresses'][view['name']] for view in group['views']],
        input_buffer_bytes=[view['size'] for view in group['views']],
        output_addresses=[group['addresses'][group['mutable'][0]]],output_bytes=sum(group['sizes'][key] for key in group['mutable']),
        addresses=group['addresses'],buffer_bytes=group['sizes'],proof=group['proof'])

def compile_group(group, snapshot, *, target_source, destination, limits):
    destination=Path(destination);target_source=Path(target_source);destination.mkdir()
    descriptors=[];selections=[]
    for index,(precision,plan) in enumerate(zip(group['precisions'],group['plans'],strict=True)):
        # Children retain their ordinary source ABI for qualification. Only the
        # linked virtual member is remapped/compacted; its selected tile body
        # and every scalar arithmetic step remain byte-for-byte identical.
        member=destination/'members'/str(index)
        descriptor, selection = compile_pointwise_member(precision, plan, snapshot, addresses=(), output=None,
            target_source=target_source, destination=member, limits=limits, ordinary_abi=True)
        descriptors.append(descriptor); selections.append(selection)
    tools=Path(os.environ.get('ATLAS_OOT_BIN_DIR',target_source/'build/bin'))
    virtual=virtual_group(group,descriptors,tools=tools)
    compile_adjacent_virtual(virtual, destination=destination, target_source=target_source)
    publish_adjacent_group(group, destination, kind='adjacent_pointwise_chain', execution=execution_plan(group),
        enlargement=dict(proof=group['proof'], members=selections,
            precision_plan=[precision.record() for precision in group['precisions']],
            plans=[plan.record() for plan in group['plans']], source_request_digest=group['region'].digest()))


def export_group(group, compiled, name, *, target_source):
    from merlin.compile.c_conversions import render_f32_array_c
    from merlin.compile.c_pointwise import render_pointwise_c
    region=group['region'];compiled=Path(compiled)
    execution = verify_adjacent_group(group, compiled, precision_plan=[p.record() for p in group['precisions']],
                                      plans=[plan.record() for plan in group['plans']])
    helpers=[render_pointwise_c(plan,name+f'_stage_{index}').replace('static void ','static inline void ')
             for index,plan in enumerate(group['plans'])]
    for index,view in enumerate(group['views']):
        helpers.append(render_f32_array_c(view['spec'],prod(region.node(view['source']).type.shape),name+f'_encode_{index}'))
        helpers.append(f'static void {name}_pack_{index}(uint8_t *out,const uint8_t *in) {{ '
                       f'{name}_stage_{view["stage"]}_pack_{view["operand"]}(out,in); }}')
    base=group['addresses'][group['mutable'][0]]
    result_offset=group['addresses']['output_2']-base
    helpers.append(f'static void {name}_assemble(uint8_t *out,const uint8_t *arena) {{ {name}_stage_2_assemble(out,arena+{result_offset}u); }}')
    input_bytes=sum(view['size'] for view in group['views'])
    status_offset=input_bytes+group['addresses']['status']-base
    check='  if ('+' | '.join(f'region->workspace[{status_offset+i}u]' for i in range(4))+f') return {name.upper()}_INPUT_DOMAIN;'
    if execution != execution_plan(group):
        raise ValueError('pointwise chain execution ABI differs from its source')
    return export_adjacent_package(group, compiled, name, kind='adjacent_pointwise_chain', target_source=target_source,
        execution=execution, input_sources=tuple(view['source'] for view in group['views']), helpers=helpers,
        completion_check=check, source_input_indices=tuple(group['inputs'].index(view['source']) for view in group['views']),
        packed_sizes=(*(view['size'] for view in group['views']),sum(group['sizes'][key] for key in group['mutable'])),
        metadata=dict(precision_plan=[p.record() for p in group['precisions']],
            outer_input_views=[dict(source=view['source'],stage=view['stage'],operand=view['operand'],conversion=view['spec'].record())
                               for view in group['views']]))
