"""Bounded two-row operand panels for ordered MXU1 contractions.

Each K panel loads both activation rows once, then each weight column once for
both rows. Completed BF16 partial sums live in the existing output buffer between
panels. Restoration runs only after all K terms; host publication still checks
one sticky status before exposing any result.
"""
from .exponent_restoration import Emitter, STATE, BF16, SLOT

FP8 = '!atlas.virtual_fp8'
WEIGHT = '!atlas.virtual_mxu_weight<1>'
ACC = '!atlas.virtual_mxu_acc<1>'
PANEL_K_TILES = 32
MAX_K_TILES = 192


def admitted(plan):
    from math import prod
    mc, nc, kc = ((v+31)//32 for v in (plan.m, plan.n, plan.k))
    # Tiny reductions cannot amortize the partial-result traffic. An admitted
    # candidate saves bytes versus a complete activation-row cache; native
    # sizing and measured timing remain separate gates.
    return (prod(plan.batch_shape) == 1 and mc >= 2 and mc % 2 == 0 and nc >= 2
            and PANEL_K_TILES <= kc <= MAX_K_TILES and plan.epilogue_precision is None)


def virtual_source(plan, layout, restoration):
    """Emit typed slots/operations; native allocation and timing remain mandatory."""
    if not admitted(plan):
        raise ValueError('paired operand panels require one batch, paired M tiles and bounded K')
    mc, nc, kc = ((v+31)//32 for v in (plan.m, plan.n, plan.k))
    a, w = layout['input_addresses'][:2]
    output = layout['output_addresses'][0]
    pages = restoration.pages()
    e = Emitter('panel')
    e.lines = ['module {', f'  func.func @looped_matmul() -> {STATE} attributes {{'
        f'atlas.retain_bf16_accumulators, atlas.input_dram_base = {a} : i64, '
        f'atlas.output_dram_base = {output} : i64, '
        f'atlas.output_dram_bytes = {layout["output_bytes"]+layout["status_bytes"]} : i64}} {{',
        f'    %start = "atlas.virtual_start"() : () -> {STATE}']
    for kind, rows in (('a', 2), ('w', 1)):
        for row in range(rows):
            for tile in range(min(PANEL_K_TILES,kc)):
                e.emit(f'%panel_{kind}_{row}_{tile} = "atlas.virtual_slot"() '
                    f'{{kind = "fp8", words = 256 : i32, slot_word = -1 : i32, purpose = "scratch"}} : () -> {SLOT}')
    e.lines.extend(restoration.entry())

    def offset(base, amount):
        return e.binary('addi',base,e.const(amount)) if amount else base

    def branch(label, names, types):
        e.emit(f'cf.br ^{label}(' + ', '.join('%'+name for name in names) + ' : ' + ', '.join(types) + ')')

    def block(label, names, types):
        e.lines.append(f'  ^{label}(' + ', '.join('%'+name+': '+dtype for name,dtype in zip(names,types,strict=True)) + '):')
        e.state = names[0]

    def load(address, dtype):
        return e.effect('dma_load_'+('fp8' if dtype==FP8 else 'bf16'),
                        [address,e.const(1024 if dtype==FP8 else 2048)],['i32','i32'],dtype)

    def save(slot, value, dtype):
        e.effect('save',[slot,value],[SLOT,dtype])

    def restore(slot, dtype):
        return e.effect('restore',[slot],[SLOT],dtype)

    def product(left, right, partial):
        weight = e.effect('mxu_load_weight',[right],[FP8],WEIGHT,' {unit = 1 : i32}')
        if partial is None:
            value = e.effect('mxu_reset',[left,weight],[FP8,WEIGHT],ACC)
        else:
            seed = e.effect('mxu_load_acc_bf16',[partial],[BF16],ACC,' {unit = 1 : i32}')
            value = e.effect('mxu_accumulate',[left,weight,seed],[FP8,WEIGHT,ACC],ACC)
        return e.effect('mxu_readout_bf16',[value],[ACC],BF16)

    def store(value, address):
        e.effect('dma_store_bf16',[value,address,e.const(2048)],[BF16,'i32','i32'])

    initial = [e.const(v) for v in (0,a,output,pages[0][0])]
    pair_types = [STATE,'i32','i32','i32','i32']
    branch('pair_header',['restore_start',*initial],pair_types)
    block('pair_header',['pair_s','pair','pair_a','pair_y','pair_lpage'],pair_types)
    more = e.cmp('ult','pair',e.const(mc//2))
    e.emit(f'cf.cond_br %{more}, ^pair_body(%pair_s : {STATE}), ^done(%pair_s : {STATE})')
    block('pair_body',['pair_body_s'],[STATE])

    # The panel count is statically capped. Each N traversal is counted, so
    # generated code grows with bounded K, never with the number of columns.
    for phase, first in enumerate(range(0,kc,PANEL_K_TILES)):
        count = min(PANEL_K_TILES,kc-first)
        for row in range(2):
            for tile in range(count):
                address = offset('pair_a',(row*kc+first+tile)*1024)
                save(f'panel_a_{row}_{tile}',load(address,FP8),FP8)
        names = [f'p{phase}_s',f'p{phase}_n',f'p{phase}_w',f'p{phase}_y']
        types = [STATE,'i32','i32','i32']
        branch(f'p{phase}_header',[e.state,e.const(0),e.const(w+first*1024),'pair_y'],types)
        block(f'p{phase}_header',names,types)
        more = e.cmp('ult',names[1],e.const(nc))
        e.emit(f'cf.cond_br %{more}, ^p{phase}_body(%{e.state} : {STATE}), ^p{phase}_done(%{e.state} : {STATE})')
        block(f'p{phase}_body',[f'p{phase}_body_s'],[STATE])
        for tile in range(count):
            save(f'panel_w_0_{tile}',load(offset(names[2],tile*1024),FP8),FP8)
        for row in range(2):
            address = offset(names[3],row*nc*2048)
            partial = load(address,BF16) if first else None
            for tile in range(count):
                left = restore(f'panel_a_{row}_{tile}',FP8)
                right = restore(f'panel_w_0_{tile}',FP8)
                partial = product(left,right,partial)
            store(partial,address)
        branch(f'p{phase}_header',[e.state,offset(names[1],1),offset(names[2],kc*1024),
                                  offset(names[3],2048)],types)
        block(f'p{phase}_done',[f'p{phase}_done_s'],[STATE])

    # Reuse the original restoration hook once, after the complete reduction.
    # Its m/n names retain logical tail checks in the scalar fallback.
    rm_types = [STATE,'i32','i32','i32']
    branch('restore_m_header',[e.state,e.const(0),'pair_y','pair_lpage'],rm_types)
    block('restore_m_header',['rm_s','rm','rm_y','restore_lpage'],rm_types)
    more = e.cmp('ult','rm',e.const(2))
    e.emit(f'cf.cond_br %{more}, ^restore_m_body(%rm_s : {STATE}), ^pair_next(%rm_s : {STATE})')
    block('restore_m_body',['rm_body_s'],[STATE])
    pair_m = e.binary('shli','pair',e.const(1))
    e.emit(f'%m = arith.addi %{pair_m}, %rm : i32')
    rn_types = [STATE,'i32','i32','i32']
    branch('restore_n_header',[e.state,e.const(0),'rm_y',e.const(pages[1][0])],rn_types)
    block('restore_n_header',['rn_s','n','ytile','restore_rpage'],rn_types)
    more = e.cmp('ult','n',e.const(nc))
    e.emit(f'cf.cond_br %{more}, ^restore_n_body(%rn_s : {STATE}), ^restore_m_next(%rn_s : {STATE})')
    block('restore_n_body',['rn_body_s'],[STATE])
    value = load('ytile',BF16)
    branch('store',[e.state,value],[STATE,BF16])
    block('store',['ss','result'],[STATE,BF16])
    e.emit('%out_bytes = arith.constant 2048 : i32')
    exponent_addresses = [name if page[1] else e.const(page[0])
                          for name,page in zip(('restore_lpage','restore_rpage'),pages,strict=True)]
    e.lines.extend(restoration.tile(output_address='ytile',exponent_addresses=exponent_addresses))
    e.state = 'stored'
    branch('restore_n_header',[e.state,offset('n',1),offset('ytile',2048),
                               offset('restore_rpage',pages[1][1])],rn_types)
    block('restore_m_next',['rm_next_s'],[STATE])
    branch('restore_m_header',[e.state,offset('rm',1),offset('rm_y',nc*2048),
                               offset('restore_lpage',pages[0][1])],rm_types)
    block('pair_next',['pair_next_s'],[STATE])
    branch('pair_header',[e.state,offset('pair',1),offset('pair_a',2*kc*1024),
                          offset('pair_y',2*nc*2048),offset('pair_lpage',2*pages[0][1])],pair_types)
    block('done',['final'],[STATE])
    e.lines.extend(restoration.finish('final'))
    e.lines += ['  }','}']
    return '\n'.join(e.lines)+'\n'
