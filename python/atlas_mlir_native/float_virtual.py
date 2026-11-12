"""Exact float routes using counted tile and lane loops in common virtual IR."""
import os
from .scalar_ssa import ScalarBuilder
from .binding import DRAM_BASE

S='!atlas.virtual_state'
T='!atlas.virtual_bf16'
P='!atlas.virtual_spill'

class Builder(ScalarBuilder):
    def __init__(self,lines):
        super().__init__(lines); self.state='%start'; self.lane_constants=None
    def constant(self,number):
        if self.lane_constants is None:
            return super().constant(number)
        number &= 0xffffffff
        if number not in self.lane_constants:
            self.lane_constants[number]=super().constant(number)
        return self.lane_constants[number]
    def names(self,count):
        result=[f'%scalar{self.serial+i}' for i in range(count)]; self.serial+=count; return result
    def slot(self,name):
        self.lines.append(f'    %{name} = "atlas.virtual_slot"() {{kind = "bf16", words = 512 : i32, slot_word = -1 : i32, purpose = "scratch"}} : () -> {P}')
    def save(self,slot,value):
        self.state=self.value(f'"atlas.virtual_save"({self.state},%{slot},{value}) : ({S},{P},{T}) -> {S}')
    def restore(self,slot):
        state,tile=self.names(2)
        self.lines.append(f'    {state}, {tile} = "atlas.virtual_restore"({self.state},%{slot}) : ({S},{P}) -> ({S},{T})')
        self.state=state; return tile
    def load(self,slot,index,width):
        state,value=self.names(2)
        lane_offset = 0
        if isinstance(index, tuple):
            index, lane_offset = index
        operand='' if isinstance(index,int) else f',{index}'
        type_='' if isinstance(index,int) else ',i32'
        offset=(index if isinstance(index,int) else lane_offset)*width
        self.lines.append(f'    {state}, {value} = "atlas.virtual_slot_load"({self.state},%{slot}{operand}) {{kind = "{"lhu" if width==2 else "lw"}", byte_offset = {offset} : i32}} : ({S},{P}{type_}) -> ({S},i32)')
        self.state=state; return value
    def store(self,slot,index,value,width):
        operand='' if isinstance(index,int) else f',{index}'
        type_='' if isinstance(index,int) else ',i32'
        offset=index*width if isinstance(index,int) else 0
        self.state=self.value(f'"atlas.virtual_slot_store"({self.state},%{slot},{value}{operand}) {{kind = "{"sh" if width==2 else "sw"}", byte_offset = {offset} : i32}} : ({S},{P},i32{type_}) -> {S}')
    def unary_lanes(self,count,accumulators,body):
        """Use immediate slot offsets within one bounded unary tile.

        Constants are shared only within this straight-line block. The outer
        counted DMA loop and the complete preflight/publication barrier stay
        intact; scalar live ranges cannot cross a CFG boundary here.
        """
        assert self.lane_constants is None
        self.lane_constants={}
        try:
            for index in range(count):
                accumulators=body(index,accumulators)
            return accumulators
        finally:
            self.lane_constants=None
    def checked_lane_groups(self, label, count, accumulators, body):
        """Share address calculation and constants across four adjacent lanes."""
        factor = 4
        if count % factor:
            return self.loop(label, count, accumulators, body)

        def group(index, values):
            base = self.binary('shl', index, self.constant(2))
            assert self.lane_constants is None
            self.lane_constants = {}
            try:
                for lane in range(factor):
                    values = body((base, lane), values)
                return values
            finally:
                self.lane_constants = None

        return self.loop(label, count // factor, accumulators, group)

    def splat(self,bits): return self.value(f'"atlas.virtual_vpu_splat"() {{bits = {bits} : i32}} : () -> {T}')
    def vector_binary(self,kind,a,b): return self.value(f'"atlas.virtual_vpu_binary"({a},{b}) {{kind = "{kind}"}} : ({T},{T}) -> {T}')
    def dma_load(self,address,bytes_):
        state,tile=self.names(2);size=self.constant(bytes_)
        self.lines.append(f'    {state}, {tile} = "atlas.virtual_dma_load_bf16"({self.state},{address},{size}) : ({S},i32,i32) -> ({S},{T})')
        self.state=state; return tile
    def dma_store(self,address,tile,bytes_):
        size=self.constant(bytes_)
        self.state=self.value(f'"atlas.virtual_dma_store_bf16"({self.state},{tile},{address},{size}) : ({S},{T},i32,i32) -> {S}')
    def inputs(self,p,index):
        a=self.load('left',index,p.width)
        d=self.load('right',index,p.width) if len(p.operands)>1 else a
        if p.op=='bf16_sub': d=self.binary('xor',d,self.constant(32768))
        return a,d
    def loop(self,label,count,accumulators,body):
        """Emit a guarded counted loop, carrying state and named scalar values."""
        initial=self.constant(0); names=[f'%{label}_acc{i}' for i in range(len(accumulators))]
        types=','.join([S,'i32']+['i32']*len(names))
        self.lines.append(f'    cf.br ^{label}({",".join([self.state,initial,*accumulators])} : {types})')
        args=','.join([f'%{label}_state: {S}',f'%{label}_index: i32']+[f'{n}: i32' for n in names])
        self.lines.append(f'  ^{label}({args}):');self.state=f'%{label}_state'
        more=self.value(f'arith.cmpi ult, %{label}_index, {self.constant(count)} : i32')
        endnames=[f'%{label}_out{i}' for i in range(len(names))]
        endtypes=','.join([S]+['i32']*len(names))
        self.lines.append(f'    cf.cond_br {more}, ^{label}_body({self.state} : {S}), ^{label}_done({",".join([self.state,*names])} : {endtypes})')
        self.lines.append(f'  ^{label}_body(%{label}_token: {S}):');self.state=f'%{label}_token'
        values=body(f'%{label}_index',names)
        next_=self.binary('add',f'%{label}_index',self.constant(1))
        self.lines.append(f'    cf.br ^{label}({",".join([self.state,next_,*values])} : {types})')
        args=','.join([f'%{label}_end: {S}']+[f'{n}: i32' for n in endnames])
        self.lines.append(f'  ^{label}_done({args}):');self.state=f'%{label}_end'
        return endnames


def qualified_vector(b,p):
    """Reuse the existing qualified residual DAG, including its constants."""
    from dataclasses import replace
    from .qualified_rne_add import prepare_source,prepare_plan
    source=p.source
    inputs=source.nodes[:-1]
    if len(inputs)==1:
        inputs=(*inputs,replace(inputs[0],id=inputs[0].id+'_rhs'))
    node=replace(source.nodes[-1],op='bf16_add',inputs=tuple(n.id for n in inputs),attrs=(("numeric_semantics","ieee754-bf16-add-rne-v1"),))
    source=replace(source,nodes=(*inputs,node),input_storages=tuple((n.id,'external') for n in inputs))
    body=prepare_plan(prepare_source(source).request).body_request()
    values={body.nodes[0].id:b.restore('left')}
    right=b.restore('right')
    if p.op=='bf16_sub':right=b.vector_binary('mul',right,b.splat(0xbf80))
    values[body.nodes[1].id]=right
    constants={c.node_id:int.from_bytes(bytes.fromhex(c.data_hex)[:2],'little') for c in body.constants}
    for node in body.nodes:
        if node.effect=='input':continue
        if node.effect=='constant':values[node.id]=b.splat(constants[node.id]);continue
        args=[values[k] for k in node.inputs]
        if len(args)==2:
            kind={'mul_bf16_zero':'mul','add_bf16_rne':'add','sub_bf16_rne':'sub','max_bf16':'max'}[node.op]
            values[node.id]=b.vector_binary(kind,*args)
        else:
            kind={'tanh_bf16':'tanh','recip_bf16_bounded':'recip'}[node.op]
            values[node.id]=b.value(f'"atlas.virtual_vpu_unary"({args[0]}) {{kind = "{kind}"}} : ({T}) -> {T}')
    return values[body.outputs[0]]


def binary_lane_groups():
    """Validate the opt-in before generating or caching target code."""
    from merlin.semantic_compiler.target_binding import NativeCompilationError
    value = os.environ.get('ATLAS_EXACT_BINARY_LANE_GROUPS', '0')
    if value not in ('0', '1'):
        raise NativeCompilationError('unsupported_semantics', 'ATLAS_EXACT_BINARY_LANE_GROUPS must be 0 or 1')
    return value == '1'


def binary_qualified_refusal(b, left, right):
    """The qualified vector domain implies valid input arithmetic."""
    c, op, cmp = b.constant, b.binary, b.compare
    bad = c(0)
    for value in (left, right):
        magnitude = op('and', value, c(32767))
        outside = op('or', op('and', cmp('ult', magnitude, c(0x2f80)),
                             cmp('ne', magnitude, c(0))),
                     cmp('ult', c(0x4f80), magnitude))
        bad = op('or', bad, outside)
    return bad


def binary_slow_flags(b, left, right):
    """After qualified refusal, classify pair admission and invalid inputs.

    The right operand already has its sign flipped for subtraction. Keep
    input validity separate from route admission: rejecting a fast route must
    still permit the exact scalar implementation.
    """
    c, op, cmp = b.constant, b.binary, b.compare
    magnitudes = [op('and', value, c(32767)) for value in (left, right)]
    pair, invalid = c(0), c(0)
    for magnitude in magnitudes:
        nonzero = cmp('ne', magnitude, c(0))
        exceptional = op('or', op('and', cmp('ult', magnitude, c(128)), nonzero),
                         cmp('ule', c(32640), magnitude))
        pair = op('or', pair, exceptional)
        invalid = op('or', invalid, cmp('ult', c(32640), magnitude))
    both_negative_zero = cmp('eq', op('or', op('xor', left, c(32768)),
                                     op('xor', right, c(32768))), c(0))
    pair = op('or', pair, both_negative_zero)
    both_inf = op('and', cmp('eq', magnitudes[0], c(32640)),
                  cmp('eq', magnitudes[1], c(32640)))
    opposite = op('shr', op('xor', left, right), c(15))
    invalid = op('or', invalid, op('and', both_inf, opposite))
    return op('or', c(1), op('or', op('shl', pair, c(1)),
                                 op('shl', invalid, c(2))))


def virtual_program(p):
    from .exact_float_device import _valid,_compute
    from merlin.semantic_compiler.codegen import current_codegen_options
    # Preserve qualified compact images unless expansion is explicitly selected.
    expand_unary=current_codegen_options().exact_float_unroll
    group_binary=binary_lane_groups()
    lines=['module {',f'  func.func @exact_float() -> {S} attributes {{atlas.input_dram_base = {p.io_memory_base} : i64, atlas.input_dram_bytes = {len(p.operands)*p.input_bytes} : i64, atlas.output_dram_base = {p.output_address} : i64, atlas.output_dram_bytes = {p.output_bytes+32} : i64}} {{',f'    %start = "atlas.virtual_start"() : () -> {S}']
    b=Builder(lines);c=b.constant;op=b.binary;cmp=b.compare
    lanes=p.tile_lanes;tiles=p.input_bytes//(lanes*p.width)
    binary=p.op in ('bf16_add','bf16_sub')
    # One word per tile retains route admission across the complete input
    # preflight. Arithmetic starts only after every tile passed validation.
    if binary and tiles > 512:
        raise ValueError('exact binary route table exceeds one 2048-byte slot')
    slots=('left','right','result','pairs','status') + (('routes',) if binary else ())
    for slot in slots:b.slot(slot)
    zero=b.splat(0)
    for slot in slots:b.save(slot,zero)
    flag=c(0)
    phases=('check','compute') if not p.op.endswith('_neg') else ('compute',)
    for phase in phases:
        pointers=[c(p.io_memory_base),c(p.io_memory_base+[n.id for n in p.operands].index(p.node.inputs[-1])*p.input_bytes),c(p.output_address)]
        def tile_body(tile,values):
            bad,ap,bp,out=values
            # A one-tile binary preflight leaves both immutable inputs in VMEM.
            if not (binary and tiles == 1 and phase == 'compute'):
                a=b.dma_load(ap,lanes*p.width);b.save('left',a)
                b.save('right',b.dma_load(bp,lanes*p.width) if len(p.operands)>1 else a)
            if phase=='check':
                if binary:
                    lane_loop=b.checked_lane_groups if group_binary else b.loop
                    def qualified_lane(index,acc):
                        a,d=b.inputs(p,index)
                        return [op('or',acc[0],binary_qualified_refusal(b,a,d))]
                    qbad,=lane_loop('qualified_lane',lanes,[c(0)],qualified_lane)
                    admitted=b.value(f'arith.cmpi eq, {qbad}, {c(0)} : i32')
                    lines.append(f'    cf.cond_br {admitted}, ^binary_checked({b.state},{c(0)} : {S},i32), ^binary_check_slow({b.state} : {S})')
                    lines.append(f'  ^binary_check_slow(%binary_slow_state: {S}):');b.state='%binary_slow_state'
                    def slow_lane(index,acc):
                        a,d=b.inputs(p,index)
                        return [op('or',acc[0],binary_slow_flags(b,a,d))]
                    flags,=lane_loop('preflight_lane',lanes,[c(0)],slow_lane)
                    lines.append(f'    cf.br ^binary_checked({b.state},{flags} : {S},i32)')
                    lines.append(f'  ^binary_checked(%binary_checked_state: {S},%binary_flags: i32):')
                    b.state='%binary_checked_state';flags='%binary_flags'
                    b.store('routes',tile,flags,4)
                    bad=op('or',bad,op('shr',flags,c(2)))
                def validate(index,acc):
                    a,d=b.inputs(p,index)
                    return [op('or',acc[0],_valid(b,p,a,d))]
                if not binary:
                    bad,=(b.loop('preflight_lane',lanes,[bad],validate) if not expand_unary
                          else b.unary_lanes(lanes,[bad],validate))
            else:
                if binary:
                    flags=b.load('routes',tile,4)
                    qbad,rbad=op('and',flags,c(1)),op('and',flags,c(2))
                    qualified=b.value(f'arith.cmpi eq, {qbad}, {c(0)} : i32')
                    lines.append(f'    cf.cond_br {qualified}, ^qualified({b.state} : {S}), ^pair_test({b.state} : {S})')
                    lines.append(f'  ^qualified(%qualified_state: {S}):');b.state='%qualified_state'
                    b.save('result',qualified_vector(b,p));lines.append(f'    cf.br ^publish({b.state} : {S})')
                    lines.append(f'  ^pair_test(%pair_test_state: {S}):');b.state='%pair_test_state'
                    pair_ok=b.value(f'arith.cmpi eq, {rbad}, {c(0)} : i32')
                    lines.append(f'    cf.cond_br {pair_ok}, ^pair_begin({b.state} : {S}), ^scalar({b.state} : {S})')
                    lines.append(f'  ^pair_begin(%pair_begin_state: {S}):');b.state='%pair_begin_state'
                    def pair_group(group,acc):
                        base=op('shl',group,c(5));b.save('pairs',b.splat(0))
                        def gather(index,unused):
                            a,d=b.inputs(p,op('add',base,index));pair_index=op('shl',index,c(4))
                            b.store('pairs',pair_index,a,2);b.store('pairs',op('add',pair_index,c(1)),d,2)
                            return []
                        b.loop('pair_gather',32,[],gather)
                        data=b.restore('pairs');sums=b.value(f'"atlas.virtual_vpu_reduce"({data}) {{kind = "row_sum"}} : ({T}) -> {T}');b.save('pairs',sums)
                        def scatter(index,sticky):
                            logical=op('add',base,index);a,d=b.inputs(p,logical)
                            result=b.load('pairs',op('shl',index,c(4)),2);b.store('result',logical,result,2)
                            magnitude=op('and',result,c(32767));zero=cmp('eq',magnitude,c(0))
                            exceptional=op('or',op('and',cmp('ult',magnitude,c(128)),cmp('ne',magnitude,c(0))),cmp('ule',c(32640),magnitude))
                            am=op('and',a,c(32767));dm=op('and',d,c(32767))
                            exact_zero=op('and',cmp('eq',am,dm),op('or',op('shr',op('xor',a,d),c(15)),cmp('eq',am,c(0))))
                            invalid_zero=op('and',zero,op('xor',exact_zero,c(1)))
                            return [op('or',sticky[0],op('or',exceptional,invalid_zero))]
                        return b.loop('pair_scatter',32,acc,scatter)
                    postbad,=b.loop('pair_group',lanes//32,[c(0)],pair_group)
                    accepted=b.value(f'arith.cmpi eq, {postbad}, {c(0)} : i32')
                    lines.append(f'    cf.cond_br {accepted}, ^publish({b.state} : {S}), ^scalar({b.state} : {S})')
                    lines.append(f'  ^scalar(%scalar_state: {S}):');b.state='%scalar_state'
                def calculate(index,unused):
                    a,d=b.inputs(p,index);b.store('result',index,_compute(b,p,a,d),p.output_width)
                    return []
                if binary or not expand_unary:
                    b.loop('scalar_lane',lanes,[],calculate)
                else:
                    b.unary_lanes(lanes,[],calculate)
                if binary:
                    lines.append(f'    cf.br ^publish({b.state} : {S})')
                    lines.append(f'  ^publish(%publish_state: {S}):');b.state='%publish_state'
                b.dma_store(out,b.restore('result'),lanes*p.output_width)
            return [bad,op('add',ap,c(lanes*p.width)),op('add',bp,c(lanes*p.width)),op('add',out,c(lanes*p.output_width))]
        flag,*_=b.loop(phase+'_tile',tiles,[flag,*pointers],tile_body)
        if phase=='check':
            accepted=b.value(f'arith.cmpi eq, {flag}, {c(0)} : i32')
            lines.append(f'    cf.cond_br {accepted}, ^accepted({b.state} : {S}), ^done({b.state},{flag} : {S},i32)')
            lines.append(f'  ^accepted(%accepted_state: {S}):');b.state='%accepted_state'
    lines.append(f'    cf.br ^done({b.state},{flag} : {S},i32)')
    lines.append(f'  ^done(%done_state: {S},%status_code: i32):');b.state='%done_state'
    b.store('status',c(0),'%status_code',4);b.dma_store(c(p.status_address),b.restore('status'),32)
    lines.extend((f'    return {b.state} : {S}','  }','}'))
    return '\n'.join(lines)+'\n'
