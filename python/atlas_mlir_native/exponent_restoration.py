"""Exact signed-exponent readout in typed virtual operations.

Scalar accesses address words modulo their complete, initialized BF16 slot.
Counted tile/word loops retain fixed addresses and transfer sizes. A certified
exponent guard selects exact VPU scaling when all intermediate results remain
normal. Small images retain full scalar admission; larger images explicitly
refuse exponents outside the vector domain to bound every possible launch.
"""
from math import prod

STATE = '!atlas.virtual_state'
BF16 = '!atlas.virtual_bf16'
SLOT = '!atlas.virtual_spill'



def fast_factor_range(plan):
    if getattr(plan, 'epilogue_precision', None) is None:
        return -50, 47
    spec = plan.epilogue_precision.operand_specs[plan.epilogue_contract_operand]
    expected = ((0,0),(0x2000,0x5f00),(0x8000,0x8000),(0xa000,0xdf00))
    if tuple(spec.admitted_code_ranges) != expected:
        raise ValueError('vector restoration requires the selected bounded BF16 product domain')
    # The proved raw exponent interval [-26,32] becomes [-62,62]. This is
    # strictly within the original product operand's normal [-63,63] interval.
    lower, upper = -18, 15
    assert -26+2*lower >= -63 and 32+2*upper < 63
    return lower, upper

def exponent_geometry(plan):
    result = []
    for key, operand_key in zip(plan.exponent_ids, plan.operand_ids, strict=True):
        operand = plan.source.node(operand_key)
        shape = plan.source.node(key).type.shape
        batches = prod(operand.type.shape[:-2])
        rows = operand.type.shape[-2]
        mode = 'scalar' if not shape else 'batch' if shape[-2:] == (1, 1) else 'row'
        tiles = (rows + 31) // 32 if mode == 'row' else 1
        pages = 1 if mode == 'scalar' else batches * tiles
        result.append(dict(mode=mode, batches=batches, rows=rows, tiles=tiles,
                           pages=pages, bytes=pages * 2048))
    return result


class Emitter:
    def __init__(self, prefix):
        self.prefix = prefix
        self.lines = []
        self.serial = 0
        self.state = None

    def name(self, stem):
        self.serial += 1
        return f'{self.prefix}_{stem}_{self.serial}'

    def emit(self, text):
        self.lines.append('    ' + text)

    def const(self, value):
        name = self.name('c')
        if value >= 1 << 31:
            value -= 1 << 32
        self.emit(f'%{name} = arith.constant {value} : i32')
        return name

    def binary(self, op, lhs, rhs, dtype='i32'):
        name = self.name(op)
        self.emit(f'%{name} = arith.{op} %{lhs}, %{rhs} : {dtype}')
        return name

    def cmp(self, predicate, lhs, rhs):
        name = self.name('cmp')
        self.emit(f'%{name} = arith.cmpi {predicate}, %{lhs}, %{rhs} : i32')
        return name

    def flag(self, condition):
        name = self.name('flag')
        self.emit(f'%{name} = arith.extui %{condition} : i1 to i32')
        return name

    def combine(self, op, *conditions):
        value = conditions[0]
        for condition in conditions[1:]:
            value = self.binary(op, value, condition, 'i1')
        return value

    def effect(self, op, args, types, result_type=None, attrs=""):
        state = self.name('s')
        value = self.name('v') if result_type else None
        lhs = f'%{state}, %{value}' if result_type else f'%{state}'
        ret = f'({STATE}, {result_type})' if result_type else STATE
        self.emit(f'{lhs} = "atlas.virtual_{op}"(%{self.state}, ' + ', '.join('%'+v for v in args) +
                  f'){attrs} : ({STATE}, ' + ', '.join(types) + f') -> {ret}')
        self.state = state
        return value

    def load(self, slot, index):
        index = self.binary('andi', index, self.const(511))
        return self.effect('slot_load', [slot, index], [SLOT, 'i32'], 'i32', ' {kind = \"lw\", byte_offset = 0 : i32}')

    def store(self, slot, index, value):
        index = self.binary('andi', index, self.const(511))
        self.effect('slot_store', [slot, value, index], [SLOT, 'i32', 'i32'], attrs=' {kind = \"sw\", byte_offset = 0 : i32}')


def _entry():
    return [
        f'    %restore_data = "atlas.virtual_slot"() {{kind = "bf16", words = 512 : i32, slot_word = -1 : i32, purpose = "scratch", atlas.exponent_scratch = "data"}} : () -> {SLOT}',
        f'    %restore_lhs = "atlas.virtual_slot"() {{kind = "bf16", words = 512 : i32, slot_word = -1 : i32, purpose = "scratch", atlas.exponent_scratch = "lhs"}} : () -> {SLOT}',
        f'    %restore_rhs = "atlas.virtual_slot"() {{kind = "bf16", words = 512 : i32, slot_word = -1 : i32, purpose = "scratch", atlas.exponent_scratch = "rhs"}} : () -> {SLOT}',
        f'    %restore_status = "atlas.virtual_slot"() {{kind = "bf16", words = 512 : i32, slot_word = -1 : i32, purpose = "scratch", atlas.exponent_scratch = "status"}} : () -> {SLOT}',
        f'    %restore_zero = "atlas.virtual_vpu_splat"() {{bits = 0 : i32}} : () -> {BF16}',
        f'    %restore_start = "atlas.virtual_save"(%start, %restore_status, %restore_zero) : ({STATE}, {SLOT}, {BF16}) -> {STATE}',
    ]


def _tile(plan, layout, *, output_address="ytile", exponent_addresses=("restore_lpage", "restore_rpage"), input_state="ss", output_state="stored", logical_rows=True):
    """Replace a completed tile store with restoration followed by that store."""
    e = Emitter('restore')
    epilogue = getattr(plan, 'epilogue_precision', None)
    if epilogue is not None:
        from .device_boundary import _admitted, _boolean
        from .pointwise import prepare_plan
        epilogue_plan = prepare_plan(epilogue.request)
        contract_operand = plan.epilogue_contract_operand
        if (epilogue.request.nodes[-1].op not in {'mul_bf16_zero','mul_bf16_finite'}
                or epilogue_plan.input_pad(contract_operand) != b'\0\0'):
            raise ValueError('quantized epilogue requires the unchanged pointwise product with positive-zero padding')
        epilogue_spec = epilogue.operand_specs[contract_operand]
    e.state = input_state
    zero, one = e.const(0), e.const(1)
    mt = 'm' if (plan.m + 31)//32 > 1 else zero
    nt = 'n' if (plan.n + 31)//32 > 1 else zero
    e.effect('save', ['restore_data', 'result'], [SLOT, BF16])
    for which, address in zip(('lhs', 'rhs'), exponent_addresses, strict=True):
        value = e.effect('dma_load_bf16', [address, e.const(2048)], ['i32', 'i32'], BF16)
        e.effect('save', ['restore_'+which, value], [SLOT, BF16])
    invalid = e.load('restore_status', zero)
    entry_state = e.state
    e.emit(f'cf.br ^restore_word_header(%{entry_state}, %{zero}, %{invalid} : {STATE}, i32, i32)')
    e.lines.append(f'  ^restore_word_header(%restore_ws: {STATE}, %restore_i: i32, %restore_invalid: i32):')
    scan_rows = min(32, 1 << (min(plan.m, 32)-1).bit_length()) if logical_rows else 32
    limit = e.const(scan_rows * 16)
    more = e.cmp('ult', 'restore_i', limit)
    e.emit(f'cf.cond_br %{more}, ^restore_word_body(%restore_ws : {STATE}), ^restore_word_done(%restore_ws, %restore_invalid : {STATE}, i32)')
    e.lines.append(f'  ^restore_word_body(%restore_wb: {STATE}):')
    e.state = 'restore_wb'
    index = 'restore_i'
    if scan_rows < 32:
        low_bits = (scan_rows * 8).bit_length()-1
        index = e.binary('ori', e.binary('andi', 'restore_i', e.const(scan_rows*8-1)),
                         e.binary('shli', e.binary('shrui', 'restore_i', e.const(low_bits)), e.const(8)))
    packed = e.load('restore_data', index)
    restored = []
    invalid = 'restore_invalid'
    lhs_geometry, rhs_geometry = exponent_geometry(plan)
    for lane in range(2):
        # Finish each admission predicate promptly, keeping only its accumulated
        # status. Rematerialize coordinates in each lane to bound scalar liveness.
        word = e.binary('andi', packed if lane == 0 else e.binary('shrui', packed, e.const(16)), e.const(65535))
        exponent = e.binary('andi', e.binary('shrui', word, e.const(7)), e.const(255))
        nonzero = e.cmp('ne', word, zero)
        raw_bad = e.combine('andi', nonzero, e.combine('ori', e.cmp('eq', exponent, zero), e.cmp('eq', exponent, e.const(255))))
        if epilogue is None:
            invalid = e.binary('ori', invalid, e.flag(raw_bad))
        else:
            lane_invalid = e.flag(raw_bad)
        row = e.binary('shrui', e.binary('andi', index, e.const(255)), e.const(3))
        column = e.binary('addi', e.binary('shli', e.binary('andi', index, e.const(7)), one),
                          e.binary('shli', e.binary('shrui', index, e.const(8)), e.const(4)))
        if lane:
            column = e.binary('addi', column, one)
        global_row = e.binary('addi', e.binary('shli', mt, e.const(5)), row)
        global_column = e.binary('addi', e.binary('shli', nt, e.const(5)), column)
        valid = e.combine('andi', e.cmp('ult', global_row, e.const(plan.m)), e.cmp('ult', global_column, e.const(plan.n)))
        lhs = e.load('restore_lhs', row if lhs_geometry['mode'] == 'row' else zero)
        rhs = e.load('restore_rhs', column if rhs_geometry['mode'] == 'row' else zero)
        for exponent_input in (lhs, rhs):
            exponent_bad = e.combine('ori', e.cmp('slt', exponent_input, e.const(-127)), e.cmp('sgt', exponent_input, e.const(127)))
            if epilogue is None:
                invalid = e.binary('ori', invalid, e.flag(e.combine('andi', valid, exponent_bad)))
            else:
                lane_invalid = e.binary('ori', lane_invalid, e.flag(e.combine('andi', valid, exponent_bad)))
        scale = e.binary('addi', lhs, rhs)
        scale_bad = e.combine('ori', e.cmp('slt', scale, e.const(-133)), e.cmp('sgt', scale, e.const(127)))
        if epilogue is None:
            invalid = e.binary('ori', invalid, e.flag(e.combine('andi', valid, scale_bad)))
        else:
            lane_invalid = e.binary('ori', lane_invalid, e.flag(e.combine('andi', valid, scale_bad)))
        target_exponent = e.binary('addi', exponent, scale)
        target_bad = e.combine('andi', valid, nonzero, e.combine('ori', e.cmp('sle', target_exponent, zero), e.cmp('sge', target_exponent, e.const(255))))
        if epilogue is None:
            invalid = e.binary('ori', invalid, e.flag(target_bad))
        else:
            lane_invalid = e.binary('ori', lane_invalid, e.flag(target_bad))
        mask = e.binary('subi', zero, e.flag(nonzero))
        change = e.binary('andi', mask, e.binary('shli', scale, e.const(7)))
        code = e.binary('andi', e.binary('addi', word, change), e.const(65535))
        if epilogue is not None:
            if epilogue_spec.canonicalize_negative_zero:
                keep = e.binary('subi',zero,e.flag(e.cmp('ne',code,e.const(0x8000))))
                code = e.binary('andi',code,keep)
            ranges = epilogue_spec.admitted_code_ranges
            # The ordinary pointwise product has a sign-symmetric code set.
            # Check exactly that set before using a magnitude comparison; no
            # new numerical domain or static preimage is introduced.
            if tuple(ranges) == ((0,0),(0x2000,0x5f00),(0x8000,0x8000),(0xa000,0xdf00)):
                magnitude = e.binary('andi',code,e.const(0x7fff))
                admitted = e.combine('ori',e.cmp('eq',magnitude,zero),
                    e.combine('andi',e.cmp('uge',magnitude,e.const(0x2000)),e.cmp('ule',magnitude,e.const(0x5f00))))
            else:
                admitted = _admitted(e,code,ranges)
            bad = e.combine('andi',valid,e.binary('xori',admitted,_boolean(e,True),'i1'))
            lane_invalid = e.binary('ori',lane_invalid,e.flag(bad))
            invalid = e.binary('ori',invalid,lane_invalid)
            # Rejection never allows a subsequent selected VPU body to read
            # an unadmitted code, including an accidentally normal wrapped
            # result of an invalid exponent. Nonlogical lanes become +0 too.
            keep = e.combine('andi',valid,admitted,e.cmp('eq',lane_invalid,zero))
            code = e.binary('andi',code,e.binary('subi',zero,e.flag(keep)))
        restored.append(code)
    result = e.binary('ori', restored[0], e.binary('shli', restored[1], e.const(16)))
    e.store('restore_data', index, result)
    next_i = e.binary('addi', 'restore_i', one)
    e.emit(f'cf.br ^restore_word_header(%{e.state}, %{next_i}, %{invalid} : {STATE}, i32, i32)')
    e.lines.append(f'  ^restore_word_done(%restore_wd: {STATE}, %restore_bad: i32):')
    e.state = 'restore_wd'
    e.store('restore_status', zero, 'restore_bad')
    tile = e.effect('restore', ['restore_data'], [SLOT], BF16)
    _store_completed(e, tile, plan, layout, output_address, output_state)
    return e.lines


def _store_completed(e, tile, plan, layout, output_address, output_state):
    epilogue = getattr(plan, 'epilogue_precision', None)
    if epilogue is not None:
        address = e.binary('addi',output_address,e.const(layout['output_bytes']+layout['status_bytes']))
        other = e.effect('dma_load_bf16',[address,e.const(2048)],['i32','i32'],BF16)
        operands = (tile,other) if plan.epilogue_contract_operand == 0 else (other,tile)
        product = e.name('epilogue_product')
        e.emit(f'%{product} = "atlas.virtual_vpu_binary"(%{operands[0]}, %{operands[1]}) {{kind = "mul"}} : ({BF16}, {BF16}) -> {BF16}')
        tile = product
    e.emit(f'%{output_state} = "atlas.virtual_dma_store_bf16"(%{e.state}, %{tile}, %{output_address}, %out_bytes) : ({STATE}, {BF16}, i32, i32) -> {STATE}')


def _scratch(name):
    return f'    %{name} = "atlas.virtual_slot"() {{kind = "bf16", words = 512 : i32, slot_word = -1 : i32, purpose = "scratch"}} : () -> {SLOT}'


def _vector_entry():
    """Create constant half masks once, using a fixed initialized tile."""
    lines = _entry()
    # Initialize masks before entering the contraction loops. The mask loop
    # depends only on its 256-word bound and writes initialized scratch.
    lines[-1] = f'    %restore_status_start = "atlas.virtual_save"(%start, %restore_status, %restore_zero) : ({STATE}, {SLOT}, {BF16}) -> {STATE}'
    lines += [_scratch(name) for name in ('restore_mask_left', 'restore_mask_right',
                                         'restore_factor_lhs', 'restore_factor_rhs0', 'restore_factor_rhs1')]
    e = Emitter('restore_mask')
    e.state = 'restore_status_start'
    e.emit(f'%restore_ones = "atlas.virtual_vpu_splat"() {{bits = 16256 : i32}} : () -> {BF16}')
    e.effect('save',['restore_mask_left','restore_ones'],[SLOT,BF16])
    e.emit(f'cf.br ^restore_mask_header(%{e.state}, %{e.const(256)} : {STATE}, i32)')
    e.lines.append(f'  ^restore_mask_header(%restore_mask_s: {STATE}, %restore_mask_i: i32):')
    more=e.cmp('ult','restore_mask_i',e.const(512))
    e.emit(f'cf.cond_br %{more}, ^restore_mask_body(%restore_mask_s : {STATE}), ^restore_mask_done(%restore_mask_s : {STATE})')
    e.lines.append(f'  ^restore_mask_body(%restore_mask_b: {STATE}):')
    e.state='restore_mask_b'
    e.store('restore_mask_left','restore_mask_i',e.const(0))
    nxt=e.binary('addi','restore_mask_i',e.const(1))
    e.emit(f'cf.br ^restore_mask_header(%{e.state}, %{nxt} : {STATE}, i32)')
    e.lines.append(f'  ^restore_mask_done(%restore_mask_d: {STATE}):')
    e.state='restore_mask_d'
    left=e.effect('restore',['restore_mask_left'],[SLOT],BF16)
    right=e.name('right')
    e.emit(f'%{right} = "atlas.virtual_vpu_binary"(%restore_ones, %{left}) {{kind = "sub"}} : ({BF16}, {BF16}) -> {BF16}')
    e.effect('save',['restore_mask_right',right],[SLOT,BF16])
    e.emit(f'cf.br ^restore_ready(%{e.state} : {STATE})')
    e.lines.append(f'  ^restore_ready(%restore_start: {STATE}):')
    return lines+e.lines


def _vector_tile(plan, layout, *, output_address, exponent_addresses, scalar_fallback=True):
    """Build separable exact factors and choose the proven normal fast path."""
    e=Emitter('restore_factor')
    e.state='ss'
    for which,address in zip(('lhs','rhs'),exponent_addresses,strict=True):
        value=e.effect('dma_load_bf16',[address,e.const(2048)],['i32','i32'],BF16)
        e.effect('save',['restore_'+which,value],[SLOT,BF16])
    zero_tile=e.name('zero')
    e.emit(f'%{zero_tile} = "atlas.virtual_vpu_splat"() {{bits = 0 : i32}} : () -> {BF16}')
    for slot in ('restore_factor_lhs','restore_factor_rhs0','restore_factor_rhs1'):
        e.effect('save',[slot,zero_tile],[SLOT,BF16])
    geometries=exponent_geometry(plan)
    lower,upper=fast_factor_range(plan)
    zero=e.const(0)
    initial_bad=zero
    hoisted={}
    def factor(which,index,bad):
        exponent=e.load('restore_'+which,index)
        rejected=e.combine('ori',e.cmp('slt',exponent,e.const(lower)),e.cmp('sgt',exponent,e.const(upper)))
        bad=e.binary('ori',bad,e.flag(rejected))
        bits=e.binary('shli',e.binary('addi',exponent,e.const(127)),e.const(7))
        return bits,bad
    # Scalar and per-batch exponent pages have one value. Admit and encode it
    # once per tile instead of repeating the same load for every row/column.
    for which,geometry in zip(('lhs','rhs'),geometries,strict=True):
        if geometry['mode']!='row':
            hoisted[which],initial_bad=factor(which,zero,initial_bad)
    for which,bits in hoisted.items():
        # col_sum repeats the sole sparse value in both physical halves.
        # row_sum then adds those two equal powers, so seed with 2^(s-1).
        half=e.binary('subi',bits,e.const(128))
        e.store('restore_factor_lhs' if which=='lhs' else 'restore_factor_rhs0',zero,half)
    e.emit(f'cf.br ^restore_factor_header(%{e.state}, %{zero}, %{initial_bad} : {STATE}, i32, i32)')
    e.lines.append(f'  ^restore_factor_header(%restore_factor_s: {STATE}, %restore_factor_i: i32, %restore_factor_bad: i32):')
    more=e.cmp('ult','restore_factor_i',e.const(8))
    e.emit(f'cf.cond_br %{more}, ^restore_factor_body(%restore_factor_s : {STATE}), ^restore_factor_done(%restore_factor_s, %restore_factor_bad : {STATE}, i32)')
    e.lines.append(f'  ^restore_factor_body(%restore_factor_b: {STATE}):')
    e.state='restore_factor_b'
    bad='restore_factor_bad'
    # Eight iterations fill all 32 row factors and both groups of sixteen
    # column factors. Each packed RHS word is written exactly once.
    if 'lhs' not in hoisted:
        row_base=e.binary('shli','restore_factor_i',e.const(2))
        for offset in range(4):
            row=row_base if offset==0 else e.binary('addi',row_base,e.const(offset))
            bits,bad=factor('lhs',row,bad)
            e.store('restore_factor_lhs',e.binary('shli',row,e.const(3)),bits)
    if 'rhs' not in hoisted:
        column_base=e.binary('shli','restore_factor_i',e.const(1))
        for group in range(2):
            column=column_base if group==0 else e.binary('addi',column_base,e.const(16))
            first,bad=factor('rhs',column,bad)
            second,bad=factor('rhs',e.binary('addi',column,e.const(1)),bad)
            packed=e.binary('ori',first,e.binary('shli',second,e.const(16)))
            e.store('restore_factor_rhs'+str(group),'restore_factor_i',packed)
    nxt=e.binary('addi','restore_factor_i',e.const(1))
    e.emit(f'cf.br ^restore_factor_header(%{e.state}, %{nxt}, %{bad} : {STATE}, i32, i32)')
    e.lines.append(f'  ^restore_factor_done(%restore_factor_d: {STATE}, %restore_factor_rejected: i32):')
    # This sole exponent-dependent branch is certified by the fixed eight-step
    # factor loop. Every DMA address/size is independent of the predicate.
    safe=e.cmp('eq','restore_factor_rejected',e.const(0))
    e.emit(f'cf.cond_br %{safe}, ^restore_vector(%restore_factor_d : {STATE}), ^restore_scalar(%restore_factor_d : {STATE}) {{atlas.exponent_restoration_guard = "normal_pow2_v1", atlas.exponent_lower = {lower} : i32, atlas.exponent_upper = {upper} : i32}}')
    e.lines.append(f'  ^restore_vector(%restore_vector_s: {STATE}):')
    e.state='restore_vector_s'
    def unary(kind,value):
        result=e.name(kind)
        e.emit(f'%{result} = "atlas.virtual_vpu_reduce"(%{value}) {{kind = "{kind}"}} : ({BF16}) -> {BF16}')
        return result
    def binary(kind,lhs,rhs):
        result=e.name(kind)
        e.emit(f'%{result} = "atlas.virtual_vpu_binary"(%{lhs}, %{rhs}) {{kind = "{kind}"}} : ({BF16}, {BF16}) -> {BF16}')
        return result
    def scalar_broadcast(slot):
        return unary('row_sum',unary('col_sum',e.effect('restore',[slot],[SLOT],BF16)))
    lhs=(scalar_broadcast('restore_factor_lhs') if 'lhs' in hoisted else
         unary('row_sum',e.effect('restore',['restore_factor_lhs'],[SLOT],BF16)))
    if 'rhs' in hoisted:
        rhs=scalar_broadcast('restore_factor_rhs0')
    else:
        right=[]
        for slot,mask in (('restore_factor_rhs0','restore_mask_left'),('restore_factor_rhs1','restore_mask_right')):
            factors=unary('col_sum',e.effect('restore',[slot],[SLOT],BF16))
            right.append(binary('mul',factors,e.effect('restore',[mask],[SLOT],BF16)))
        rhs=binary('add',*right)
    tile=binary('mul',binary('mul','result',lhs),rhs)
    _store_completed(e,tile,plan,layout,output_address,'restore_vector_stored')
    e.emit(f'cf.br ^restore_join(%restore_vector_stored : {STATE})')
    e.lines.append(f'  ^restore_scalar(%restore_scalar_s: {STATE}):')
    if scalar_fallback:
        e.lines += _tile(plan,layout,output_address=output_address,exponent_addresses=exponent_addresses,
                         input_state='restore_scalar_s',output_state='restore_scalar_stored')
    else:
        # The successful numerical domain is explicitly narrowed in the
        # execution plan. Never publish a value for an unavailable fallback:
        # status is sticky and the C boundary checks it before reading output.
        # Zero is admitted by the optional product, including all padded lanes.
        e.state='restore_scalar_s'
        e.store('restore_status',e.const(0),e.const(1))
        _store_completed(e,zero_tile,plan,layout,output_address,'restore_scalar_stored')
    e.emit(f'cf.br ^restore_join(%restore_scalar_stored : {STATE})')
    e.lines.append(f'  ^restore_join(%stored: {STATE}):')
    return e.lines


class Restoration:
    """Typed readout hook used by the counted-loop emitter."""
    def __init__(self, plan, layout, *, vector=True, logical_rows=True, scalar_fallback=True):
        self.plan, self.layout = plan, layout
        self.vector, self.logical_rows = vector, logical_rows
        self.scalar_fallback = scalar_fallback

    def pages(self):
        return tuple((address, 2048 if geometry['mode'] == 'row' else 0,
                      geometry['tiles'] * 2048 if geometry['mode'] != 'scalar' and geometry['batches'] > 1 else 0)
                     for address, geometry in zip(self.layout['exponent_addresses'], exponent_geometry(self.plan), strict=True))

    def entry(self):
        return _vector_entry() if self.vector else _entry()

    def tile(self, **kwargs):
        if self.vector:
            return _vector_tile(self.plan, self.layout, scalar_fallback=self.scalar_fallback, **kwargs)
        return _tile(self.plan, self.layout, logical_rows=self.logical_rows, **kwargs)

    def finish(self, state):
        e = Emitter('restore_finish')
        e.state = state
        status = e.effect('restore', ['restore_status'], [SLOT], BF16)
        e.effect('dma_store_bf16', [status, e.const(self.layout['status_address']), e.const(2048)], [BF16, 'i32', 'i32'])
        e.emit(f'return %{e.state} : {STATE}')
        return e.lines
