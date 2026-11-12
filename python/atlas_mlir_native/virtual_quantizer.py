"""Counted full-tile quantization through the ordinary virtual Atlas IR.

A launch owns complete reduction groups. Vector extrema and PACK handle 1,024
lanes at a time; exceptional tiles use an indexed, counted finite-only scan.
The standard compiler owns resource placement, waits, scheduling and contracts.
"""
from math import gcd

from .scalar_ssa import ScalarBuilder

STATE = '!atlas.virtual_state'
BF16 = '!atlas.virtual_bf16'
FP8 = '!atlas.virtual_fp8'
SLOT = '!atlas.virtual_spill'


class Builder(ScalarBuilder):
    def __init__(self):
        super().__init__([])

    def emit(self, text):
        self.lines.append('    ' + text)

    def op(self, name, operands, types, results, attrs=''):
        names = [f'%quant{self.serial + index}' for index in range(len(results))]
        self.serial += len(results)
        prefix = ', '.join(names) + ' = ' if names else ''
        attributes = ' {' + attrs + '}' if attrs else ''
        result_type = results[0] if len(results) == 1 else '(' + ', '.join(results) + ')'
        self.emit(f'{prefix}"atlas.{name}"({", ".join(operands)}){attributes} : ({", ".join(types)}) -> {result_type}')
        return names[0] if len(names) == 1 else names

    def cmp(self, pred, lhs, rhs):
        return self.value(f'arith.cmpi {pred}, {lhs}, {rhs} : i32', 'i1')

    def slot(self, kind, words):
        return self.op('virtual_slot', [], [], [SLOT],
                       f'kind = "{kind}", words = {words} : i32, slot_word = -1 : i32, purpose = "scratch"')

    def save(self, state, slot, value, dtype):
        return self.op('virtual_save', [state, slot, value], [STATE, SLOT, dtype], [STATE])

    def load(self, state, slot, index=None):
        operands, types = [state, slot], [STATE, SLOT]
        if index is not None:
            operands.append(index)
            types.append('i32')
        return self.op('virtual_slot_load', operands, types, [STATE, 'i32'],
                       'kind = "lhu", byte_offset = 0 : i32')


def exponent(b, maximum, rule):
    """Select the authored signed exponent, preserving finite subnormals."""
    zero = b.constant(0)
    encoded = b.binary('shr', maximum, b.constant(7))
    normal = b.binary('sub', encoded, b.constant(135))
    highest, residual = zero, maximum
    for bits in (4, 2, 1):
        high = b.compare('uge', residual, b.constant(1 << bits))
        residual = b.select(b.binary('shr', residual, b.constant(bits)), residual, high)
        highest = b.binary('add', highest, b.select(b.constant(bits), zero, high))
    selected = b.select(normal, b.binary('sub', highest, b.constant(141)), b.compare('ne', encoded, zero))
    nonzero = b.compare('ne', maximum, zero)
    selected = b.select(selected, zero, nonzero)
    if rule == 'nc':
        normal_bump = b.compare('uge', b.binary('and', maximum, b.constant(127)), b.constant(97))
        # The finite subnormal branch uses only highest bits 0..6; mask the
        # speculative normal-path value too so every SSA shift is proven.
        power = b.binary('shl', b.constant(1), b.binary('and', highest, b.constant(7)))
        threshold = b.binary('sub', b.binary('shl', power, b.constant(3)), power)
        subnormal_bump = b.compare('ult', threshold, b.binary('shl', maximum, b.constant(2)))
        bump = b.select(normal_bump, subnormal_bump, b.compare('ne', encoded, zero))
        selected = b.binary('add', selected, b.binary('and', bump, nonzero))
    clamped = b.select(b.constant(-127), selected, b.compare('slt', selected, b.constant(-127)))
    clamped = b.select(b.constant(127), clamped, b.compare('slt', b.constant(127), clamped))
    return selected, clamped


def program(p):
    if p.width != 2 or any(n % 32 for n in (p.count, p.group_elements)) or p.count % p.group_elements:
        raise ValueError('ordinary virtual quantizer body requires complete aligned BF16 groups')
    b = Builder()
    state = b.op('virtual_start', [], [], [STATE])
    # Local constants keep outer-loop invariants out of the counted lane
    # loop's live set; hoisting them caused scalar spills above the 1M cap.
    c, add = b.constant, lambda lhs, rhs: b.binary('add', lhs, rhs)
    groups = p.count // p.group_elements
    # Smaller aligned groups use one prefix tile; non-full tails retain an
    # integral counted tile size, while authored full groups use 1,024 lanes.
    lanes = gcd(1024, p.group_elements)
    scratch, exponent_slot = b.slot('bf16', 512), b.slot('fp8', 256)
    exponent_zero = b.op('virtual_vpu_splat', [], [], [BF16], 'bits = 0 : i32')
    zeros = b.op('virtual_raw_pack_fp8', [exponent_zero, c(127)], [BF16, 'i32'], [FP8])
    state = b.save(state, exponent_slot, zeros, FP8)
    b.emit(f'cf.br ^group({state}, {c(0)}, {c(p.io_memory_base)}, {c(p.code_address)} : {STATE}, i32, i32, i32)')
    b.emit(f'^group(%gs: {STATE}, %gi: i32, %gip: i32, %gop: i32):')
    more = b.cmp('ult', '%gi', c(groups))
    b.emit(f'cf.cond_br {more}, ^group_body(%gs : {STATE}), ^finish(%gs : {STATE})')
    b.emit(f'^group_body(%gbs: {STATE}):')
    if p.attrs['mode'] == 'dynamic':
        b.emit(f'cf.br ^amax(%gbs, {c(0)}, {c(0)}, %gip : {STATE}, i32, i32, i32)')
        b.emit(f'^amax(%as: {STATE}, %ai: i32, %am: i32, %aptr: i32):')
        more = b.cmp('ult', '%ai', c(p.group_elements // lanes))
        b.emit(f'cf.cond_br {more}, ^amax_body(%as : {STATE}), ^scale(%as, %am : {STATE}, i32)')
        b.emit(f'^amax_body(%abs: {STATE}):')
        ds, tile = b.op('virtual_dma_load_bf16', ['%abs', '%aptr', c(lanes * 2)], [STATE, 'i32', 'i32'], [STATE, BF16])
        maxrow = b.op('virtual_vpu_reduce', [tile], [BF16], [BF16], 'kind = "row_max"')
        maxtile = b.op('virtual_vpu_reduce', [maxrow], [BF16], [BF16], 'kind = "col_max"')
        ds = b.save(ds, scratch, maxtile, BF16)
        ds, vmax = b.load(ds, scratch)
        minrow = b.op('virtual_vpu_reduce', [tile], [BF16], [BF16], 'kind = "row_min"')
        mintile = b.op('virtual_vpu_reduce', [minrow], [BF16], [BF16], 'kind = "col_min"')
        ds = b.save(ds, scratch, mintile, BF16)
        ds, vmin = b.load(ds, scratch)
        vmax, vmin = b.binary('and', vmax, c(0x7fff)), b.binary('and', vmin, c(0x7fff))
        absmax = b.select(vmin, vmax, b.compare('ult', vmax, vmin))
        exceptional = b.cmp('uge', absmax, c(0x7f80))
        b.emit(f'cf.cond_br {exceptional}, ^exception({ds} : {STATE}), ^fast({ds} : {STATE})')
        b.emit(f'^exception(%es: {STATE}):')
        es = b.save('%es', scratch, tile, BF16)
        b.emit(f'cf.br ^lane({es}, {c(0)}, %am : {STATE}, i32, i32)')
        b.emit(f'^lane(%ls: {STATE}, %li: i32, %lm: i32):')
        more = b.cmp('ult', '%li', c(lanes))
        b.emit(f'cf.cond_br {more}, ^lane_body(%ls : {STATE}), ^join(%ls, %lm : {STATE}, i32)')
        b.emit(f'^lane_body(%lbs: {STATE}):')
        ls, val = b.load('%lbs', scratch, '%li')
        val = b.binary('and', val, c(0x7fff))
        update = b.binary('and', b.compare('ult', '%lm', val), b.compare('ult', val, c(0x7f80)))
        corrected = b.select(val, '%lm', update)
        ni = add('%li', c(1))
        b.emit(f'cf.br ^lane({ls}, {ni}, {corrected} : {STATE}, i32, i32)')
        b.emit(f'^fast(%fs: {STATE}):')
        fast = b.select(absmax, '%am', b.compare('ult', '%am', absmax))
        b.emit(f'cf.br ^join(%fs, {fast} : {STATE}, i32)')
        b.emit(f'^join(%js: {STATE}, %jm: i32):')
        ni, np = add('%ai', c(1)), add('%aptr', c(lanes * 2))
        b.emit(f'cf.br ^amax(%js, {ni}, %jm, {np} : {STATE}, i32, i32, i32)')
        b.emit(f'^scale(%ss: {STATE}, %sm: i32):')
        exp, clamped = exponent(b, '%sm', p.attrs['rule'])
        state = '%ss'
    else:
        exp, clamped, state = c(p.attrs['static_exp']), c(max(-127, min(127, p.attrs['static_exp']))), '%gbs'
    scale = add(clamped, c(127))
    exp_count = 1 if p.attrs['mode'] == 'static' or p.attrs['reduce'] in ('tensor', 'head') else p.group_elements // p.row_elements
    if groups * exp_count > 256:
        raise ValueError('ordinary virtual quantizer exponent body exceeds one slot')
    b.emit(f'cf.br ^exponent({state}, {c(0)} : {STATE}, i32)')
    b.emit(f'^exponent(%xs: {STATE}, %xi: i32):')
    more = b.cmp('ult', '%xi', c(exp_count))
    b.emit(f'cf.cond_br {more}, ^exponent_body(%xs : {STATE}), ^encode_start(%xs : {STATE})')
    b.emit(f'^exponent_body(%xbs: {STATE}):')
    absolute = c(0)
    for bit in range(exp_count.bit_length()):
        if exp_count & (1 << bit):
            term = b.binary('shl', '%gi', c(bit)) if bit else '%gi'
            absolute = add(absolute, term)
    absolute = add(absolute, '%xi')
    index = b.binary('and', absolute, c(255))
    state = b.op('virtual_slot_store', ['%xbs', exponent_slot, exp, index], [STATE, SLOT, 'i32', 'i32'], [STATE],
                 'kind = "sw", byte_offset = 0 : i32')
    ni = add('%xi', c(1))
    b.emit(f'cf.br ^exponent({state}, {ni} : {STATE}, i32)')
    b.emit(f'^encode_start(%start_encode: {STATE}):')
    b.emit(f'cf.br ^encode(%start_encode, {c(0)}, %gip, %gop : {STATE}, i32, i32, i32)')
    b.emit(f'^encode(%cs: {STATE}, %ci: i32, %cip: i32, %cop: i32):')
    more = b.cmp('ult', '%ci', c(p.group_elements // lanes))
    b.emit(f'cf.cond_br {more}, ^encode_body(%cs : {STATE}), ^group_end(%cs : {STATE})')
    b.emit(f'^encode_body(%cbs: {STATE}):')
    state, tile = b.op('virtual_dma_load_bf16', ['%cbs', '%cip', c(lanes * 2)], [STATE, 'i32', 'i32'], [STATE, BF16])
    codes = b.op('virtual_raw_pack_fp8', [tile, scale], [BF16, 'i32'], [FP8])
    state = b.op('virtual_dma_store_fp8', [state, codes, '%cop', c(lanes)], [STATE, FP8, 'i32', 'i32'], [STATE])
    ni, np, no = add('%ci', c(1)), add('%cip', c(lanes * 2)), add('%cop', c(lanes))
    b.emit(f'cf.br ^encode({state}, {ni}, {np}, {no} : {STATE}, i32, i32, i32)')
    b.emit(f'^group_end(%ges: {STATE}):')
    gi, ip, op = add('%gi', c(1)), add('%gip', c(p.group_elements * 2)), add('%gop', c(p.group_elements))
    b.emit(f'cf.br ^group(%ges, {gi}, {ip}, {op} : {STATE}, i32, i32, i32)')
    b.emit(f'^finish(%final: {STATE}):')
    state, exptile = b.op('virtual_restore', ['%final', exponent_slot], [STATE, SLOT], [STATE, FP8])
    state = b.op('virtual_dma_store_fp8', [state, exptile, c(p.exponent_address), c(p.exponent_bytes)],
                 [STATE, FP8, 'i32', 'i32'], [STATE])
    b.emit(f'return {state} : {STATE}')
    attrs = (f'atlas.input_dram_base = {p.io_memory_base} : i64, atlas.output_dram_base = {p.code_address} : i64, '
             f'atlas.output_dram_bytes = {p.workspace_bytes - p.input_bytes} : i64')
    return 'module {\n  func.func @activation_quantizer() -> ' + STATE + ' attributes {' + attrs + '} {\n' + '\n'.join(b.lines) + '\n  }\n}\n'
