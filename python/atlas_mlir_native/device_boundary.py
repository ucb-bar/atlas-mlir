"""Typed packed-BF16 transitions between admitted device regions.

These transitions reproduce an internal logical boundary: admission follows
optional negative-zero canonicalization, rejected logical words become the
consumer's padding code, and nonlogical lanes always become padding. A shared
invocation-local status slot records rejection without data-dependent control.
"""
from __future__ import annotations

from .exponent_restoration import BF16, SLOT, STATE, Emitter
from .region_composition import rematerialize_scalar_constants


KINDS = ('identity', 'column_to_row', 'broadcast_column')


def page_count(rows, columns, kind='identity'):
    if type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0:
        raise ValueError('device boundary dimensions must be positive integers')
    if kind not in KINDS or (kind != 'identity' and columns != 1):
        raise ValueError('device boundary layout requires a supported kind and column input')
    return ((rows+31)//32) * (((columns+31)//32) if kind == 'identity' else 1)


def _configuration(source, destination, status, rows, columns, kind, ranges, pad,
                   canonical, output_base, output_bytes):
    pages = page_count(rows, columns, kind)
    if type(canonical) is not bool or type(pad) is not int or not 0 <= pad <= 65535:
        raise ValueError('device boundary padding or zero policy differs')
    ranges = tuple(tuple(pair) for pair in ranges)
    if any(len(pair) != 2 or any(type(v) is not int for v in pair) or not 0 <= pair[0] <= pair[1] <= 65535
           for pair in ranges):
        raise ValueError('device boundary admission requires inclusive BF16 code ranges')
    if canonical and pad == 0x8000:
        pad = 0
    if ranges and not any(low <= pad <= high for low, high in ranges):
        raise ValueError('device boundary pad must be admitted by its consumer')
    spans = ((source,pages*2048),(destination,pages*2048),(status,2048))
    if any(type(address) is not int or not 0x80000000 <= address <= 0xffffffff or address % 2048
           or address+size > 1 << 32 for address,size in spans):
        raise ValueError('device boundary pages must be aligned fixed DRAM ranges')
    if any(a < b+bn and b < a+an for i,(a,an) in enumerate(spans) for b,bn in spans[i+1:]):
        raise ValueError('device boundary source, destination, and status must be disjoint')
    if output_base is None and output_bytes is None:
        output_base = min(destination,status)
        output_bytes = max(destination+pages*2048,status+2048)-output_base
        if source < output_base+output_bytes and output_base < source+pages*2048:
            raise ValueError('implicit boundary output window overlaps its source')
    if (type(output_base) is not int or type(output_bytes) is not int or output_base % 32 or output_bytes % 32
            or output_base < 0x80000000 or output_bytes <= 0 or output_base+output_bytes > 1 << 32
            or any(address < output_base or address+size > output_base+output_bytes for address,size in spans[1:])):
        raise ValueError('device boundary output window does not cover its publications')
    return pages,ranges,pad,output_base,output_bytes


def _boolean(e, value):
    name = e.name('bool')
    e.emit(f'%{name} = arith.constant {int(value)} : i1')
    return name


def _admitted(e, code, ranges):
    if not ranges:
        return _boolean(e, True)
    admitted = None
    for low, high in ranges:
        if low == high:
            within = e.cmp('eq', code, e.const(low))
        elif low == 0:
            within = e.cmp('ule', code, e.const(high))
        elif high == 65535:
            within = e.cmp('uge', code, e.const(low))
        else:
            within = e.combine('andi', e.cmp('uge', code, e.const(low)), e.cmp('ule', code, e.const(high)))
        admitted = within if admitted is None else e.combine('ori', admitted, within)
    return admitted


def virtual_transition(*, source_address, destination_address, status_address,
                       rows, columns, kind='identity', admitted_code_ranges=(), pad=0,
                       canonicalize_negative_zero=False, output_base=None, output_bytes=None,
                       initialize_status=True):
    """Return one composable packed-tensor boundary program.

    ``rows`` and ``columns`` describe the producer's logical matrix. The two
    column transforms require ``columns=1``. Empty admission ranges admit every
    raw BF16 code. Explicit output bounds allow a caller-owned mutable frame;
    otherwise the destination/status hull must exclude the source pages.
    A later boundary may explicitly reload the preceding boundary's DRAM status
    to preserve rejection while keeping independent member-local scratch slots.
    """
    pages,ranges,pad,output_base,output_bytes = _configuration(
        source_address,destination_address,status_address,rows,columns,kind,
        admitted_code_ranges,pad,canonicalize_negative_zero,output_base,output_bytes)
    mc = (rows+31)//32
    nc = (columns+31)//32 if kind == 'identity' else 1
    e = Emitter('boundary')
    e.lines = ['module {', f'  func.func @device_boundary() -> {STATE} attributes {{atlas.input_dram_base = {source_address} : i64, atlas.output_dram_base = {output_base} : i64, atlas.output_dram_bytes = {output_bytes} : i64}} {{',
               f'    %start = "atlas.virtual_start"() : () -> {STATE}']
    for name,role in (('boundary_source','boundary_source'),('boundary_destination','boundary_destination'),('boundary_status','status')):
        e.emit(f'%{name} = "atlas.virtual_slot"() {{kind = "bf16", words = 512 : i32, slot_word = -1 : i32, purpose = "scratch", atlas.exponent_scratch = "{role}"}} : () -> {SLOT}')
    e.emit(f'%boundary_zero_tile = "atlas.virtual_vpu_splat"() {{bits = 0 : i32}} : () -> {BF16}')
    e.emit(f'%boundary_pad_tile = "atlas.virtual_vpu_splat"() {{bits = {pad} : i32}} : () -> {BF16}')
    e.state = 'start'
    status_tile = 'boundary_zero_tile' if initialize_status else e.effect(
        'dma_load_bf16', [e.const(status_address), e.const(2048)], ['i32', 'i32'], BF16)
    e.effect('save',['boundary_status',status_tile],[SLOT,BF16])
    zero = e.const(0)
    source = e.const(source_address)
    destination = e.const(destination_address)
    e.emit(f'cf.br ^boundary_rows(%{e.state}, %{zero}, %{source}, %{destination} : {STATE}, i32, i32, i32)')
    e.lines.append(f'  ^boundary_rows(%boundary_rs: {STATE}, %boundary_rt: i32, %boundary_row_source: i32, %boundary_row_destination: i32):')
    more = e.cmp('ult','boundary_rt',e.const(mc))
    e.emit(f'cf.cond_br %{more}, ^boundary_row_body(%boundary_rs : {STATE}), ^boundary_finished(%boundary_rs : {STATE})')
    e.lines.append(f'  ^boundary_row_body(%boundary_rb: {STATE}):')
    e.emit(f'cf.br ^boundary_columns(%boundary_rb, %{zero}, %boundary_row_source, %boundary_row_destination : {STATE}, i32, i32, i32)')
    e.lines.append(f'  ^boundary_columns(%boundary_cs: {STATE}, %boundary_ct: i32, %boundary_source_page: i32, %boundary_destination_page: i32):')
    more = e.cmp('ult','boundary_ct',e.const(nc))
    e.emit(f'cf.cond_br %{more}, ^boundary_page(%boundary_cs : {STATE}), ^boundary_next_row(%boundary_cs : {STATE})')
    e.lines.append(f'  ^boundary_page(%boundary_ps: {STATE}):')
    e.state = 'boundary_ps'
    tile = e.effect('dma_load_bf16',['boundary_source_page',e.const(2048)],['i32','i32'],BF16)
    e.effect('save',['boundary_source',tile],[SLOT,BF16])
    e.effect('save',['boundary_destination','boundary_pad_tile'],[SLOT,BF16])
    invalid = e.load('boundary_status',zero)
    e.emit(f'cf.br ^boundary_words(%{e.state}, %{zero}, %{invalid} : {STATE}, i32, i32)')
    e.lines.append(f'  ^boundary_words(%boundary_ws: {STATE}, %boundary_i: i32, %boundary_invalid: i32):')
    more = e.cmp('ult','boundary_i',e.const(512))
    e.emit(f'cf.cond_br %{more}, ^boundary_word_body(%boundary_ws : {STATE}), ^boundary_word_done(%boundary_ws, %boundary_invalid : {STATE}, i32)')
    e.lines.append(f'  ^boundary_word_body(%boundary_wb: {STATE}):')
    e.state = 'boundary_wb'
    invalid = 'boundary_invalid'
    result = []
    for lane in range(2):
        row = e.binary('shrui',e.binary('andi','boundary_i',e.const(255)),e.const(3))
        column = e.binary('addi',e.binary('shli',e.binary('andi','boundary_i',e.const(7)),e.const(1)),
                          e.binary('shli',e.binary('shrui','boundary_i',e.const(8)),e.const(4)))
        if lane:
            column = e.binary('addi',column,e.const(1))
        if kind == 'identity':
            word = e.load('boundary_source','boundary_i')
            if lane:
                word = e.binary('shrui',word,e.const(16))
            global_row = e.binary('addi',e.binary('shli','boundary_rt',e.const(5)),row)
            global_column = e.binary('addi',e.binary('shli','boundary_ct',e.const(5)),column)
            logical = e.combine('andi',e.cmp('ult',global_row,e.const(rows)),e.cmp('ult',global_column,e.const(columns)))
        elif kind == 'column_to_row':
            word = e.load('boundary_source',e.binary('shli',column,e.const(3)))
            global_column = e.binary('addi',e.binary('shli','boundary_rt',e.const(5)),column)
            logical = e.combine('andi',e.cmp('eq',row,zero),e.cmp('ult',global_column,e.const(rows)))
        else:
            word = e.load('boundary_source',e.binary('shli',row,e.const(3)))
            global_row = e.binary('addi',e.binary('shli','boundary_rt',e.const(5)),row)
            logical = e.cmp('ult',global_row,e.const(rows))
        code = e.binary('andi',word,e.const(65535))
        if canonicalize_negative_zero:
            keep = e.binary('subi',zero,e.flag(e.cmp('ne',code,e.const(0x8000))))
            code = e.binary('andi',code,keep)
        admitted = _admitted(e,code,ranges)
        rejected = e.combine('andi',logical,e.binary('xori',admitted,_boolean(e,True),'i1'))
        invalid = e.binary('ori',invalid,e.flag(rejected))
        take = e.binary('subi',zero,e.flag(e.combine('andi',logical,admitted)))
        difference = e.binary('xori',code,e.const(pad))
        result.append(e.binary('xori',e.const(pad),e.binary('andi',difference,take)))
    packed = e.binary('ori',result[0],e.binary('shli',result[1],e.const(16)))
    e.store('boundary_destination','boundary_i',packed)
    index = e.binary('addi','boundary_i',e.const(1))
    e.emit(f'cf.br ^boundary_words(%{e.state}, %{index}, %{invalid} : {STATE}, i32, i32)')
    e.lines.append(f'  ^boundary_word_done(%boundary_wd: {STATE}, %boundary_bad: i32):')
    e.state = 'boundary_wd'
    e.store('boundary_status',zero,'boundary_bad')
    tile = e.effect('restore',['boundary_destination'],[SLOT],BF16)
    e.effect('dma_store_bf16',[tile,'boundary_destination_page',e.const(2048)],[BF16,'i32','i32'])
    column = e.binary('addi','boundary_ct',e.const(1))
    source = e.binary('addi','boundary_source_page',e.const(2048))
    destination = e.binary('addi','boundary_destination_page',e.const(2048))
    e.emit(f'cf.br ^boundary_columns(%{e.state}, %{column}, %{source}, %{destination} : {STATE}, i32, i32, i32)')
    e.lines.append(f'  ^boundary_next_row(%boundary_nr: {STATE}):')
    row = e.binary('addi','boundary_rt',e.const(1))
    source = e.binary('addi','boundary_row_source',e.const(nc*2048))
    destination = e.binary('addi','boundary_row_destination',e.const(nc*2048))
    e.emit(f'cf.br ^boundary_rows(%boundary_nr, %{row}, %{source}, %{destination} : {STATE}, i32, i32, i32)')
    e.lines.append(f'  ^boundary_finished(%boundary_fs: {STATE}):')
    e.state = 'boundary_fs'
    tile = e.effect('restore',['boundary_status'],[SLOT],BF16)
    e.effect('dma_store_bf16',[tile,e.const(status_address),e.const(2048)],[BF16,'i32','i32'])
    e.emit(f'return %{e.state} : {STATE}')
    e.lines += ['  }','}']
    return rematerialize_scalar_constants('\n'.join(e.lines)+'\n')
