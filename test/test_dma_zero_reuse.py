"""Partial DMA padding reuses only surviving zero tensors and staging tails."""
import os

import pytest

from atlas_mlir_native.binding import DRAM_BASE, NATIVE_CHECKS
from test_exponent_restoration import compile_fixture
from test_virtual_lowering import run

S, SLOT = '!atlas.virtual_state', '!atlas.virtual_spill'


def source(fmt, boundary):
    tensor = '!atlas.virtual_' + fmt
    size = 1024 if fmt == 'fp8' else 2048
    body = [f'%start = "atlas.virtual_start"() : () -> {S}']
    if boundary == 'copy':
        body.extend(f'%slot{i} = "atlas.virtual_slot"() {{kind = "{fmt}", words = {size//4} : i32, '
                    f'slot_word = -1 : i32, purpose = "scratch"}} : () -> {SLOT}' for i in range(2))
    state = 'start'
    for index, prefix in enumerate((32,96,128)):
        if index == 1 and boundary == 'branch':
            body.extend([f'cf.br ^next(%{state} : {S})',f'^next(%next_state: {S}):'])
            state = 'next_state'
        body.extend([
            f'%input{index} = arith.constant {DRAM_BASE+index*2048-(1<<32)} : i32',
            f'%output{index} = arith.constant {DRAM_BASE+8192+index*size-(1<<32)} : i32',
            f'%prefix{index} = arith.constant {prefix} : i32',
            f'%full{index} = arith.constant {size} : i32',
            f'%loaded{index}, %tile{index} = "atlas.virtual_dma_load_{fmt}"(%{state}, %input{index}, %prefix{index}) '
            f': ({S}, i32, i32) -> ({S}, {tensor})',
        ])
        state, value = f'loaded{index}',f'tile{index}'
        if index == 0 and boundary == 'copy':
            body.extend([
                f'%saved = "atlas.virtual_save"(%{state}, %slot0, %{value}) : ({S}, {SLOT}, {tensor}) -> {S}',
                f'%copied = "atlas.virtual_slot_copy"(%saved, %slot0, %slot1) : ({S}, {SLOT}, {SLOT}) -> {S}',
                f'%restored, %copy = "atlas.virtual_restore"(%copied, %slot1) : ({S}, {SLOT}) -> ({S}, {tensor})',
            ])
            state,value = 'restored','copy'
        elif index == 0 and boundary == 'pack':
            body.append(f'%roundtrip = "atlas.virtual_pack_roundtrip_bf16"(%{value}) '
                        f'{{scale_code = 127 : i32}} : ({tensor}) -> {tensor}')
            value = 'roundtrip'
        body.append(f'%stored{index} = "atlas.virtual_dma_store_{fmt}"(%{state}, %{value}, %output{index}, %full{index}) '
                    f': ({S}, {tensor}, i32, i32) -> {S}')
        state = f'stored{index}'
    body.append(f'return %{state} : {S}')
    return (f'module {{ func.func @prefix() -> {S} attributes {{atlas.input_dram_base = {DRAM_BASE} : i64, '
            f'atlas.output_dram_base = {DRAM_BASE+8192} : i64, atlas.output_dram_bytes = {3*size} : i64}} {{\n'
            + '\n'.join(body) + '\n}}\n')


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline',['mandatory','optimized'])
@pytest.mark.parametrize('fmt,boundary',[(fmt,boundary) for fmt in ('fp8','bf16')
                                       for boundary in ('none','copy','branch')] + [('bf16','pack')])
def test_zero_reuse_and_clobbers_keep_every_output_padding_byte(tmp_path,pipeline,fmt,boundary):
    from rust_model import execute

    machine,program = compile_fixture(source(fmt,boundary),pipeline=pipeline)
    zeros = [row for row in program['instructions'] if row['operation'] == 'atlas.vli'
             and row['fields']['immediate'] == 0 and row['fields']['mode'] == 'all']
    if boundary == 'none':
        assert len(zeros) == 1
    elif pipeline == 'mandatory':
        assert len(zeros) == 2
    else:
        # Optimization can forward the slot copy or schedule the prefix loads
        # before the clobber; numerical padding checks below still apply.
        assert 1 <= len(zeros) <= 2
    size = 1024 if fmt == 'fp8' else 2048
    memory = bytearray([0xa5])*(8192+3*size+64)
    expected = bytearray(memory)
    for index,prefix in enumerate((32,96,128)):
        payload = bytes([0x38])*prefix if fmt == 'fp8' else b'\x80\x3f'*(prefix//2)
        memory[index*2048:index*2048+prefix] = payload
        expected[index*2048:index*2048+prefix] = payload
        output = 8192+index*size
        expected[output:output+size] = payload+bytes(size-prefix)
    actual,_ = execute(program,memory,DRAM_BASE,tmp_path/'run')
    assert actual == expected
    # Reusing an unchecked constant is not sufficient: final buffer verification
    # must still prove the exact zero consumed by every prefill VSTORE.
    poisoned = machine.replace('immediate = 0 : i32, mode = "all"',
                               'immediate = 1 : i32, mode = "all"',1)
    assert poisoned != machine
    checked = run('atlas-opt',poisoned,*NATIVE_CHECKS)
    assert checked.returncode != 0 and 'zero tensor' in checked.stderr


def staging_source(formats,prefixes,barrier):
    body = [f'%start = "atlas.virtual_start"() : () -> {S}']
    for i,fmt in enumerate(formats):
        words = 256 if fmt == 'fp8' else 512
        body.append(f'%slot{i} = "atlas.virtual_slot"() {{kind = "{fmt}", words = {words} : i32, '
                    f'slot_word = -1 : i32, purpose = "scratch"}} : () -> {SLOT}')
    state = 'start'
    for i,(fmt,prefix) in enumerate(zip(formats,prefixes,strict=True)):
        if i == 1 and barrier == 'branch':
            body.extend([f'cf.br ^next(%{state} : {S})',f'^next(%next_state: {S}):'])
            state = 'next_state'
        t = '!atlas.virtual_' + fmt
        body.extend([
            f'%input{i} = arith.constant {DRAM_BASE+i*2048-(1<<32)} : i32',
            f'%prefix{i} = arith.constant {prefix} : i32',
            f'%loaded{i}, %tile{i} = "atlas.virtual_dma_load_{fmt}"(%{state}, %input{i}, %prefix{i}) '
            f': ({S}, i32, i32) -> ({S}, {t})',
            f'%saved{i} = "atlas.virtual_save"(%loaded{i}, %slot{i}, %tile{i}) : ({S}, {SLOT}, {t}) -> {S}',
        ])
        state = f'saved{i}'
        if i == 0 and barrier == 'store':
            size = 1024 if fmt == 'fp8' else 2048
            body.extend([
                f'%extra = arith.constant {DRAM_BASE+8192+3*2048-(1<<32)} : i32',
                f'%extra_size = arith.constant {size} : i32',
                f'%extra_stored = "atlas.virtual_dma_store_{fmt}"(%{state}, %tile{i}, %extra, %extra_size) '
                f': ({S}, {t}, i32, i32) -> {S}',
            ])
            state = 'extra_stored'
    for i,fmt in enumerate(formats):
        t = '!atlas.virtual_' + fmt
        size = 1024 if fmt == 'fp8' else 2048
        body.extend([
            f'%restored{i}, %value{i} = "atlas.virtual_restore"(%{state}, %slot{i}) : ({S}, {SLOT}) -> ({S}, {t})',
            f'%output{i} = arith.constant {DRAM_BASE+8192+i*2048-(1<<32)} : i32',
            f'%full{i} = arith.constant {size} : i32',
            f'%stored{i} = "atlas.virtual_dma_store_{fmt}"(%restored{i}, %value{i}, %output{i}, %full{i}) '
            f': ({S}, {t}, i32, i32) -> {S}',
        ])
        state = f'stored{i}'
    body.append(f'return %{state} : {S}')
    return (f'module {{ func.func @prefix() -> {S} attributes {{atlas.input_dram_base = {DRAM_BASE} : i64, '
            f'atlas.output_dram_base = {DRAM_BASE+8192} : i64, atlas.output_dram_bytes = 8192 : i64}} {{\n'
            + '\n'.join(body) + '\n}}\n')


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline',['mandatory','optimized'])
@pytest.mark.parametrize('formats,prefixes,barrier',[
    ((fmt,)*3,prefixes,barrier) for fmt in ('fp8','bf16') for prefixes,barrier in (
        ((32,32,32),'none'),((32,96,128),'none'),((128,32,96),'none'),
        ((32,1024,32),'none'),((32,32,32),'branch'),((32,32,32),'store'))
] + [(('fp8','bf16','fp8'),(32,96,128),'none'),
     (('bf16',)*3,(32,2048,32),'none')])
def test_reused_staging_tail_reinitializes_after_shorter_writes_or_barriers(tmp_path,pipeline,formats,prefixes,barrier):
    from rust_model import execute

    _,program = compile_fixture(staging_source(formats,prefixes,barrier),pipeline=pipeline)
    if pipeline == 'mandatory' and prefixes == (32,32,32) and barrier == 'none':
        # Three slot saves, three output stores and one shared staging prefill.
        halves = 1 if formats[0] == 'fp8' else 2
        assert sum(row['operation'] == 'atlas.vstore' for row in program['instructions']) == 7*halves
    memory = bytearray([0xa5])*(16384+64)
    expected = bytearray(memory)
    for i,(fmt,prefix) in enumerate(zip(formats,prefixes,strict=True)):
        size = 1024 if fmt == 'fp8' else 2048
        payload = bytes([0x31+i])*prefix
        memory[i*2048:i*2048+prefix] = payload
        expected[i*2048:i*2048+prefix] = payload
        output = 8192+i*2048
        expected[output:output+size] = payload+bytes(size-prefix)
        if i == 0 and barrier == 'store':
            expected[14336:14336+size] = payload+bytes(size-prefix)
    actual,_ = execute(program,memory,DRAM_BASE,tmp_path/'run')
    assert actual == expected


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'),reason='select Atlas tools')
@pytest.mark.parametrize('fmt',['fp8','bf16'])
def test_reused_tail_proof_rejects_intervening_write_and_nonzero_prefill(fmt):
    from verification_support import constant, insert_after, line_index, records

    machine,_ = compile_fixture(staging_source((fmt,)*3,(32,32,32),'none'))
    facts = records(machine,'tile')
    wait = next(row for row in facts if row['kind'] == 'dma_wait' and row['transfer'] == 1)
    load = next(row for row in facts if row['kind'] == 'dma_load' and row['transfer'] == 1)
    anchor = line_index(machine,f'atlas.virtual_tile_command = {wait["id"]} : i32')
    # The second prefix retains the first prefill's tail. A later scalar store
    # in that tail must invalidate its provenance, even with valid DMA tags.
    overwritten = insert_after(machine,anchor,constant(27,load['vmem_byte']) + [('scalar_store',
        'kind = "sw", src = 7 : i32, base = 27 : i32, offset = 64 : i32')])
    nonzero = machine.replace('immediate = 0 : i32, mode = "all"',
                              'immediate = 1 : i32, mode = "all"',1)
    for changed,diagnostic in ((overwritten,'buffer contract'),(nonzero,'zero tensor')):
        assert changed != machine
        checked = run('atlas-opt',changed,*NATIVE_CHECKS)
        assert checked.returncode != 0 and diagnostic in checked.stderr, checked.stderr
