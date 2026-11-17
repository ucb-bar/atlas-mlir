"""Shared stage code preserves explicit addresses, order and one completion."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from test_program_composition import composed_source

TOOLS = Path(os.environ.get('ATLAS_OOT_BIN_DIR', 'build/bin'))
NATIVE = bool(os.environ.get('ATLAS_OOT_BIN_DIR'))


def tool(name, source, *options):
    return subprocess.run([str(TOOLS/name), *options, '-'], input=source,
                          capture_output=True, text=True)


def copy_stages(count=3):
    functions = []
    for index in range(count):
        functions.append(f'''
  func.func @stage{index}() -> !atlas.virtual_state attributes {{
    atlas.input_dram_base = 2415919104 : i64,
    atlas.output_dram_base = {0x90004000+index*2048} : i64,
    atlas.output_dram_bytes = 2048 : i64}} {{
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %a = arith.constant {0x90000000+index*2048-(1<<32)} : i32
    %b = arith.constant {0x90004000+index*2048-(1<<32)} : i32
    %n = arith.constant 2048 : i32
    %t, %v = "atlas.virtual_dma_load_bf16"(%s, %a, %n)
      : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %end = "atlas.virtual_dma_store_bf16"(%t, %v, %b, %n)
      : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    return %end : !atlas.virtual_state
  }}''')
    names = ', '.join(f'"stage{i}"' for i in range(count))
    return f'module attributes {{atlas.program_entries = [{names}]}} {{'+''.join(functions)+'}'


def compile_program(source, shared, pipeline='--atlas-compile-unscheduled'):
    composed = tool('atlas-opt', source, f'--compose-atlas-program=share-bodies={str(shared).lower()}')
    assert composed.returncode == 0, composed.stderr
    native = tool('atlas-opt', composed.stdout, pipeline)
    assert native.returncode == 0, native.stderr
    checked = tool('atlas-opt', native.stdout, '--verify-atlas-generated-schedule',
                   '--verify-atlas-machine-stream', '--verify-atlas-timing')
    assert checked.returncode == 0, checked.stderr
    emitted = tool('atlas-emit', native.stdout, '--program-json')
    assert emitted.returncode == 0, emitted.stderr
    program = json.loads(emitted.stdout)
    traps = [r for r in program['instructions'] if r['operation'] == 'atlas.trap']
    assert len(traps) == 1 and traps[0]['fields']['kind'] == 'ecall'
    return composed.stdout, native.stdout, program


@pytest.mark.skipif(not NATIVE, reason='select Atlas tools')
@pytest.mark.parametrize('count', [1, 2, 3, 5])
def test_address_parameters_share_one_body_and_keep_a_finite_bound(count):
    from atlas_mlir_native.restoration_budget import instruction_bound

    source, machine, program = compile_program(copy_stages(count), True)
    assert source.count('"atlas.virtual_dma_store_bf16"') == 1
    assert source.count('"atlas.virtual_start"') == 1
    bound = instruction_bound(source, machine, program, max_instructions=1000000)
    assert bound['admitted']


@pytest.mark.skipif(not NATIVE, reason='select Atlas tools')
def test_shared_nested_k_loops_reduce_encoded_image_size():
    source = composed_source(8, 257)
    plain, _, before = compile_program(source, False)
    shared, machine, after = compile_program(source, True)
    assert shared.count('"atlas.virtual_mxu_reset"')*2 == plain.count('"atlas.virtual_mxu_reset"')
    assert len(after['instructions']) < len(before['instructions'])
    from atlas_mlir_native.restoration_budget import instruction_bound
    assert instruction_bound(shared, machine, after, max_instructions=1000000)['admitted']


@pytest.mark.skipif(not NATIVE, reason='select Atlas tools')
def test_loop_bounds_stay_literal_and_shared_image_obeys_the_code_budget():
    changed = composed_source(1,65).replace('%k_count = arith.constant 2',
                                            '%k_count = arith.constant 3',1)
    shared = tool('atlas-opt',changed,'--compose-atlas-program=share-bodies=true')
    plain = tool('atlas-opt',changed,'--compose-atlas-program')
    assert shared.returncode == plain.returncode == 0
    assert shared.stdout == plain.stdout
    source = composed_source(8,257).replace('atlas.input_dram_base',
        'atlas.program_size_limit_bytes = 1600 : i64, atlas.input_dram_base')
    assert source.count('atlas.program_size_limit_bytes = 1600 : i64') == 2
    _, _, program = compile_program(source,True)
    assert len(program['instructions'])*4 <= 1600
    refused = tool('atlas-opt',source,'--compose-atlas-program','--atlas-compile-unscheduled')
    assert refused.returncode != 0
    refused = tool('atlas-opt',source.replace('1600 : i64','1024 : i64'),
                   '--compose-atlas-program=share-bodies=true','--atlas-compile-unscheduled')
    assert refused.returncode != 0


@pytest.mark.skipif(not NATIVE, reason='select Atlas tools')
def test_different_operations_attributes_and_value_uses_do_not_share():
    source = copy_stages(2)
    insertion = '''
    %u = "atlas.virtual_vpu_binary"(%v, %v) {kind = "add"}
      : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %end = "atlas.virtual_dma_store_bf16"(%t, %u, %b, %n)'''
    source = source.replace('%end = "atlas.virtual_dma_store_bf16"(%t, %v, %b, %n)', insertion)
    for changed in (
        source.replace('kind = "add"', 'kind = "sub"', 1),
        source.replace('(%t, %u, %b, %n)', '(%t, %v, %b, %n)', 1),
        source.replace('arith.constant 2048 : i32',
                       '"arith.constant"() <{value = 2048 : i32}> {test.tag = 1 : i32} : () -> i32', 1),
    ):
        result = tool('atlas-opt', changed, '--compose-atlas-program=share-bodies=true')
        assert result.returncode == 0, result.stderr
        assert result.stdout.count('"atlas.virtual_dma_store_bf16"') == 2
    # Function-scoped policies must still agree before body sharing is considered.
    changed = source.replace('atlas.input_dram_base', 'atlas.test_policy = 1 : i64, atlas.input_dram_base', 1)
    refused = tool('atlas-opt', changed, '--compose-atlas-program=share-bodies=true')
    assert refused.returncode != 0 and 'matching Atlas function attributes' in refused.stderr


def merged_addresses(middle):
    return f'''module {{
  func.func @joined_addresses() -> !atlas.virtual_state attributes {{
    atlas.input_dram_base = 2415919104 : i64,
    atlas.output_dram_base = 2415984640 : i64,
    atlas.output_dram_bytes = 2048 : i64}} {{
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %cond = arith.constant true
    cf.cond_br %cond, ^first(%s : !atlas.virtual_state), ^choose(%s : !atlas.virtual_state)
  ^first(%a: !atlas.virtual_state):
    %p0 = arith.constant -1879048192 : i32
    cf.br ^read(%a, %p0 : !atlas.virtual_state, i32)
  ^choose(%b: !atlas.virtual_state):
    cf.cond_br %cond, ^second(%b : !atlas.virtual_state), ^third(%b : !atlas.virtual_state)
  ^second(%c: !atlas.virtual_state):
    %p1 = arith.constant {middle} : i32
    cf.br ^read(%c, %p1 : !atlas.virtual_state, i32)
  ^third(%d: !atlas.virtual_state):
    %p2 = arith.constant -1879044096 : i32
    cf.br ^read(%d, %p2 : !atlas.virtual_state, i32)
  ^read(%e: !atlas.virtual_state, %p: i32):
    %n = arith.constant 2048 : i32
    %o = arith.constant -1878982656 : i32
    %t, %v = "atlas.virtual_dma_load_bf16"(%e, %p, %n)
      : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %end = "atlas.virtual_dma_store_bf16"(%t, %v, %o, %n)
      : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
    return %end : !atlas.virtual_state
  }}
}}'''


@pytest.mark.skipif(not NATIVE, reason='select Atlas tools')
def test_every_incoming_address_is_checked_including_unaligned_interior():
    valid = tool('atlas-opt', merged_addresses(-1879046144), '--atlas-compile-unscheduled')
    assert valid.returncode == 0, valid.stderr
    replay = tool('atlas-opt', valid.stdout, '--verify-atlas-generated-schedule', '--verify-atlas-timing')
    assert replay.returncode == 0, replay.stderr
    for middle in (-1879046143, -1, 0x40000):
        invalid = tool('atlas-opt', merged_addresses(middle), '--atlas-compile-unscheduled')
        assert invalid.returncode != 0
    # Replay independently reconstructs the retained source address choices;
    # the live analysis has already finished by the time this is changed.
    original = 'constant = -1879046144 : i32'
    assert original in valid.stdout
    for middle in (-1879046143, -1, 0x40000):
        forged = valid.stdout.replace(original, f'constant = {middle} : i32', 1)
        refused = tool('atlas-opt', forged, '--verify-atlas-generated-schedule')
        assert refused.returncode != 0
        assert 'bounded DMA address' in refused.stderr


def execute(tmp_path, program, initial, trial):
    binary = b''.join(row['word_u32'].to_bytes(4,'little') for row in program['instructions'])
    (tmp_path/'program.bin').write_bytes(binary)
    (tmp_path/'config.json').write_text(json.dumps(dict(numerics='rtl',compute='cpu',
        pack_layout='register_rows',dram_base=0x90000000,dram_size=len(initial))))
    (tmp_path/'initial.bin').write_bytes(initial)
    out = tmp_path/f'state{trial}'
    result = subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'],str(tmp_path/'program.bin'),
        '--config',str(tmp_path/'config.json'),'--load',f'{0x90000000}:{tmp_path/"initial.bin"}',
        '--dump-state',str(out)],capture_output=True,check=True)
    report = json.loads(result.stdout)
    assert report['completed'] and report['halt_reason'] == 'ecall'
    return b''.join((out/f'dram_{address:016x}.bin').read_bytes()
                   for address in range(0x90000000,0x90000000+len(initial),4096))


@pytest.mark.skipif(not NATIVE or not os.environ.get('ATLAS_RUST_NPU_MODEL'),
                    reason='select Atlas tools and Rust')
@pytest.mark.parametrize('pipeline', ['--atlas-compile-unscheduled',
                                     '--atlas-compile=enhanced-virtual-scheduling=true'])
def test_shared_copies_run_once_with_distinct_inputs_and_guarded_outputs(tmp_path, pipeline):
    _, _, program = compile_program(copy_stages(), True, pipeline)
    for trial in range(3):
        initial = bytes((i*17+trial*23+i//256)%256 for i in range(32768))
        expected = bytearray(initial)
        expected[16384:16384+6144] = initial[:6144]
        actual = execute(tmp_path, program, initial, trial)
        assert actual == expected


@pytest.mark.skipif(not NATIVE or not os.environ.get('ATLAS_RUST_NPU_MODEL'),
                    reason='select Atlas tools and Rust')
def test_shared_matrix_body_preserves_ordered_k_and_resets_each_invocation(tmp_path):
    from atlas_mlir_native.looped_tiling import prepare_plan
    from merlin.semantic_compiler.ordered_mnk_tiling import materialize_looped_mnk_inputs
    from test_looped_mnk import source_request

    plan = prepare_plan(source_request(1, 1, 257))
    _, _, program = compile_program(composed_source(8,257), True)
    for trial in range(3):
        a = bytes(0x38 if i in (0,256) else 0 for i in range(257))
        w = bytes((0x38 if trial != 1 else 0xb8) if i in (0,256) else 0 for i in range(257))
        operands = materialize_looped_mnk_inputs(plan, {'a':a,'w':w})
        initial = bytearray((i*29+trial*47)%256 for i in range(81920))
        for offset, payload in zip((0,16384),operands,strict=True):
            initial[offset:offset+len(payload)] = payload
        expected = bytearray(initial)
        tile = (0x4000 if trial != 1 else 0xc000).to_bytes(2,'little')+bytes(2046)
        expected[65536:69632] = tile*2
        assert execute(tmp_path, program, initial, trial) == expected


@pytest.mark.skipif(not NATIVE or not os.environ.get('ATLAS_RUST_NPU_MODEL'),
                    reason='select Atlas tools and Rust')
def test_composition_api_keeps_dependent_intermediates_on_device(tmp_path):
    from atlas_mlir_native.region_composition import DeviceBuffer, VirtualStage, compose_regions
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.printer import Printer
    import io

    module = parse_mlir_text(copy_stages())
    stages = []
    for index, function in enumerate(tuple(module.body.block.ops)):
        # All stages have the same shape. Only their explicit addresses differ.
        # The second and third stages consume the previous stage's result.
        if index:
            from xdsl.dialects.arith import ConstantOp
            from xdsl.dialects.builtin import IntegerAttr
            for op in function.walk():
                if isinstance(op, ConstantOp) and op.value.value.data == 0x90000000+index*2048-(1<<32):
                    op.properties['value'] = IntegerAttr(0x90004000+(index-1)*2048-(1<<32),32)
        module.body.block.detach_op(function)
        fragment = ModuleOp([function])
        stream = io.StringIO(); Printer(stream=stream).print_op(fragment)
        stages.append(VirtualStage(f'stage{index}',stream.getvalue(),
            ('input' if index==0 else f'out{index-1}',),(f'out{index}',)))
    buffers = [DeviceBuffer('input',0x90000000,2048,'bf16_bytes')]
    buffers += [DeviceBuffer(f'out{i}',0x90004000+i*2048,2048,'bf16_bytes') for i in range(3)]
    compiled = tmp_path/'compiled'
    manifest = compose_regions(stages,buffers=buffers,initialized=('input',),outputs=('out2',),
        destination=compiled,tools=TOOLS,io_memory_bytes=32768,share_bodies=True)
    assert manifest['launches_per_invocation'] == 1 and manifest['stage_count'] == 3
    record = json.loads((compiled/'execution_plan.json').read_text())
    assert record['share_bodies'] is True
    program = json.loads((compiled/'physical-program.json').read_text())
    initial = bytes((i*71+i//128)%256 for i in range(32768))
    expected = bytearray(initial)
    expected[16384:16384+6144] = initial[:2048]*3
    assert execute(tmp_path,program,initial,0) == expected
