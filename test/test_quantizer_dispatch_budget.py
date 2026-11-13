"""Selected execution budgets batch whole groups without weakening admission."""
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from atlas_mlir_native import activation_quantize_application as application
from atlas_mlir_native import vector_quantize as device
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_activation_quantize_device import dynamic_source, host_reference

ROOT = Path(__file__).resolve().parents[1]


def options(budget=128000000, memory=67108864, **kwargs):
    return codegen_options(CodegenOptions(execution_instruction_limit=budget,
                                         io_memory_bytes=memory, **kwargs))


@pytest.mark.parametrize('heads,length', [(32, 4096), (13, 1024), (10, 96), (9, 32768)])
def test_selected_budget_batches_whole_heads(heads, length):
    source = dynamic_source(shape=(heads, 1, length), reduce='head')
    with options():
        plan = application.prepare_plan(source)
        assert len(plan.launches) == 1
        assert plan.launches[0]['code_bytes'] == heads * length
        assert plan.launches[0]['exponent_bytes'] == heads * 4
        assert plan.record()['functional_instruction_limit'] == 128000000
        limited = application.prepare_plan(source, limits=SimpleNamespace(functional_instruction_limit=1000000))
        assert limited.record()['functional_instruction_limit'] == 1000000
        assert len(limited.launches) > 1
    with options(1000000):
        assert application.prepare_plan(source).record() == limited.record()


def test_exponent_slot_io_and_indivisible_group_limits_remain_active():
    with options():
        plan = application.prepare_plan(dynamic_source(shape=(257, 1, 32), reduce='head'))
        assert [x['exponent_bytes'] for x in plan.launches] == [1024, 4]
        for length in (1056, 2016, 4064):
            assert len(application.prepare_plan(dynamic_source(shape=(10, 1, length), reduce='head')).launches) == 10
        with pytest.raises(ValueError, match='complete 1024-element tiles'):
            application.prepare_plan(dynamic_source(shape=(2, 1, 4128), reduce='head'))
    with options(memory=8192):
        assert len(application.prepare_plan(dynamic_source(shape=(10, 1, 1024), reduce='head')).launches) == 5
        with pytest.raises(ValueError, match='I/O window'):
            application.prepare_plan(dynamic_source(shape=(1, 1, 4096), reduce='head'))
    with options(1000000), pytest.raises(ValueError, match='indivisible dynamic reduction group'):
        application.prepare_plan(dynamic_source(shape=(1, 1, 33792), reduce='head'))


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
def test_replayed_budget_and_resource_rejections(tmp_path):
    source = dynamic_source(shape=(12, 1, 1024), reduce='head')
    with options():
        application.compile_source(None, source, target_source=ROOT, destination=tmp_path / 'compiled',
                                   limits=SimpleNamespace(functional_instruction_limit=1000000))
        exported = application.prepare_c_package(source, tmp_path / 'compiled', 'quant', target_source=ROOT)
        assert exported.description['launches_per_invocation'] == 2
        schedule = exported.description['quantizer_schedule']
        assert schedule['functional_instruction_limit'] == 1000000
        application.compile_source(None, source, target_source=ROOT, destination=tmp_path / 'large', limits=None)
        assert application.prepare_c_package(source, tmp_path / 'large', 'large', target_source=ROOT).description['launches_per_invocation'] == 1
    with options(1000000), pytest.raises(ValueError, match='differs from checked source'):
        application.prepare_c_package(source, tmp_path / 'large', 'quant', target_source=ROOT)
    with options(1), pytest.raises(NativeCompilationError, match='instruction bound'):
        device.compile_source(None, source, target_source=ROOT, destination=tmp_path / 'too_many', limits=None)
    with options(program_size_limit_bytes=4), pytest.raises(NativeCompilationError, match='policy limit'):
        device.compile_source(None, source, target_source=ROOT, destination=tmp_path / 'image', limits=None)


@pytest.mark.skipif(not os.environ.get('ATLAS_RUST_NPU_MODEL') or not os.environ.get('ATLAS_OOT_BIN_DIR'),
                   reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('heads,length,rule', [(13, 96, 'nc'), (32, 4096, 'ref'), (9, 32768, 'nc'), (257, 32, 'ref')])
def test_rust_all_codes_exponents_padding_and_guards(tmp_path, heads, length, rule):
    source = dynamic_source(shape=(heads, 1, length), reduce='head', rule=rule)
    with options():
        schedule = application.prepare_plan(source)
        compiled = tmp_path / 'compiled'
        application.compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
        # Different magnitudes in each head catch accidental cross-head amax.
        raw = np.concatenate([np.full(length, 0x3000 + (head % 90) * 128, dtype='<u2')
                              for head in range(heads)])
        raw[::47] = 0x8000
        raw[::83] = 1
        raw[::191] = 0x7f80
        raw[::389] = 0xffc1
        raw = raw.tobytes()
        codes, exponents = bytearray(heads * length), bytearray(heads * 4)
        for launch in schedule.launches:
            index = launch['body_index']
            body = schedule.bodies[index]
            plan = device.prepare_plan(body)
            work = tmp_path / str(launch['input_offset_bytes'])
            work.mkdir()
            initial = bytearray([0xa5]) * ((plan.workspace_bytes + 4095) // 4096 * 4096 + 4096)
            at = launch['input_offset_bytes']
            payload = raw[at:at + launch['input_bytes']]
            initial[:plan.input_bytes] = payload + bytes(plan.input_bytes - len(payload))
            expected = bytearray(initial)
            code, exp = host_reference(body, payload)
            ca, ea = plan.code_address - plan.io_memory_base, plan.exponent_address - plan.io_memory_base
            expected[ca:ea] = code + bytes(ea - ca - len(code))
            expected[ea:plan.workspace_bytes] = exp + bytes(plan.exponent_bytes - len(exp))
            (work / 'memory.bin').write_bytes(initial)
            (work / 'config.json').write_text(json.dumps({'pack_layout': 'register_rows',
                'dram_base': plan.io_memory_base, 'dram_size': len(initial)}))
            program = compiled / f'segments/body_{index:03d}/program.bin'
            assert program.stat().st_size <= 131072
            subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program), '--compute', 'cpu',
                '--config', str(work / 'config.json'), '--load', f'{plan.io_memory_base}:{work / "memory.bin"}',
                '--dump-state', str(work / 'state'), '--max-cycles', '100000000'], check=True, capture_output=True)
            actual = b''.join((work / 'state' / f'dram_{a:016x}.bin').read_bytes()
                             for a in range(plan.io_memory_base, plan.io_memory_base + len(initial), 4096))
            assert actual == expected
            at = launch['code_offset_bytes']
            codes[at:at + len(code)] = actual[ca:ca + len(code)]
            at = launch['exponent_offset_bytes']
            exponents[at:at + len(exp)] = actual[ea:ea + len(exp)]
        assert (bytes(codes), bytes(exponents)) == host_reference(source, raw)
