"""Large static quantizers retain bounded counted bodies and exact bytes."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from atlas_mlir_native import activation_quantize_application as application
from atlas_mlir_native import vector_quantize as device
from test_activation_quantize_device import host_reference
from test_already_quantized import quantizer_source
from test_quantizer_dispatch_budget import options

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('count', [1056, 524256, 786432, 12582912])
def test_large_aligned_tensor_uses_selected_budget(count):
    source = quantizer_source(count=count)
    with options():
        plan = application.prepare_plan(source)
        assert len(plan.launches) == 1
        assert plan.launches[0]['code_bytes'] == count
        assert plan.record()['functional_instruction_limit'] == 128000000
    with options(1000000):
        limited = application.prepare_plan(source)
        assert sum(row['code_bytes'] for row in limited.launches) == count
        assert all(50 * (row['code_bytes'] // 32) + 128 <= 1000000
                   for row in limited.launches)
        if count > 640000:
            assert len(limited.launches) > 1


def test_memory_capacity_and_instruction_capacity_keep_aligned_remainders():
    for budget, memory in ((1000000, 8192), (10000, 67108864), (1000, 67108864)):
        with options(budget, memory):
            plan = application.prepare_plan(quantizer_source(count=65568))
            at = 0
            for launch in plan.launches:
                count = launch['code_bytes']
                assert launch['code_offset_bytes'] == at
                assert launch['input_offset_bytes'] == at * 2
                assert launch['exponent_offset_bytes'] == 0
                assert count % 32 == 0
                assert 3 * count + 32 <= memory
                assert 50 * (count // 32) + 128 <= budget
                at += count
            assert at == 65568
    with options(177), pytest.raises(ValueError, match='cannot fit a static quantizer tile'):
        application.prepare_plan(quantizer_source(count=32))


@pytest.mark.skipif(not os.environ.get('ATLAS_RUST_NPU_MODEL') or not os.environ.get('ATLAS_OOT_BIN_DIR'),
                   reason='select Rust simulator and Atlas compiler')
@pytest.mark.parametrize('count,budget,memory', [
    (1056, 128000000, 67108864),
    (786432, 128000000, 67108864),
    (12582912, 128000000, 67108864),
    (65568, 1000000, 8192),
])
def test_compiled_chunks_match_all_codes_exponents_and_memory_guards(tmp_path, count, budget, memory):
    source = quantizer_source(count=count, fixed=-8)
    # Include every BF16 encoding, signed zeros, infinities and NaN payloads.
    raw = np.resize(np.arange(65536, dtype='<u2'), count).tobytes()
    with options():
        expected_source = host_reference(source, raw)
    with options(budget, memory):
        schedule = application.prepare_plan(source)
        compiled = tmp_path / 'compiled'
        application.compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
        exported = application.prepare_c_package(source, compiled, 'quant', target_source=ROOT)
        assert exported.description['launches_per_invocation'] == len(schedule.launches)
        codes = bytearray(count)
        exponent = None
        for index, launch in enumerate(schedule.launches):
            work = tmp_path / f'launch-{index}'
            work.mkdir()
            body = schedule.bodies[launch['body_index']]
            plan = device.prepare_plan(body)
            first = launch['input_offset_bytes']
            payload = raw[first:first + launch['input_bytes']]
            initial = bytearray([0xa5]) * ((plan.workspace_bytes + 4095) // 4096 * 4096 + 4096)
            initial[:plan.input_bytes] = payload
            expected = bytearray(initial)
            code, exp = host_reference(body, payload)
            ca = plan.code_address - plan.io_memory_base
            ea = plan.exponent_address - plan.io_memory_base
            expected[ca:ea] = code + bytes(ea - ca - len(code))
            expected[ea:plan.workspace_bytes] = exp + bytes(plan.exponent_bytes - len(exp))
            (work / 'memory.bin').write_bytes(initial)
            (work / 'config.json').write_text(json.dumps(dict(pack_layout='register_rows',
                dram_base=plan.io_memory_base, dram_size=len(initial))))
            program = compiled / f"segments/body_{launch['body_index']:03d}/program.bin"
            assert program.stat().st_size <= 131072
            subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program), '--compute', 'cpu',
                '--config', str(work / 'config.json'), '--load', f'{plan.io_memory_base}:{work / "memory.bin"}',
                '--dump-state', str(work / 'state'), '--max-cycles', '100000000'],
                check=True, capture_output=True)
            actual = b''.join((work / 'state' / f'dram_{address:016x}.bin').read_bytes()
                             for address in range(plan.io_memory_base, plan.io_memory_base + len(initial), 4096))
            assert actual == expected
            at = launch['code_offset_bytes']
            codes[at:at + len(code)] = actual[ca:ca + len(code)]
            if exponent is not None:
                assert exponent == actual[ea:ea + len(exp)]
            exponent = actual[ea:ea + len(exp)]
        assert (bytes(codes), exponent) == expected_source
