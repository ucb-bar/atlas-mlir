"""Tile-local validation may write only explicitly private device results."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from atlas_mlir_native import exact_float_application as application
from atlas_mlir_native import exact_float_device as device
from merlin.compiler.semantic.codegen import codegen_options, current_codegen_options
from test_exact_float_device import host_reference, request

ROOT = Path(__file__).resolve().parents[1]


def test_private_ownership_is_explicit_and_limited_to_binary_leaves():
    source = request('bf16_add', 2048)
    raw = device.prepare_plan(source)
    assert not raw.private_output and 'private_output' not in raw.record()
    private = device.prepare_plan(source, private_output=True)
    assert private.record()['private_output'] is True
    assert 'private_complete' in device.virtual_program(private)
    assert 'private_complete' not in device.virtual_program(raw)
    for value in (1, 'true', None):
        with pytest.raises(ValueError):
            device.prepare_plan(source, private_output=value)
    with pytest.raises(ValueError):
        device.prepare_plan(request('f32_to_bf16', 32), private_output=True)


@pytest.mark.skipif(not all(os.environ.get(k) for k in ('ATLAS_RUST_NPU_MODEL','ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('operation,count,fast,shared', [
    ('bf16_add', 35, False, False), ('bf16_sub', 35, True, False),
    ('bf16_add', 2048, True, False), ('bf16_sub', 2048, True, False),
    ('bf16_add', 2048, False, False), ('bf16_sub', 2048, True, True)])
def test_exact_rust_results_read_once_and_raw_rejection_is_unchanged(tmp_path, operation, count, fast, shared):
    source = request(operation, count, shared=shared)
    options = replace(current_codegen_options(), io_memory_bytes=67108864,
        execution_instruction_limit=4000000, exact_binary_vector_preflight=fast,
        exact_binary_lane_groups=fast, exact_binary_routes=fast)
    with codegen_options(options):
        programs = []
        for private in (False, True):
            compiled = tmp_path / str(private)
            plan = device.prepare_plan(source, private_output=private)
            device.compile_source(None, source, target_source=ROOT, destination=compiled,
                                  limits=None, private_output=private)
            programs.append((plan, compiled))
            assert (compiled/'program.bin').stat().st_size <= 131072
            # Replaying a program under a different ownership contract refuses.
            with pytest.raises(ValueError, match='descriptor'):
                device.prepare_c_package(source, compiled, 'wrong', target_source=ROOT,
                                         private_output=not private)
        cases = [(pair, count-1) for pair in (
            (0x3f80,0x3f00), (0x8000,0x8000), (1,127), (0x2000,0x1c00),
            (0x7f7f,0x7f7f), (0x7f80,0x3f80), (0x7fc1,0x3f80), (0x7f80,0xff80))]
        cases.append(((0x7fc1,0x3f80), 0))
        for trial, (pair, position) in enumerate(cases):
            raw = [word.to_bytes(2,'little')*count for word in (0x3f80,0x3f00)]
            for i, word in enumerate(pair):
                # Exercise fallback/domain cases at the end and an early
                # invalid lane. Partial tiles have private padding.
                raw[i] = raw[i][:2*position] + word.to_bytes(2,'little') + raw[i][2*position+2:]
            if shared:
                raw=raw[:1]
            expected, status = host_reference(source, raw)
            stats = []
            for plan, compiled in programs:
                work = compiled / str(trial)
                work.mkdir()
                initial = bytearray([0xa5])*((plan.workspace_bytes+4095)//4096*4096)
                for i, data in enumerate(raw):
                    at=i*plan.input_bytes
                    initial[at:at+plan.input_bytes]=bytes(plan.input_bytes)
                    initial[at:at+len(data)]=data
                (work/'input.bin').write_bytes(initial)
                result = subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'],str(compiled/'program.bin'),
                    '--compute','cpu','--load',f'{plan.io_memory_base}:{work/"input.bin"}',
                    '--dump-state',str(work/'state'),'--max-cycles','10000000'],
                    check=True,capture_output=True,text=True)
                stats.append(json.loads(result.stdout))
                memory=b''.join((work/'state'/f'dram_{a:016x}.bin').read_bytes()
                    for a in range(plan.io_memory_base,plan.io_memory_base+len(initial),4096))
                output=plan.output_address-plan.io_memory_base
                code=plan.status_address-plan.io_memory_base
                assert bool(int.from_bytes(memory[code:code+4],'little')) == bool(status)
                assert memory[:output] == initial[:output]
                assert memory[code+32:] == initial[code+32:]
                if not status:
                    assert memory[output:output+len(expected)] == expected
                elif not plan.private_output:
                    assert memory[output:code] == initial[output:code]
                elif count > 1024 and position >= 1024:
                    assert memory[output:output+2048] != initial[output:output+2048]
                    assert memory[output+2048:code] == initial[output+2048:code]
                else:
                    assert memory[output:code] == initial[output:code]
            if count > 1024 and not status:
                assert stats[0]['engine_instructions'][5]-stats[1]['engine_instructions'][5] == len(plan.operands)*(plan.input_bytes//(plan.tile_lanes*plan.width))


def test_application_replay_preserves_legacy_and_private_contracts(tmp_path):
    if not os.environ.get('ATLAS_OOT_BIN_DIR'):
        pytest.skip('select Atlas tools')
    source = request('bf16_add', 1059)
    from atlas_mlir_native.application_materialization import bind_exact_float
    for private in (False, True):
        compiled=tmp_path/str(private)
        application.compile_source(None,source,target_source=ROOT,destination=compiled,
                                   limits=None,private_output=private)
        package=application.prepare_c_package(source,compiled,'checked',target_source=ROOT)
        assert package.outputs == source.outputs
        rebound=tmp_path/('rebound-'+str(private))
        bind_exact_float(source,source,compiled,rebound)
        assert application.prepare_c_package(source,rebound,'checked',target_source=ROOT) == package
        schedule=json.loads((compiled/'float-schedule.json').read_text())
        assert schedule.get('private_output',False) == private
        # Rehashing the parent flag cannot relabel the native child programs.
        schedule['private_output']=not private
        path=compiled/'float-schedule.json'
        path.write_text(json.dumps(schedule))
        from hashlib import sha256
        manifest_path=compiled/'float-manifest.json'
        manifest=json.loads(manifest_path.read_text())
        manifest['stage_artifacts']['float-schedule.json']=sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            application.prepare_c_package(source,compiled,'wrong',target_source=ROOT)
        with pytest.raises(ValueError):
            bind_exact_float(source,source,compiled,tmp_path/('wrong-'+str(private)))


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
def test_private_results_do_not_bypass_execution_or_image_limits(tmp_path):
    from merlin.compiler.semantic.target_binding import NativeCompilationError
    source = request('bf16_add', 2048)
    with codegen_options(replace(current_codegen_options(), execution_instruction_limit=4000000)):
        with pytest.raises(NativeCompilationError, match='instruction bound exceeds 1:'):
            device.compile_source(None, source, target_source=ROOT, destination=tmp_path/'execution',
                limits=SimpleNamespace(functional_instruction_limit=1), private_output=True)
    with codegen_options(replace(current_codegen_options(), execution_instruction_limit=4000000,
                                program_size_limit_bytes=4)):
        with pytest.raises(NativeCompilationError, match='program.*budget|program.*limit'):
            device.compile_source(None, source, target_source=ROOT, destination=tmp_path/'image',
                                  limits=None, private_output=True)
    assert not (tmp_path/'execution').exists()
    assert not (tmp_path/'image').exists()
