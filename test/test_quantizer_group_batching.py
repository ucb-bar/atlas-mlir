"""Token groups keep independent exponents when sharing a bounded launch."""
import ctypes
from dataclasses import replace
import json
import os
from unittest.mock import patch

import pytest

from atlas_mlir_native.activation_quantize_application import prepare_plan
from atlas_mlir_native.options import codegen_options, current_codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from atlas_mlir_native.work_budget import CountedWorkExhausted
from test_activation_quantize_device import dynamic_source, host_reference


def test_group_batching_is_explicit_and_keeps_each_source_slice():
    source = dynamic_source(shape=(2, 17, 32), reduce='token')
    with patch('atlas_mlir_native.activation_quantize_device.prove_plan'):
        legacy = prepare_plan(source)
        assert len(legacy.launches) == 34
        assert 'dynamic_group_batching' not in legacy.record()
        with codegen_options(replace(current_codegen_options(), dynamic_quantizer_group_batching=True)):
            selected = prepare_plan(source)
        assert selected.record()['dynamic_group_batching'] is True
        assert [body.node('x').type.shape for body in selected.bodies] == [(34,32)]
        assert [row['exponent_bytes'] for row in selected.launches] == [136]
        assert [row['input_offset_bytes'] for row in selected.launches] == [0]
        assert [row['code_offset_bytes'] for row in selected.launches] == [0]
        assert [row['exponent_offset_bytes'] for row in selected.launches] == [0]
        with pytest.raises(ValueError, match='Boolean'):
            prepare_plan(source, _group_batching=1)


def test_failed_resource_proofs_reduce_group_count_but_other_errors_propagate():
    source = dynamic_source(shape=(10,32), reduce='token')
    calls = []

    def prove(plan, **kwargs):
        calls.append(plan.count)
        if plan.count > 96:
            raise NativeCompilationError('resource_limit', 'selected image exceeds its bound')

    with codegen_options(replace(current_codegen_options(), dynamic_quantizer_group_batching=True)):
        with patch('atlas_mlir_native.activation_quantize_device.prove_plan', side_effect=prove):
            selected = prepare_plan(source)
        assert [row['code_bytes'] for row in selected.launches] == [64]*5
        assert calls == [320,160,64]
        with patch('atlas_mlir_native.activation_quantize_device.prove_plan',
                   side_effect=NativeCompilationError('compile_error', 'bad dependency proof')):
            with pytest.raises(NativeCompilationError, match='bad dependency'):
                prepare_plan(source)
        with patch('atlas_mlir_native.activation_quantize_device.prove_plan',
                   side_effect=CountedWorkExhausted('solver work limit')) as proof:
            with pytest.raises(CountedWorkExhausted, match='solver work limit'):
                prepare_plan(source)
            assert proof.call_count == 1


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
def test_real_instruction_bound_reuses_an_admitted_batch():
    from atlas_mlir_native.activation_quantize_device import prepare_plan as body_plan, prove_plan

    source = dynamic_source(shape=(256,1024), reduce='token')
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=1048576,
                                 native_pipeline='mandatory', dynamic_quantizer_group_batching=True)):
        schedule = prepare_plan(source)
        assert len(schedule.launches) == 8
        assert len(schedule.bodies) == 1
        bound = prove_plan(body_plan(schedule.bodies[0]))[3]
        assert bound['admitted'] and bound['instruction_bound'] <= 1000000
        assert sum(row['code_bytes'] for row in schedule.launches) == schedule.count


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline', ['mandatory', 'optimized'])
@pytest.mark.parametrize('rule', ['ref', 'nc'])
@pytest.mark.parametrize('shape,launches', [((259,32),2), ((2,17,96),1)])
def test_exported_application_keeps_group_exponents_special_values_and_tails(tmp_path, pipeline, rule, shape, launches):
    from atlas_mlir_native.application import AtlasApplicationAdapter
    from atlas_mlir_native.binding import DRAM_BASE
    from merlin.compile.semantic_application import compile_semantic_application, verify_semantic_application
    from merlin.semantic_compiler.search import SearchLimits
    from test_already_quantized import _policy
    from test_looped_mnk import ROOT, build_snapshot
    from rust_model import Application

    source = dynamic_source(shape=shape, reduce='token', rule=rule)
    adapter = AtlasApplicationAdapter({**_policy(), 'activation_quantizer': 'device'},
                                     original_source_sha256='0'*64)
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=1048576,
                                 native_pipeline=pipeline, dynamic_quantizer_group_batching=True)):
        schedule = prepare_plan(source)
        assert len(schedule.launches) == launches
        snapshot = build_snapshot(tmp_path)
        artifact = tmp_path / 'application'
        manifest = compile_semantic_application(source, adapter=adapter, snapshot=snapshot,
            target_source=ROOT, destination=artifact, limits=SearchLimits(), name='group_quant', capacity=65536)
        assert len(manifest['regions']) == 1
        assert verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
            target_source=ROOT, artifact=artifact)['status'] == 'checked'
    # Different row maxima expose accidental reduction across adjacent groups.
    special = (0, 0x8000, 1, 0x807f, 0x7f80, 0xff80, 0x7fc1, 0xffa3)
    raw = bytearray()
    for index in range(schedule.count):
        row, lane = divmod(index, shape[-1])
        code = (special[lane] if row % 7 == 0 and lane < len(special) else
                (0x3b00 + (row % 11)*128 + (lane*17)%128) | ((lane % 3 == 0) << 15))
        raw.extend(code.to_bytes(2, 'little'))
    expected_codes, expected_exponents = host_reference(source, bytes(raw))
    application = Application(artifact / 'c-package', 'group_quant', tmp_path / 'runtime',
                              base=DRAM_BASE-64, size=1048704)
    argument = ctypes.create_string_buffer(bytes(raw))
    codes = ctypes.create_string_buffer(bytes([0x7a])*(len(expected_codes)+64))
    exponents = ctypes.create_string_buffer(bytes([0x7a])*(len(expected_exponents)+64))
    application.invoke([argument], [codes, exponents])
    assert len(application.stats) == launches
    # Device DMA padding is private; the host reads only public code/exponent
    # bytes, including the last body with a non-beat-sized exponent result.
    assert application.download_bytes == len(expected_codes) + len(expected_exponents)
    assert argument.raw[:len(raw)] == raw
    for actual, expected in ((codes, expected_codes), (exponents, expected_exponents)):
        assert actual.raw[:len(expected)] == expected
        assert actual.raw[len(expected):len(expected)+64] == bytes([0x7a])*64
    assert application.memory[:64] == bytes([0xa5])*64
    saved = json.loads((artifact / 'segments/region_000/quantizer-schedule.json').read_text())
    assert saved['dynamic_group_batching'] is True and saved['functional_instruction_limit'] == 1000000
    from atlas_mlir_native.activation_quantize_application import prepare_c_package
    from merlin.semantic_compiler.model import KernelRequest
    region = KernelRequest.from_record(json.loads(
        (artifact / 'segments/region_000/source-request.json').read_text()))
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=1048576,
                                 dynamic_quantizer_group_batching=False,
                                 native_pipeline='mandatory' if pipeline == 'optimized' else 'optimized')):
        replay = prepare_c_package(region, artifact / 'segments/region_000', 'replay', target_source=ROOT)
    assert replay.description['launches_per_invocation'] == launches
