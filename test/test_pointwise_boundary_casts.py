"""Private activation casts preserve selected arithmetic, ownership and guards."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pytest

from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.pointwise_narrowing import prepare, compile_group
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from test_gelu_precision import source_request
from test_pointwise_narrowing import adapter


def case(shape, kind='gelu', *, input_dtype='f32', output_dtype='bf16', reverse=False):
    source = source_request(shape)
    if kind in ('sigmoid', 'silu'):
        from atlas_mlir_native.sigmoid import SOURCE_ATTRS
        source = replace(source, nodes=(source.nodes[0], replace(source.nodes[1], attrs=SOURCE_ATTRS)))
    if kind == 'silu':
        x, sigmoid = source.nodes
        product = SemanticNode('product', 'f32_mul', (sigmoid.id, x.id) if reverse else (x.id, sigmoid.id),
            x.type, (('numeric_semantics', 'ieee754-f32-mul-rne-v1'),),
            index_maps=pointwise_identity_maps(len(shape), 2))
        source = replace(source, nodes=(*source.nodes, product), outputs=(product.id,))
    value = source.nodes[-1]
    cast = SemanticNode('narrowed', 'f32_to_bf16', (value.id,), replace(value.type, dtype='bf16'),
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),), index_maps=pointwise_identity_maps(len(shape), 1))
    if output_dtype == 'bf16':
        source = replace(source, nodes=(*source.nodes, cast), outputs=(cast.id,))
    if input_dtype == 'bf16':
        x = source.nodes[0]
        operand = replace(x, id='codes', type=replace(x.type, dtype='bf16'))
        widened = replace(x, op='bf16_to_f32', inputs=(operand.id,), effect='pure',
            attrs=(('numeric_semantics', 'bf16-to-f32-exact-v1'),),
            index_maps=pointwise_identity_maps(len(shape), 1))
        source = replace(source, nodes=(operand, widened, *source.nodes[1:]), input_storages=((operand.id, 'external'),))
    return source


@pytest.mark.parametrize('shape', [(1, 7), (33, 65), (2, 3, 35)])
def test_selection_keeps_original_gelu_and_requires_qualified_policy(shape):
    from atlas_mlir_native.gelu import prepare_source
    from merlin.compile.semantic_regions import extract_semantic_region
    source = case(shape)
    base = adapter()
    group = prepare(source, base)
    assert group.precision == prepare_source(extract_semantic_region(source, ('y',)))
    selected, = AdjacentDeviceAdapter(base, patterns=('pointwise_narrowing',)).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'pointwise_narrowing'
    base.gelu = False
    with pytest.raises(ValueError, match='non-NaN'):
        prepare(source, base)
    escaped = replace(source, outputs=('y', 'narrowed'), output_storages=('external',)*2)
    with pytest.raises(ValueError):
        prepare(escaped, adapter())
    changed = replace(source.nodes[1], attrs=tuple((key, value.replace('3d372713', '3d372714'))
                      for key, value in source.nodes[1].attrs))
    with pytest.raises(ValueError):
        prepare(replace(source, nodes=(source.nodes[0], changed, source.nodes[2])), adapter())


def independent_unary(codes, kind):
    """Replay the selected operations through the model's arithmetic primitives."""
    import torch
    from npu_model.configs.numerics import RtlNumerics as rtl

    value = torch.from_numpy(codes.copy()).view(torch.bfloat16)
    constant = lambda bits: torch.from_numpy(np.full(codes.shape, bits, '<u2')).view(torch.bfloat16)
    if kind == 'gelu':
        cubic = rtl.mul(rtl.unary('cube', value), constant(0x3d37))
        inner = rtl.mul(rtl.add(value, cubic), constant(0x3f4c))
        outer = rtl.add(constant(0x3f80), rtl.unary('tanh', inner))
        result = rtl.mul(rtl.mul(value, constant(0x3f00)), outer)
    else:
        denominator = rtl.add(constant(0x3f80), rtl.unary('exp', rtl.sub(constant(0), value)))
        result = rtl.unary('rcp', denominator)
        if kind == 'silu':
            result = rtl.mul(value, result)
    return result.contiguous().view(torch.uint16).numpy().copy()


@pytest.mark.skipif(not all(os.environ.get(key) for key in ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR')),
                    reason='select Rust model and Atlas tools')
@pytest.mark.parametrize('shape,kind,input_dtype,output_dtype', [
    ((33,35), 'gelu', 'f32', 'bf16'), ((2,3,35), 'gelu', 'f32', 'bf16'),
    ((129,128), 'gelu', 'f32', 'bf16'), ((137,128), 'silu', 'f32', 'bf16'),
    ((33,35), 'sigmoid', 'f32', 'bf16'), ((129,128), 'gelu', 'bf16', 'bf16'),
    ((2,3,35), 'gelu', 'bf16', 'f32'), ((137,128), 'silu', 'bf16', 'bf16'),
    ((33,35), 'silu', 'bf16', 'f32'), ((1,7), 'sigmoid', 'bf16', 'f32')])
def test_generated_c_rust_all_codes_tails_and_atomic_refusals(tmp_path, shape, kind, input_dtype, output_dtype):
    from atlas_mlir_native.pointwise import prepare_plan
    from atlas_mlir_native.pointwise_c import prepare_precision_c_package
    from merlin.semantic_compiler.pointwise_tiling import materialize_pointwise_input
    from merlin.semantic_compiler.search import SearchLimits
    from test_looped_mnk import build_snapshot, ROOT
    from test_pointwise_loop import check_source_c_call

    from atlas_mlir_native.pointwise_input_widening import prepare as prepare_inputs
    source = case(shape, kind, input_dtype=input_dtype, output_dtype=output_dtype)
    group = (prepare_inputs(source, adapter(), output_dtype=output_dtype) if input_dtype == 'bf16'
             else prepare(source, adapter()))
    compiled = tmp_path/'compiled'
    compile_group(group, build_snapshot(tmp_path/'snapshot'), target_source=ROOT,
                  destination=compiled, limits=SearchLimits())
    record, files = prepare_precision_c_package(group.precision, compiled, 'pointwise',
        target_source=ROOT, runtime_linkage='embedded', output_dtype=output_dtype,
        input_dtype='bf16' if input_dtype == 'bf16' else None)
    assert record.get('output_dtype', input_dtype) == output_dtype
    package = tmp_path/'package'
    package.mkdir()
    for name, data in files.items():
        (package/name).write_bytes(data)
    lower, upper = (0x2f80, 0x4f80) if kind == 'gelu' else (0x2000, 0x4200)
    all_codes = np.array([0, 0x8000, *(sign|code for sign in (0, 0x8000)
                                     for code in range(lower, upper+1))], '<u2')
    codes = np.resize(all_codes, shape)
    values = codes.copy() if input_dtype == 'bf16' else (codes.astype('<u4') << 16).view('<f4')
    wanted = independent_unary(codes, kind)
    assert np.all((wanted & 0x7f80) != 0x7f80)
    if output_dtype == 'f32':
        wanted = wanted.astype('<u4') << 16
    plan = prepare_plan(group.precision.request)
    execution = json.loads((compiled/'execution_plan.json').read_text())
    end = execution['output_addresses'][0] + execution['output_bytes'] - DRAM_BASE
    initial = bytearray([0xa5]) * ((end+4095)//4096*4096+4096)
    payload = materialize_pointwise_input(plan, 0, codes.tobytes())
    payload = np.frombuffer(payload, '<u2').reshape(-1,32,2,16).transpose(0,2,1,3).copy().tobytes()
    at = execution['input_addresses'][0] - DRAM_BASE
    streaming = execution.get('streaming')
    check_inputs = None
    launches = []
    if streaming:
        chunk = execution['input_buffer_bytes'][0]
        chunks = [payload[i:i+chunk].ljust(chunk, b'\0') for i in range(0, len(payload), chunk)]
        assert len(chunks) == streaming['launch_count']
        def check_inputs(memory):
            expected = chunks[len(launches) % len(chunks)]
            assert memory[at:at+chunk] == expected
            launches.append(True)
    else:
        initial[at:at+len(payload)] = payload
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(compute='cpu', pack_layout='register_rows', dram_base=DRAM_BASE,
                                     dram_size=len(initial))))
    def execute(memory):
        with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
            directory = Path(directory)
            (directory/'initial.bin').write_bytes(memory)
            subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(compiled/'program.bin'),
                '--config', str(config), '--load', f'{DRAM_BASE}:{directory/"initial.bin"}',
                '--dump-state', str(directory/'state')], check=True, capture_output=True)
            actual = b''.join((directory/'state'/f'dram_{address:016x}.bin').read_bytes()
                             for address in range(DRAM_BASE, DRAM_BASE+len(memory),4096))
            begin = execution['output_addresses'][0] - DRAM_BASE
            assert actual[:begin] == memory[:begin] and actual[end:] == memory[end:]
            return actual
    invalid = []
    for value in (float('nan'), float('inf'), 2.0**33, 2.0**-64):
        bad = values.copy()
        bad.flat[-1] = int(np.array(value, '<f4').view('<u4')) >> 16 if input_dtype == 'bf16' else value
        invalid.append((bad,))
    check_source_c_call(unittest.TestCase(), package, initial, (compiled/'program.bin').read_bytes(), execute,
                        (values,), wanted.tobytes(), invalid, output_dtype=output_dtype,
                        source_dtype='opaque' if input_dtype == 'bf16' else 'f32',
                        check_launch_inputs=check_inputs)
    if streaming:
        assert len(launches) == 2 * streaming['launch_count']


@pytest.mark.parametrize('kind', ['gelu', 'sigmoid', 'silu'])
@pytest.mark.parametrize('output_dtype', ['bf16', 'f32'])
def test_raw_input_selection_and_private_boundaries(kind, output_dtype):
    from atlas_mlir_native.pointwise_input_widening import prepare as prepare_inputs
    source = case((2,3,35), kind, input_dtype='bf16', output_dtype=output_dtype)
    group = prepare_inputs(source, adapter(), output_dtype=output_dtype)
    assert group.input_id == 'codes' and group.output_dtype == output_dtype
    selected, = AdjacentDeviceAdapter(adapter(), patterns=('pointwise_narrowing', 'pointwise_input_widening')).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'pointwise_input_widening'
    escaped = replace(source, outputs=('x', *source.outputs), output_storages=('external',)*2)
    with pytest.raises(ValueError):
        prepare_inputs(escaped, adapter(), output_dtype=output_dtype)


@pytest.mark.parametrize('reverse', [False, True])
def test_selected_silu_product_order_and_cast_ownership(reverse):
    source = case((33,35), 'silu', reverse=reverse)
    group = prepare(source, adapter())
    assert group.precision.source.nodes[-1].inputs == source.nodes[-2].inputs
    selected, = AdjacentDeviceAdapter(adapter(), patterns=('pointwise_narrowing',)).plan(source).selected
    assert len(selected.nodes) == 3
    base = adapter()
    base.sigmoid = False
    with pytest.raises(ValueError, match='non-NaN'):
        prepare(source, base)


@pytest.mark.parametrize('change', [
    {'max_normal_exponent_code': 200},
    {'flush_below_normal': True},
    {'canonicalize_negative_zero': True},
    {'pre_encode_scale_exponent': 1},
    {'column_scale_exponents': ((0, 35, 1),)},
])
def test_raw_input_refuses_encoders_that_change_values(monkeypatch, change):
    from atlas_mlir_native.gelu import GeluPrecision, input_spec
    from atlas_mlir_native.pointwise_input_widening import prepare as prepare_inputs

    spec = replace(input_spec(), **change)
    monkeypatch.setattr(GeluPrecision, 'operand_specs', property(lambda self: (spec,)))
    source = case((2, 3, 35), input_dtype='bf16')
    with pytest.raises(ValueError, match='clamping|numerical conversion'):
        prepare_inputs(source, adapter(), output_dtype='bf16')


@pytest.mark.skipif(not all(os.environ.get(key) for key in ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR')),
                    reason='select Rust model and Atlas tools')
def test_raw_input_streaming_keeps_tail_and_late_failure_atomic(tmp_path):
    from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options

    with codegen_options(CodegenOptions(io_memory_bytes=65536)):
        test_generated_c_rust_all_codes_tails_and_atomic_refusals(
            tmp_path, (129, 129), 'gelu', 'bf16', 'bf16')
    execution = json.loads((tmp_path/'compiled/execution_plan.json').read_text())
    assert execution['streaming']['launch_count'] > 1
