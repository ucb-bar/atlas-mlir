"""Fused pointwise static PACK retains source rounding, tails and ABI guards."""
from dataclasses import replace

import numpy as np
import pytest

from atlas_mlir_native import pointwise_quantize as fused
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.pointwise import BF16_POINTWISE_POLICY
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps, IndexMap
from test_already_quantized import quantizer_source
from test_pointwise_narrowing import adapter as base_adapter, case as narrowed_case
from test_bf16_pointwise import request as bf16_case
from test_pointwise_boundary_casts import case as unary_case, independent_unary

PATTERNS = ('pointwise_narrowing', 'pointwise_input_widening', 'pointwise_quantize')


def adapter():
    result = base_adapter()
    result.bf16_pointwise = BF16_POINTWISE_POLICY
    return result


def case(shape=(3, 35), kind='pointwise', exponent=-7, broadcast=False):
    if kind == 'pointwise':
        source = bf16_case(shape)
    elif kind == 'pointwise_narrowing':
        source = narrowed_case(shape=shape)
    else:
        source = unary_case(shape, input_dtype='bf16')
    if broadcast:
        a, b, op, *rest = source.nodes
        b = replace(b, type=replace(b.type, shape=(shape[-1],)))
        mapping = IndexMap(len(shape), (tuple([0]*(len(shape)-1) + [1]),), (0,))
        op = replace(op, index_maps=(op.index_maps[0], mapping, op.index_maps[-1]))
        source = replace(source, nodes=(a, b, op, *rest))
    value = source.node(source.outputs[0])
    quantizer = quantizer_source(fixed=exponent)
    codes, scale = quantizer.nodes[-2:]
    codes = replace(codes, id='packed', inputs=(value.id,), type=replace(codes.type, shape=shape))
    scale = replace(scale, inputs=(value.id,))
    return replace(source, nodes=(*source.nodes, codes, scale), outputs=(codes.id, scale.id),
                   output_storages=('external',)*2)


@pytest.mark.parametrize('shape', [(1, 7), (33, 65), (2, 3, 35)])
@pytest.mark.parametrize('kind', ['pointwise', 'pointwise_narrowing', 'pointwise_input_widening'])
def test_selector_fuses_only_private_bf16_result(shape, kind):
    source = case(shape, kind)
    group = fused.prepare(source, adapter(), kind)
    selected, = AdjacentDeviceAdapter(adapter(), patterns=PATTERNS).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'pointwise_quantize'
    assert group.exponent == -7
    assert group.input_ids == tuple(n.id for n in source.nodes if n.effect == 'input')


def test_shared_result_dynamic_scale_and_changed_rounding_refuse():
    source = case(kind='pointwise_narrowing')
    producer = source.nodes[-3]
    wrapper = AdjacentDeviceAdapter(adapter(), patterns=PATTERNS)
    shared = replace(source, outputs=(*source.outputs, producer.id), output_storages=('external',)*3)
    extra = SemanticNode('extra', 'copy', (producer.id,), producer.type, index_maps=pointwise_identity_maps(2, 1))
    reader = replace(source, nodes=(*source.nodes, extra), outputs=(*source.outputs, extra.id),
                     output_storages=('external',)*3)
    for request in (shared, reader):
        assert all(dict(r.metadata)['kind'] != 'pointwise_quantize' for r in wrapper.plan(request).selected)
    for key, value in [('mode', 'dynamic'), ('static_exp', True), ('static_exp', 1 << 31), ('rule', 'unknown')]:
        nodes = tuple(replace(n, attrs=tuple(sorted(dict(n.attrs, **{key: value}).items())))
                      if n.op.startswith('atlas.quantize_act.') else n for n in source.nodes)
        with pytest.raises(ValueError):
            fused.prepare(replace(source, nodes=nodes), adapter(), 'pointwise_narrowing')
    changed = replace(producer, attrs=())
    with pytest.raises(ValueError):
        fused.prepare(replace(source, nodes=(*source.nodes[:-3], changed, *source.nodes[-2:])),
                      adapter(), 'pointwise_narrowing')
    base = adapter()
    base.activation_quantizer_route = 'host'
    assert all(dict(r.metadata)['kind'] != 'pointwise_quantize'
               for r in AdjacentDeviceAdapter(base, patterns=PATTERNS).plan(source).selected)


def test_streamed_frames_refuse_the_single_dispatch_fusion():
    from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
    with codegen_options(CodegenOptions(io_memory_bytes=65536)):
        with pytest.raises(ValueError, match='single resident frame'):
            fused.prepare(case((513, 513)), adapter(), 'pointwise')


@pytest.mark.skipif(not all(os.environ.get(k) for k in ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR')),
                    reason='select Rust model and Atlas tools')
@pytest.mark.parametrize('shape,kind,exponent,broadcast', [
    ((1,7), 'pointwise', -7, False), ((33,65), 'pointwise', -8, True),
    ((2,3,35), 'pointwise_narrowing', -6, False),
    ((33,35), 'pointwise_narrowing', -7, True),
    ((2,32), 'pointwise', -(1 << 31), False), ((32,32), 'pointwise', (1 << 31)-1, False),
    ((33,35), 'pointwise_input_widening', -7, False),
])
def test_rust_c_exact_packed_outputs_and_atomic_refusals(tmp_path, shape, kind, exponent, broadcast):
    from atlas_mlir_native.binding import DRAM_BASE
    from atlas_mlir_native.pointwise import prepare_plan
    from atlas_mlir_native.pointwise_c import prepare_precision_c_package
    from atlas_mlir_native.static_quantize import pack_bf16
    from merlin.semantic_compiler.pointwise_tiling import materialize_pointwise_input
    from merlin.semantic_compiler.search import SearchLimits
    from test_looped_mnk import build_snapshot
    from test_pointwise_loop import check_source_c_call

    group = fused.prepare(case(shape, kind, exponent, broadcast), adapter(), kind)
    compiled = tmp_path/'compiled'
    fused.compile_group(group, build_snapshot(tmp_path/'snapshot'), target_source=ROOT,
                        destination=compiled, limits=SearchLimits())
    for wrong in (None, exponent+1 if exponent < (1 << 31)-1 else exponent-1):
        with pytest.raises(ValueError):
            prepare_precision_c_package(group.precision, compiled/'pointwise', 'pointwise',
                target_source=ROOT, input_dtype=group.input_dtype, output_dtype='i8', static_output_exponent=wrong)
    exported = fused.export_group(group, compiled, 'pointwise', target_source=ROOT)
    package = tmp_path/'package'
    for name, data in exported.files.items():
        path = package/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with (package/'pointwise.c').open('a') as stream:
        stream.write('\n#include "runtime/atlas_host.c"\n')
    a = np.resize(np.array([-2, -0.0, 0, .5, 1, 2, 3.140625], '<f4'), shape)
    codes_a = (a.view('<u4') >> 16).astype('<u2')
    if kind == 'pointwise_input_widening':
        wanted_bf16 = independent_unary(codes_a, 'gelu')
        values, operands = (codes_a.copy(),), (codes_a,)
    else:
        b = np.resize(np.array([.5, 1.0078125, -3.140625, 0, -0.0], '<f4'),
                      (shape[-1],) if broadcast else shape)
        codes_b = (b.view('<u4') >> 16).astype('<u2')
        import torch
        from npu_model.configs.numerics import RtlNumerics
        wanted_bf16 = RtlNumerics.mul(torch.from_numpy(codes_a.copy()).view(torch.bfloat16),
            torch.from_numpy(codes_b.copy()).view(torch.bfloat16)).contiguous().view(torch.uint16).numpy()
        values = (a, b) if kind == 'pointwise_narrowing' else (codes_a.copy(), codes_b.copy())
        operands = (codes_a, codes_b)
    wanted = pack_bf16(wanted_bf16, exponent).tobytes() + exponent.to_bytes(4, 'little', signed=True)
    plan = prepare_plan(group.precision.request)
    execution = json.loads((compiled/'pointwise/execution_plan.json').read_text())
    output = execution['output_addresses'][0]-DRAM_BASE
    end = output+execution['output_bytes']
    initial = bytearray([0xa5])*((end+4095)//4096*4096+4096)
    for index, codes in enumerate(operands):
        payload = materialize_pointwise_input(plan, index, codes.tobytes())
        payload = np.frombuffer(payload, '<u2').reshape(-1,32,2,16).transpose(0,2,1,3).copy().tobytes()
        at = execution['input_addresses'][index]-DRAM_BASE
        initial[at:at+len(payload)] = payload
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(dram_base=DRAM_BASE, dram_size=len(initial),
                                      numerics='rtl', pack_layout='register_rows')))
    check = unittest.TestCase()

    def execute(memory):
        with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
            directory = Path(directory)
            (directory/'initial.bin').write_bytes(memory)
            subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(compiled/'pointwise/program.bin'),
                '--config', str(config), '--compute', 'cpu', '--load', f'{DRAM_BASE}:{directory/"initial.bin"}',
                '--dump-state', str(directory/'state'), '--max-cycles', '10000000'], check=True, capture_output=True)
            actual = b''.join((directory/'state'/f'dram_{address:016x}.bin').read_bytes()
                             for address in range(DRAM_BASE, DRAM_BASE+len(memory), 4096))
            assert actual[:output] == memory[:output]
            assert actual[end:] == memory[end:]
            for at in range(output, end, 2048):
                assert actual[at+1024:at+2048] == memory[at+1024:at+2048]
            return actual

    invalid = values[0].copy()
    invalid.flat[-1] = np.nan if kind == 'pointwise_narrowing' else 0x7fc0
    check_source_c_call(check, package, initial, (compiled/'pointwise/program.bin').read_bytes(), execute,
        values, wanted, ((invalid, *values[1:]),), source_dtype='opaque', output_dtype='i8', output_exponent=exponent)
