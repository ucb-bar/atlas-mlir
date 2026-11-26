"""Private static quantization preserves the selected softmax codes and guards."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pytest

from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native import softmax_quantize as fused
from merlin.compiler.semantic.model import SemanticNode, pointwise_identity_maps
from test_already_quantized import quantizer_source
from test_pointwise_narrowing import adapter
from test_softmax_narrowing import case as softmax_case

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ('softmax_narrowing', 'softmax_input_widening', 'softmax_quantize')


def case(shape=(33, 65), kind='softmax', exponent=-8, input_dtype='f32'):
    from test_softmax_input_widening import case as widened_case

    source = widened_case(shape, kind) if input_dtype == 'bf16' else softmax_case(shape, kind)
    probability = source.node(source.outputs[0])
    quantizer = quantizer_source(fixed=exponent, count=int(np.prod(shape)))
    codes, scale = quantizer.nodes[-2:]
    codes = replace(codes, inputs=(probability.id,), type=replace(codes.type, shape=shape))
    scale = replace(scale, inputs=(probability.id,))
    return replace(source, nodes=(*source.nodes, codes, scale), outputs=(codes.id, scale.id),
                   output_storages=('external', 'external'))


@pytest.mark.parametrize('input_dtype', ['f32', 'bf16'])
@pytest.mark.parametrize('kind', ['softmax', 'masked_softmax', 'guarded_softmax'])
@pytest.mark.parametrize('shape', [(1, 7), (33, 65), (2, 3, 35)])
def test_selection_retains_the_original_softmax_arithmetic(kind, shape, input_dtype):
    source = case(shape, kind, input_dtype=input_dtype)
    base = adapter()
    group = fused.prepare(source, base, kind, input_dtype=input_dtype)
    selected, = AdjacentDeviceAdapter(base, patterns=PATTERNS).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'softmax_quantize'
    assert group.softmax.precision.source_region.shape == shape
    assert group.exponent == -8


def test_shared_results_and_dynamic_quantizers_keep_separate_regions():
    source = case()
    base = adapter()
    wrapper = AdjacentDeviceAdapter(base, patterns=PATTERNS)
    probability = source.nodes[-3]
    shared = replace(source, outputs=(*source.outputs, probability.id), output_storages=('external',)*3)
    copy = SemanticNode('extra', 'copy', (probability.id,), probability.type,
                        index_maps=pointwise_identity_maps(2, 1))
    reader = replace(source, nodes=(*source.nodes, copy), outputs=(*source.outputs, copy.id),
                     output_storages=('external',)*3)
    for request in (shared, reader):
        assert all(dict(row.metadata)['kind'] != 'softmax_quantize' for row in wrapper.plan(request).selected)
    for key, value in [('mode', 'dynamic'), ('static_exp', True), ('rule', 'unknown')]:
        nodes = tuple(replace(n, attrs=tuple(sorted(dict(n.attrs, **{key: value}).items())))
                      if n.op.startswith('atlas.quantize_act.') else n for n in source.nodes)
        with pytest.raises(ValueError):
            fused.prepare(replace(source, nodes=nodes), base, 'softmax')
    base.activation_quantizer_route = 'host'
    assert all(dict(row.metadata)['kind'] != 'softmax_quantize' for row in wrapper.plan(source).selected)


def test_only_final_stores_are_packed_and_ambiguous_bodies_refuse():
    from atlas_mlir_native.row_output_quantize import final_stores
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    text = '''module {
      func.func @row() -> !atlas.virtual_state {
        %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
        %p = arith.constant 8192 : i32
        %n = arith.constant 2048 : i32
        %x = "atlas.virtual_vpu_splat"() {bits = 0 : i32} : () -> !atlas.virtual_bf16
        %a = "atlas.virtual_dma_store_bf16"(%s,%x,%p,%n) : (!atlas.virtual_state,!atlas.virtual_bf16,i32,i32) -> !atlas.virtual_state
        %b,%y = "atlas.virtual_dma_load_bf16"(%a,%p,%n) : (!atlas.virtual_state,i32,i32) -> (!atlas.virtual_state,!atlas.virtual_bf16)
        %c = "atlas.virtual_dma_store_bf16"(%b,%y,%p,%n) : (!atlas.virtual_state,!atlas.virtual_bf16,i32,i32) -> !atlas.virtual_state
        return %c : !atlas.virtual_state
      }
    }'''
    function = parse_mlir_text(text).body.block.first_op
    selected, = final_stores(function, 8192, 2048)
    assert selected.results[0] == function.body.block.last_op.arguments[0]
    # A final reader needs BF16, so removing/changing its preceding store is unsafe.
    reader = text.replace('return %c', '''%d,%z = "atlas.virtual_dma_load_bf16"(%c,%p,%n) : (!atlas.virtual_state,i32,i32) -> (!atlas.virtual_state,!atlas.virtual_bf16)
        return %d''')
    for changed in (reader, text.replace('constant 2048', 'constant 1024')):
        with pytest.raises(ValueError):
            final_stores(parse_mlir_text(changed).body.block.first_op, 8192, 2048)
    # An unconditional cycle cannot prove completion or a final publication.
    cyclic = text.replace('return %c : !atlas.virtual_state', 'cf.br ^again\n      ^again:\n        cf.br ^again')
    with pytest.raises(ValueError, match='cycle'):
        final_stores(parse_mlir_text(cyclic).body.block.first_op, 8192, 2048)


@unittest.skipUnless(os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                     'select Rust simulator and Atlas compiler')
class QuantizedSoftmaxRuntimeTest(unittest.TestCase):
    def test_generated_c_tail_rows_exact_pack_and_scale(self):
        from atlas_mlir_native.layout import unpack_bf16_pair_halves
        from atlas_mlir_native.row_loop import prepare_plan
        from atlas_mlir_native.row_softmax import prepare_inputs
        from atlas_mlir_native.static_quantize import pack_bf16
        from merlin.compiler.semantic.search.exploration import SearchLimits
        from merlin.compiler.semantic.tiled_softmax import assemble_row_softmax_outputs
        from test_looped_mnk import build_snapshot
        from test_pointwise_loop import check_source_c_call
        from test_tiled_softmax import reference, scores_for

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = build_snapshot(root)
            for shape, exponent, kind, dtype in [
                    ((1, 7), -8, 'softmax', 'f32'), ((33, 65), -8, 'softmax', 'f32'),
                    ((2, 32), -2147483648, 'softmax', 'f32'), ((32, 32), 2147483647, 'softmax', 'f32'),
                    ((33, 65), -7, 'masked_softmax', 'f32'), ((2, 35), -8, 'guarded_softmax', 'bf16'),
                    ((32, 64), -8, 'softmax', 'bf16'), ((2, 35), -8, 'masked_softmax', 'bf16')]:
                with self.subTest(shape=shape, exponent=exponent, kind=kind, dtype=dtype):
                    group = fused.prepare(case(shape, kind, exponent, dtype), adapter(), kind, input_dtype=dtype)
                    compiled = root / f'compiled-{shape[0]}-{exponent}-{kind}-{dtype}'
                    fused.compile_group(group, snapshot, target_source=ROOT, destination=compiled,
                                        limits=SearchLimits(candidate_nodes=256))
                    plan = prepare_plan(group.softmax.precision.request, static_output_exponent=exponent)
                    exported = fused.export_group(group, compiled, 'pointwise', target_source=ROOT)
                    from atlas_mlir_native.row_loop import verify_row_loop

                    # The raw source alone cannot authorize a different result
                    # encoding or scale during replay/export.
                    for wrong in (None, exponent + 1 if exponent < 2147483647 else exponent - 1):
                        with self.assertRaises(ValueError):
                            verify_row_loop(group.softmax.precision.request, compiled / 'row',
                                            target_source=ROOT, static_output_exponent=wrong)
                    package = compiled / 'package'
                    for name, data in exported.files.items():
                        path = package / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                    # The shared runtime is normally linked once by the application.
                    with (package / 'pointwise.c').open('a') as stream:
                        stream.write('\n#include "runtime/atlas_host.c"\n')
                    scores = scores_for(shape, 'split').astype('<u2')
                    values = (scores.astype('<u4') << 16).view('<f4')
                    operands = (scores if dtype == 'bf16' else values,)
                    sanitized, keep = scores.copy(), None
                    if kind != 'softmax':
                        if kind == 'guarded_softmax':
                            mask = np.zeros(shape, dtype='u1')
                            mask[0] = 1
                            mask[1, 1::3] = 1
                        else:
                            mask = np.zeros(shape[-1], dtype='<f4')
                            mask[1::3] = -np.inf
                        keep = np.broadcast_to(mask == 0, shape).copy()
                        sanitized[~keep] = 0xc200
                        operands = (*operands, mask)
                    safe_keep = None if keep is None else keep.copy()
                    if kind == 'guarded_softmax':
                        safe_keep[0, 0] = True
                    probability = assemble_row_softmax_outputs(plan.tiling,
                        [[unpack_bf16_pair_halves(tile) for tile in row]
                         for row in reference(plan.tiling, sanitized, safe_keep)])
                    if kind == 'guarded_softmax':
                        probabilities = np.frombuffer(probability, '<u2').reshape(shape).copy()
                        probabilities[0] = 0
                        probability = probabilities.tobytes()
                    codes = pack_bf16(np.frombuffer(probability, dtype='<u2'), exponent).tobytes()
                    wanted = codes + exponent.to_bytes(4, 'little', signed=True)
                    initial = bytearray([0xa5]) * ((plan.end-plan.io_memory_base+4095)//4096*4096+4096)
                    execution = json.loads((compiled / 'row/body/execution_plan.json').read_text())
                    keep_bytes = None if keep is None else np.where(keep, 0x3f80, 0).astype('<u2').tobytes()
                    for index, inputs in enumerate(prepare_inputs(plan.tiling, sanitized.tobytes(), keep_bytes)):
                        payloads = dict(inputs)
                        for item in json.loads((compiled / 'row/body/manifest.json').read_text()).get('constant_inputs', []):
                            payloads[item['node_id']] = (compiled / 'row/body' / item['path']).read_bytes()
                        leaves = [node.id for node in plan.tiling.regions[index].request.nodes
                                  if node.effect in ('input', 'constant')]
                        for address, key in zip(execution['input_addresses'], leaves, strict=True):
                            at = plan.address(address, index)-plan.io_memory_base
                            initial[at:at+len(payloads[key])] = payloads[key]
                    config = compiled / 'config.json'
                    config.write_text(json.dumps(dict(pack_layout='register_rows',
                        dram_base=plan.io_memory_base, dram_size=len(initial), numerics='rtl')))

                    def execute(memory):
                        with tempfile.TemporaryDirectory(dir=compiled) as directory:
                            directory = Path(directory)
                            (directory / 'initial.bin').write_bytes(memory)
                            subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(compiled / 'row/program.bin'),
                                '--config', str(config), '--compute', 'cpu', '--load',
                                f'{plan.io_memory_base}:{directory / "initial.bin"}', '--dump-state',
                                str(directory / 'state'), '--max-cycles', '10000000'],
                                check=True, capture_output=True)
                            actual = b''.join((directory / 'state' / f'dram_{address:016x}.bin').read_bytes()
                                for address in range(plan.io_memory_base, plan.io_memory_base+len(memory), 4096))
                            self.assertEqual(actual[:plan.writable_base-plan.io_memory_base],
                                             memory[:plan.writable_base-plan.io_memory_base])
                            self.assertEqual(actual[plan.end-plan.io_memory_base:], memory[plan.end-plan.io_memory_base:])
                            return actual

                    invalid = operands[0].copy()
                    invalid[-1, -1] = 0x7fc0 if dtype == 'bf16' else np.nan
                    check_source_c_call(self, package, initial, (compiled / 'row/program.bin').read_bytes(),
                        execute, operands, wanted, ((invalid, *operands[1:]),),
                        source_dtype='opaque', output_dtype='i8', output_exponent=exponent)


def test_packed_softmax_compiles_and_exports_through_native_compile(tmp_path):
    """The fused static-quantizer softmax through native compile and C export, with the selected snapshot."""
    from atlas_mlir_native.options import CodegenOptions, codegen_options
    from atlas_mlir_native.provider_deployment import WINDOW_ADDRESS, WINDOW_BYTES
    from merlin.compiler.semantic.search.exploration import SearchLimits
    from merlin.compiler.semantic.target_binding import open_native_snapshot

    missing = [name for name in ('MERLIN_NATIVE_SNAPSHOT', 'ATLAS_OOT_BIN_DIR') if not os.environ.get(name)]
    assert not missing, 'required prerequisite not selected: ' + ', '.join(missing)
    snapshot = open_native_snapshot(Path(os.environ['MERLIN_NATIVE_SNAPSHOT']))
    with codegen_options(CodegenOptions(io_memory_bytes=WINDOW_BYTES, io_memory_base=WINDOW_ADDRESS,
                                        io_memory_capacity_base=WINDOW_ADDRESS, io_memory_capacity_bytes=WINDOW_BYTES)):
        group = fused.prepare(case((2, 35), 'softmax', -8), adapter(), 'softmax', input_dtype='f32')
        compiled = tmp_path / 'compiled'
        fused.compile_group(group, snapshot, target_source=ROOT, destination=compiled,
                            limits=SearchLimits(candidate_nodes=256))
        exported = fused.export_group(group, compiled, 'pointwise', target_source=ROOT)
    assert any(b'scatter_row_tile' in data for data in exported.files.values()), sorted(exported.files)
