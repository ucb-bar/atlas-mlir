"""Keep pointwise arithmetic and guards while eliminating its private cast."""
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.binding import DRAM_BASE, TARGET_IDENTITY
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.pointwise_narrowing import compile_group, prepare
from merlin.semantic_compiler.model import (
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from test_application_float_arithmetic import policy

ROOT = Path(__file__).resolve().parents[1]


def case(op='mul', shape=(3, 35)):
    dtype = TensorType(shape, 'f32', 'raw-bit-pattern')
    a, b = (SemanticNode(name, 'input', (), dtype, effect='input') for name in ('a', 'b'))
    value = SemanticNode('y', 'f32_' + op, ('a', 'b'), dtype,
        (('numeric_semantics', 'ieee754-f32-' + op + '-rne-v1'),),
        index_maps=pointwise_identity_maps(len(shape), 2))
    cast = SemanticNode('out', 'f32_to_bf16', ('y',), replace(dtype, dtype='bf16'),
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),),
        index_maps=pointwise_identity_maps(len(shape), 1))
    return KernelRequest((a, b, value, cast), ('out',), ('external',),
        (('a', 'external'), ('b', 'external')), TARGET_IDENTITY, 'diagnostic')


def adapter():
    from atlas_mlir_native.already_quantized import POLICY
    selected = {**policy(), "already_quantized": POLICY}
    return AtlasApplicationAdapter(selected, original_source_sha256=selected['source_sha256'])


class PointwiseNarrowingTest(unittest.TestCase):
    def test_real_selector_and_unchanged_selected_operation(self):
        for operation in ('mul', 'div'):
            source = case(operation)
            base = adapter()
            ordinary = base.plan(source).selected
            self.assertEqual(len(ordinary), 2)
            wrapper = AdjacentDeviceAdapter(base, patterns=('pointwise_narrowing',))
            selected, = wrapper.plan(source).selected
            self.assertEqual(dict(selected.metadata)['kind'], 'pointwise_narrowing')
            group = prepare(source, base)
            from atlas_mlir_native.pointwise import prepare_source
            from merlin.compile.semantic_regions import extract_semantic_region
            original = extract_semantic_region(source, ('y',))
            expected = prepare_source(original, allow_math=True)
            self.assertEqual(group.precision.request, expected.request)
            self.assertEqual(group.precision.operand_specs, expected.operand_specs)

    def test_declines_escaping_result_changed_rounding_and_unsupported_policy(self):
        source = case()
        base = adapter()
        wrapper = AdjacentDeviceAdapter(base, patterns=('pointwise_narrowing',))
        escaped = replace(source, outputs=('y', 'out'), output_storages=('external', 'external'))
        self.assertEqual(len(wrapper.plan(escaped).selected), 2)
        cast = source.nodes[-1]
        for changed in (replace(cast, attrs=()), replace(cast, index_maps=()),
                        replace(cast, inputs=('a',)), replace(cast, type=source.nodes[0].type)):
            with self.assertRaises(ValueError):
                prepare(replace(source, nodes=(*source.nodes[:-1], changed)), base)
        with self.assertRaises(ValueError):
            prepare(escaped, base)
        for patterns in (('unknown',), ('pointwise_narrowing', 'pointwise_narrowing')):
            with self.assertRaises(ValueError):
                AdjacentDeviceAdapter(base, patterns=patterns)
        self.assertEqual(len(AdjacentDeviceAdapter(base, patterns=()).plan(source).selected), 2)

    def test_non_nan_bf16_identity_includes_subnormals_infinities_and_signed_zero(self):
        from test_exact_float_device import host_reference, request
        codes = np.arange(65536, dtype='<u2')
        codes = codes[(codes & 0x7fff) <= 0x7f80]
        widened = (codes.astype('<u4') << 16).tobytes()
        actual, status = host_reference(request('f32_to_bf16', len(codes)), [widened])
        self.assertEqual(status, 0)
        self.assertEqual(actual, codes.tobytes())

    @unittest.skipUnless(os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                         'select Rust NPU and native compiler tools')
    def test_generated_mixed_abi_padding_and_invalid_input_publication(self):
        for operation in ('mul', 'div'):
            with self.subTest(operation=operation):
                self.check_generated_call(operation)

    def check_generated_call(self, operation):
        from atlas_mlir_native.pointwise import prepare_plan
        from atlas_mlir_native.pointwise_c import prepare_precision_c_package
        from merlin.semantic_compiler.pointwise_tiling import (
            materialize_pointwise_input,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from test_looped_mnk import build_snapshot
        from test_pointwise_loop import check_source_c_call

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = build_snapshot(root)
            group = prepare(case(operation), adapter())
            target = root / 'compiled'
            compile_group(group, snapshot, target_source=ROOT, destination=target, limits=SearchLimits())
            record, files = prepare_precision_c_package(group.precision, target, 'pointwise',
                target_source=ROOT, runtime_linkage='embedded', output_dtype='bf16')
            self.assertEqual(record['output_dtype'], 'bf16')
            package = root / 'package'
            package.mkdir()
            for name, data in files.items():
                (package / name).write_bytes(data)
            plan = prepare_plan(group.precision.request)
            execution = json.loads((target / 'execution_plan.json').read_text())
            end = execution['output_addresses'][0] + execution['output_bytes'] - DRAM_BASE
            initial = bytearray([0xa5]) * ((end + 4095) // 4096 * 4096)
            a = np.resize(np.array([-2, -0.0, 0, 0.5, 1, 2], dtype='<f4'), (3, 35))
            b = np.full((3, 35), 0.5, dtype='<f4')
            for index, values in enumerate((a, b)):
                codes = (values.view('<u4') >> 16).astype('<u2')
                payload = materialize_pointwise_input(plan, index, codes.tobytes())
                # The source C ABI converts logical row-major tiles to register pairs.
                tiles = np.frombuffer(payload, dtype='<u2').reshape(-1, 32, 32)
                payload = tiles.reshape(-1, 32, 2, 16).transpose(0, 2, 1, 3).copy().tobytes()
                at = execution['input_addresses'][index] - DRAM_BASE
                initial[at:at + len(payload)] = payload
            if operation == 'div':
                from test_softmax_primitives import _reciprocal
                reciprocal = np.array([_reciprocal(0x3f00)], dtype='<u2')
                multiplier = (reciprocal.astype('<u4') << 16).view('<f4')[0]
            else:
                multiplier = np.float32(0.5)
            expected = ((a * multiplier).view('<u4') >> 16).astype('<u2')
            invalid = a.copy()
            invalid[-1, -1] = np.nan
            config = root / 'config.json'
            config.write_text(json.dumps({'compute': 'cpu', 'dram_base': DRAM_BASE, 'dram_size': len(initial)}))

            def execute(memory):
                with tempfile.TemporaryDirectory(dir=root) as work:
                    work = Path(work)
                    (work / 'input.bin').write_bytes(memory)
                    subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(target / 'program.bin'),
                        '--config', str(config), '--load', f'{DRAM_BASE}:{work / "input.bin"}',
                        '--dump-state', str(work / 'state')], check=True, capture_output=True)
                    result = b''.join((work / 'state' / f'dram_{address:016x}.bin').read_bytes()
                        for address in range(DRAM_BASE, DRAM_BASE + len(memory), 4096))
                    first = execution['output_addresses'][0] - DRAM_BASE
                    self.assertEqual(result[:first], memory[:first])
                    self.assertEqual(result[end:], memory[end:])
                    return result

            check_source_c_call(self, package, initial, (target / 'program.bin').read_bytes(), execute,
                (a, b), expected.tobytes(), ((invalid, b),), output_dtype='bf16')


if __name__ == '__main__':
    unittest.main()
